from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any

WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


@dataclass
class CapturedTraffic:
    device_info: dict[str, Any] | None = None
    ws_messages: list[dict[str, Any]] = field(default_factory=list)


class MockNetworkServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.actual_port: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.base_events.Server | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._traffic = CapturedTraffic()
        self._condition = threading.Condition()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise TimeoutError("mock network server failed to start")

    def stop(self) -> None:
        if self._loop is None or self._server is None:
            return

        async def shutdown() -> None:
            assert self._server is not None
            self._server.close()
            await self._server.wait_closed()

        future = asyncio.run_coroutine_threadsafe(shutdown(), self._loop)
        future.result(timeout=5.0)
        self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def wait_for_device_info(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._traffic.device_info is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for device info request")
                self._condition.wait(timeout=remaining)
            return self._traffic.device_info

    def wait_for_ws_messages(self, count: int, timeout: float) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self._traffic.ws_messages) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for websocket messages")
                self._condition.wait(timeout=remaining)
            return list(self._traffic.ws_messages)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._ready.set()
        self._loop.run_forever()
        self._loop.close()

    async def _start_server(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port)
        socket_info = self._server.sockets[0].getsockname()
        self.actual_port = int(socket_info[1])

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return

            method, path, _ = request_line.decode("utf-8").strip().split(" ", 2)
            headers = await self._read_headers(reader)

            if method == "POST" and path == "/v1/device-info":
                await self._handle_device_info(reader, writer, headers)
                return

            if method == "GET" and path == "/v1/ws":
                await self._handle_websocket(reader, writer, headers)
                return

            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
        finally:
            if not writer.is_closing():
                writer.close()
                await writer.wait_closed()

    async def _read_headers(self, reader: asyncio.StreamReader) -> dict[str, str]:
        headers: dict[str, str] = {}

        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                return headers
            key, value = line.decode("utf-8").split(":", 1)
            headers[key.strip().lower()] = value.strip()

    async def _handle_device_info(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: dict[str, str],
    ) -> None:
        content_length = int(headers.get("content-length", "0"))
        body = await reader.readexactly(content_length) if content_length else b""
        payload = json.loads(body.decode("utf-8"))

        with self._condition:
            self._traffic.device_info = payload
            self._condition.notify_all()

        response_body = json.dumps({"ok": True}).encode("utf-8")
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(response_body)}\r\n\r\n".encode("utf-8")
            + response_body
        )
        await writer.drain()

    async def _handle_websocket(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: dict[str, str],
    ) -> None:
        key = headers.get("sec-websocket-key")
        if key is None:
            writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            return

        accept = base64.b64encode(
            hashlib.sha1(f"{key}{WEBSOCKET_GUID}".encode("utf-8")).digest()
        ).decode("utf-8")
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode("utf-8")
        writer.write(response)
        await writer.drain()

        while True:
            opcode, payload = await self._read_ws_frame(reader)
            if opcode == 0x8:
                writer.write(self._encode_ws_frame(0x8, payload))
                await writer.drain()
                return
            if opcode == 0x9:
                writer.write(self._encode_ws_frame(0xA, payload))
                await writer.drain()
                continue
            if opcode != 0x1:
                continue

            message = json.loads(payload.decode("utf-8"))
            with self._condition:
                self._traffic.ws_messages.append(message)
                self._condition.notify_all()

    async def _read_ws_frame(self, reader: asyncio.StreamReader) -> tuple[int, bytes]:
        header = await reader.readexactly(2)
        opcode = header[0] & 0x0F
        masked = (header[1] & 0x80) != 0
        length = header[1] & 0x7F

        if length == 126:
            length = struct.unpack("!H", await reader.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", await reader.readexactly(8))[0]

        mask_key = await reader.readexactly(4) if masked else b""
        payload = await reader.readexactly(length)

        if masked:
            payload = bytes(
                byte ^ mask_key[index % 4] for index, byte in enumerate(payload)
            )

        return opcode, payload

    def _encode_ws_frame(self, opcode: int, payload: bytes) -> bytes:
        header = bytearray([0x80 | (opcode & 0x0F)])
        length = len(payload)

        if length < 126:
            header.append(length)
        elif length <= 0xFFFF:
            header.append(126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(127)
            header.extend(struct.pack("!Q", length))

        return bytes(header) + payload


def resolve_bind_host_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if not ip.startswith("127."):
                return ip
        except OSError:
            pass

    hostname = socket.gethostname()
    for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None, socket.AF_INET):
        if family == socket.AF_INET and not sockaddr[0].startswith("127."):
            return sockaddr[0]

    raise RuntimeError("unable to determine a non-loopback host IPv4 address")
