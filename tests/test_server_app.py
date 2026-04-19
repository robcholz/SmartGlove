from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server.app import create_app


class ServerAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()

    def test_device_info_accepts_valid_payload(self) -> None:
        response = self.client.post(
            "/v1/device-info",
            json={
                "device_id": "aca704299de8",
                "events": ["gesture.wave", "battery.low"],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})

    def test_device_info_rejects_invalid_payload_as_bad_request(self) -> None:
        response = self.client.post(
            "/v1/device-info",
            json={"device_id": "BAD-ID", "events": ["gesture.wave"]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["ok"], False)

    def test_device_info_overwrites_latest_metadata(self) -> None:
        first = self.client.post(
            "/v1/device-info",
            json={"device_id": "aca704299de8", "events": ["gesture.wave"]},
        )
        second = self.client.post(
            "/v1/device-info",
            json={"device_id": "aca704299de8", "events": ["battery.low"]},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        snapshot = self.app.state.network_state.snapshot()
        self.assertEqual(snapshot["devices"]["aca704299de8"]["events"], ["battery.low"])

    def test_websocket_accepts_contract_frames(self) -> None:
        self.client.post(
            "/v1/device-info",
            json={"device_id": "aca704299de8", "events": ["battery.low"]},
        )

        with self.client.websocket_connect("/v1/ws") as websocket:
            websocket.send_json({"kind": "device.online", "device_id": "aca704299de8"})
            websocket.send_json(
                {
                    "kind": "device.event",
                    "device_id": "aca704299de8",
                    "event": "battery.low",
                    "payload": {"percent": 11},
                }
            )
            websocket.send_json(
                {
                    "kind": "sensor.batch",
                    "device_id": "aca704299de8",
                    "sample_rate_hz": 100,
                    "start_tick_ms": 128340,
                    "samples": [
                        {
                            "imu_acc": [0.11321, 0.42855, 1.00231],
                            "vel": [0.01340, 0.20441, -0.03210],
                            "flex": {
                                "thumb": 0.41231,
                                "index": 0.53311,
                                "middle": 0.48421,
                                "ring": 0.39200,
                                "pinky": 0.31541,
                            },
                        }
                    ],
                }
            )

        snapshot = self.app.state.network_state.snapshot()
        self.assertEqual(len(snapshot["recent_frames"]), 3)
        self.assertFalse(snapshot["devices"]["aca704299de8"]["connected"])

    def test_network_state_marks_online_transition_once(self) -> None:
        self.client.post(
            "/v1/device-info",
            json={"device_id": "aca704299de8", "events": ["battery.low"]},
        )

        with self.client.websocket_connect("/v1/ws") as websocket:
            websocket.send_json({"kind": "device.online", "device_id": "aca704299de8"})
            websocket.send_json({"kind": "device.online", "device_id": "aca704299de8"})

        snapshot = self.app.state.network_state.snapshot()
        self.assertFalse(snapshot["devices"]["aca704299de8"]["connected"])

    def test_websocket_rejects_invalid_frame(self) -> None:
        with self.assertRaises(WebSocketDisconnect) as context:
            with self.client.websocket_connect("/v1/ws") as websocket:
                websocket.send_json(
                    {"kind": "device.online", "device_id": "not-a-device-id"}
                )
                websocket.receive_text()

        self.assertEqual(context.exception.code, 1008)


if __name__ == "__main__":
    unittest.main()
