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
                "kind": "glove",
                "events": ["gesture.wave", "battery.low"],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})

    def test_device_info_rejects_invalid_payload_as_bad_request(self) -> None:
        response = self.client.post(
            "/v1/device-info",
            json={"device_id": "BAD-ID", "kind": "glove", "events": ["gesture.wave"]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["ok"], False)

    def test_device_info_overwrites_latest_metadata(self) -> None:
        first = self.client.post(
            "/v1/device-info",
            json={
                "device_id": "aca704299de8",
                "kind": "glove",
                "events": ["gesture.wave"],
            },
        )
        second = self.client.post(
            "/v1/device-info",
            json={
                "device_id": "aca704299de8",
                "kind": "machine",
                "events": ["battery.low"],
            },
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        snapshot = self.app.state.network_state.snapshot()
        self.assertEqual(snapshot["devices"]["aca704299de8"]["kind"], "machine")
        self.assertEqual(snapshot["devices"]["aca704299de8"]["events"], ["battery.low"])

    def test_websocket_accepts_contract_frames(self) -> None:
        self.client.post(
            "/v1/device-info",
            json={
                "device_id": "aca704299de8",
                "kind": "glove",
                "events": ["battery.low"],
            },
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

    def test_mapping_routes_glove_event_to_online_machine(self) -> None:
        self.client.post(
            "/v1/device-info",
            json={
                "device_id": "aca704299de8",
                "kind": "glove",
                "events": ["event.infer.waving", "event.infer.none"],
            },
        )
        self.client.post(
            "/v1/device-info",
            json={
                "device_id": "e072a1a9d114",
                "kind": "machine",
                "events": ["hw.rgb_green", "hw.rgb_off"],
            },
        )
        mapping_response = self.client.post(
            "/v1/gloves/aca704299de8/mappings",
            json={
                "source_event": "event.infer.waving",
                "target_machine_id": "e072a1a9d114",
                "target_event": "hw.rgb_green",
            },
        )
        self.assertEqual(mapping_response.status_code, 201)

        with self.client.websocket_connect("/v1/machine-ws") as machine_socket:
            machine_socket.send_json(
                {"kind": "machine.online", "device_id": "e072a1a9d114"}
            )
            with self.client.websocket_connect("/v1/ws") as glove_socket:
                glove_socket.send_json(
                    {"kind": "device.online", "device_id": "aca704299de8"}
                )
                glove_socket.send_json(
                    {
                        "kind": "device.event",
                        "device_id": "aca704299de8",
                        "event": "event.infer.waving",
                        "payload": {"score": 0.9},
                    }
                )

            trigger = machine_socket.receive_json()

        self.assertEqual(trigger["kind"], "machine.trigger")
        self.assertEqual(trigger["device_id"], "e072a1a9d114")
        self.assertEqual(trigger["event"], "hw.rgb_green")
        self.assertEqual(trigger["source_device_id"], "aca704299de8")
        self.assertEqual(trigger["source_event"], "event.infer.waving")
        self.assertEqual(trigger["payload"], {"score": 0.9})

    def test_mapping_endpoints_list_and_delete_mappings(self) -> None:
        self.client.post(
            "/v1/device-info",
            json={
                "device_id": "aca704299de8",
                "kind": "glove",
                "events": ["event.infer.thumb-up"],
            },
        )
        self.client.post(
            "/v1/device-info",
            json={
                "device_id": "feedfacecafe",
                "kind": "machine",
                "events": ["hw.motor_on"],
            },
        )

        create_response = self.client.post(
            "/v1/gloves/aca704299de8/mappings",
            json={
                "source_event": "event.infer.thumb-up",
                "target_machine_id": "feedfacecafe",
                "target_event": "hw.motor_on",
            },
        )
        self.assertEqual(create_response.status_code, 201)
        mapping_id = create_response.json()["id"]

        list_response = self.client.get("/v1/gloves/aca704299de8/mappings")
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()["items"]), 1)
        self.assertEqual(list_response.json()["items"][0]["id"], mapping_id)

        delete_response = self.client.delete(
            f"/v1/gloves/aca704299de8/mappings/{mapping_id}"
        )
        self.assertEqual(delete_response.status_code, 204)

        list_after_delete = self.client.get("/v1/gloves/aca704299de8/mappings")
        self.assertEqual(list_after_delete.status_code, 200)
        self.assertEqual(list_after_delete.json()["items"], [])

    def test_mapping_rejects_undeclared_events(self) -> None:
        self.client.post(
            "/v1/device-info",
            json={
                "device_id": "aca704299de8",
                "kind": "glove",
                "events": ["event.infer.thumb-up"],
            },
        )
        self.client.post(
            "/v1/device-info",
            json={
                "device_id": "feedfacecafe",
                "kind": "machine",
                "events": ["hw.motor_on"],
            },
        )

        response = self.client.post(
            "/v1/gloves/aca704299de8/mappings",
            json={
                "source_event": "event.infer.waving",
                "target_machine_id": "feedfacecafe",
                "target_event": "hw.motor_on",
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_machine_websocket_rejects_unregistered_machine(self) -> None:
        with self.assertRaises(WebSocketDisconnect) as context:
            with self.client.websocket_connect("/v1/machine-ws") as websocket:
                websocket.send_json(
                    {"kind": "machine.online", "device_id": "feedfacecafe"}
                )
                websocket.receive_text()

        self.assertEqual(context.exception.code, 1008)

    def test_network_state_marks_online_transition_once(self) -> None:
        self.client.post(
            "/v1/device-info",
            json={
                "device_id": "aca704299de8",
                "kind": "glove",
                "events": ["battery.low"],
            },
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
