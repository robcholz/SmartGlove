use core::fmt;
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use esp_idf_svc::ws::{
    client::{
        EspWebSocketClient, EspWebSocketClientConfig, EspWebSocketTransport, WebSocketEventType,
    },
    FrameType,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

type Callback = Box<dyn FnMut(&MachineTriggerFrame) -> Result<Option<Value>, String> + Send>;

#[derive(Debug)]
pub enum RemoteControlError {
    InvalidConfig(&'static str),
    Io(esp_idf_svc::io::EspIOError),
    WebSocket(esp_idf_svc::sys::EspError),
    WebSocketNotConnected,
    Json(serde_json::Error),
}

impl fmt::Display for RemoteControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => f.write_str(message),
            Self::Io(err) => write!(f, "i/o failed: {err}"),
            Self::WebSocket(err) => write!(f, "websocket operation failed: {err}"),
            Self::WebSocketNotConnected => f.write_str("websocket client is not connected"),
            Self::Json(err) => write!(f, "json failed: {err}"),
        }
    }
}

impl std::error::Error for RemoteControlError {}

impl From<esp_idf_svc::io::EspIOError> for RemoteControlError {
    fn from(value: esp_idf_svc::io::EspIOError) -> Self {
        Self::Io(value)
    }
}

impl From<esp_idf_svc::sys::EspError> for RemoteControlError {
    fn from(value: esp_idf_svc::sys::EspError) -> Self {
        Self::WebSocket(value)
    }
}

impl From<serde_json::Error> for RemoteControlError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RemoteControlConfig<'a> {
    pub websocket_url: &'a str,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct MachineTriggerFrame {
    pub kind: String,
    pub request_id: String,
    pub device_id: String,
    pub event: String,
    pub source_device_id: String,
    pub source_event: String,
    pub payload: Value,
}

#[derive(Serialize)]
struct MachineOnlineFrame<'a> {
    kind: &'static str,
    device_id: &'a str,
}

#[derive(Serialize)]
struct MachineResultFrame<'a> {
    kind: &'static str,
    device_id: &'a str,
    request_id: &'a str,
    status: &'a str,
    payload: Value,
}

pub struct RemoteControlClient {
    device_id: String,
    websocket: EspWebSocketClient<'static>,
    incoming: Receiver<String>,
    callbacks: HashMap<String, Callback>,
}

pub fn derive_machine_ws_url(status_ws_url: &str) -> Result<String, RemoteControlError> {
    let Some(prefix) = status_ws_url.strip_suffix("/v1/ws") else {
        return Err(RemoteControlError::InvalidConfig(
            "status websocket url must end with /v1/ws",
        ));
    };

    Ok(format!("{prefix}/v1/machine-ws"))
}

pub fn connect_remote_control(
    config: &RemoteControlConfig<'_>,
    device_id: &str,
) -> Result<RemoteControlClient, RemoteControlError> {
    if config.websocket_url.is_empty() {
        return Err(RemoteControlError::InvalidConfig(
            "websocket url must not be empty",
        ));
    }

    let transport = if config.websocket_url.starts_with("wss://") {
        EspWebSocketTransport::TransportOverSSL
    } else {
        EspWebSocketTransport::TransportOverTCP
    };

    let client_config = EspWebSocketClientConfig {
        transport,
        buffer_size: 4096,
        network_timeout_ms: Duration::from_secs(10),
        reconnect_timeout_ms: Duration::from_secs(5),
        ping_interval_sec: Duration::from_secs(30),
        pingpong_timeout_sec: Duration::from_secs(10),
        ..Default::default()
    };

    let (tx, rx) = mpsc::channel();
    let websocket = EspWebSocketClient::new(
        config.websocket_url,
        &client_config,
        Duration::from_secs(10),
        move |event| match event {
            Ok(event) => match event.event_type {
                WebSocketEventType::Text(text) => {
                    if tx.send(text.to_owned()).is_err() {
                        log::warn!("remote-control text frame dropped because receiver closed");
                    }
                }
                WebSocketEventType::Connected => log::info!("remote-control websocket connected"),
                WebSocketEventType::Disconnected => {
                    log::warn!("remote-control websocket disconnected")
                }
                WebSocketEventType::Close(reason) => {
                    log::info!("remote-control websocket close: {reason:?}")
                }
                WebSocketEventType::Closed => log::info!("remote-control websocket closed"),
                WebSocketEventType::BeforeConnect => {
                    log::debug!("remote-control websocket before connect")
                }
                WebSocketEventType::Binary(_) => {
                    log::warn!("remote-control websocket ignored unexpected binary frame")
                }
                WebSocketEventType::Ping => log::debug!("remote-control websocket ping"),
                WebSocketEventType::Pong => log::debug!("remote-control websocket pong"),
            },
            Err(err) => log::warn!("remote-control websocket callback error: {err}"),
        },
    )?;

    wait_until_connected(&websocket)?;

    Ok(RemoteControlClient {
        device_id: device_id.to_owned(),
        websocket,
        incoming: rx,
        callbacks: HashMap::new(),
    })
}

impl RemoteControlClient {
    pub fn register<F>(&mut self, event_name: &str, callback: F)
    where
        F: FnMut(&MachineTriggerFrame) -> Result<Option<Value>, String> + Send + 'static,
    {
        self.callbacks
            .insert(event_name.to_owned(), Box::new(callback));
    }

    pub fn send_online(&mut self) -> Result<(), RemoteControlError> {
        let payload = serde_json::to_string(&MachineOnlineFrame {
            kind: "machine.online",
            device_id: &self.device_id,
        })?;
        self.send_text_json(&payload)
    }

    pub fn poll(&mut self) -> Result<usize, RemoteControlError> {
        let mut handled = 0usize;

        while let Ok(text) = self.incoming.try_recv() {
            let trigger: MachineTriggerFrame = serde_json::from_str(&text)?;
            if trigger.kind != "machine.trigger" {
                log::warn!(
                    "remote-control ignored unexpected frame kind {}",
                    trigger.kind
                );
                continue;
            }

            let result = if let Some(callback) = self.callbacks.get_mut(&trigger.event) {
                callback(&trigger)
            } else {
                Err(format!("no callback registered for {}", trigger.event))
            };

            match result {
                Ok(payload) => {
                    self.send_result(&trigger.request_id, "ok", payload.unwrap_or(Value::Null))?;
                    handled += 1;
                }
                Err(message) => {
                    log::error!(
                        "remote-control action {} failed: {}",
                        trigger.event,
                        message
                    );
                    self.send_result(
                        &trigger.request_id,
                        "error",
                        serde_json::json!({ "message": message }),
                    )?;
                    handled += 1;
                }
            }
        }

        Ok(handled)
    }

    fn send_result(
        &mut self,
        request_id: &str,
        status: &str,
        payload: Value,
    ) -> Result<(), RemoteControlError> {
        let payload = serde_json::to_string(&MachineResultFrame {
            kind: "machine.result",
            device_id: &self.device_id,
            request_id,
            status,
            payload,
        })?;
        self.send_text_json(&payload)
    }

    fn send_text_json(&mut self, payload: &str) -> Result<(), RemoteControlError> {
        if !self.websocket.is_connected() {
            return Err(RemoteControlError::WebSocketNotConnected);
        }

        self.websocket
            .send(FrameType::Text(false), payload.as_bytes())
            .map_err(RemoteControlError::WebSocket)?;

        Ok(())
    }
}

fn wait_until_connected(websocket: &EspWebSocketClient<'static>) -> Result<(), RemoteControlError> {
    for _ in 0..20 {
        if websocket.is_connected() {
            return Ok(());
        }

        thread::sleep(Duration::from_millis(100));
    }

    Err(RemoteControlError::WebSocketNotConnected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_machine_ws_url_from_status_ws_url() {
        let url = derive_machine_ws_url("ws://10.0.0.1:8000/v1/ws").expect("derive url");
        assert_eq!(url, "ws://10.0.0.1:8000/v1/machine-ws");
    }

    #[test]
    fn parses_trigger_frame() {
        let json = r#"{
            "kind":"machine.trigger",
            "request_id":"req_123",
            "device_id":"feedfacecafe",
            "event":"hw.rgb_green",
            "source_device_id":"aca704299de8",
            "source_event":"event.infer.wave",
            "payload":null
        }"#;

        let parsed: MachineTriggerFrame = serde_json::from_str(json).expect("parse trigger");
        assert_eq!(parsed.event, "hw.rgb_green");
        assert_eq!(parsed.payload, Value::Null);
    }
}
