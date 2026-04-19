use esp_idf_svc::http::{client::EspHttpConnection, Method};
use identity::DeviceIdentity;
use serde::Serialize;

use crate::NetworkError;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    Glove,
    Machine,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceInfo<'a> {
    pub device_id: &'a str,
    pub kind: DeviceKind,
    pub events: &'a [&'a str],
}

#[derive(Serialize)]
struct DeviceInfoPayload<'a> {
    device_id: &'a str,
    kind: DeviceKind,
    events: &'a [&'a str],
}

#[derive(Serialize)]
struct OnlinePayload<'a> {
    kind: &'static str,
    device_id: &'a str,
}

#[derive(Serialize)]
struct EventPayload<'a> {
    kind: &'static str,
    device_id: &'a str,
    event: &'a str,
    payload: serde_json::Value,
}

impl<'a> DeviceInfo<'a> {
    pub fn from_identity(
        identity: &'a DeviceIdentity,
        kind: DeviceKind,
        events: &'a [&'a str],
    ) -> Self {
        Self {
            device_id: Box::leak(identity.device_id().to_hex_string().into_boxed_str()),
            kind,
            events,
        }
    }
}

pub fn send_device_info(
    endpoint: &str,
    device_id: &str,
    kind: DeviceKind,
    events: &[&str],
) -> Result<u16, NetworkError> {
    if endpoint.is_empty() {
        return Err(NetworkError::InvalidConfig(
            "device info endpoint must not be empty",
        ));
    }

    let body = serde_json::to_string(&DeviceInfoPayload {
        device_id,
        kind,
        events,
    })?;
    let content_length = body.len().to_string();
    let headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", content_length.as_str()),
    ];

    let mut connection = EspHttpConnection::new(&Default::default()).map_err(NetworkError::Http)?;

    connection
        .initiate_request(Method::Post, endpoint, &headers)
        .map_err(NetworkError::Http)?;
    connection
        .write_all(body.as_bytes())
        .map_err(NetworkError::Http)?;
    connection.initiate_response().map_err(NetworkError::Http)?;

    let status = connection.status();

    if !(200..300).contains(&status) {
        return Err(NetworkError::HttpStatus(status));
    }

    Ok(status)
}

pub fn build_online_json(device_id: &str) -> String {
    serde_json::to_string(&OnlinePayload {
        kind: "device.online",
        device_id,
    })
    .expect("online json")
}

pub fn build_event_json(
    device_id: &str,
    event_name: &str,
    payload_json: &str,
) -> Result<String, NetworkError> {
    let payload =
        serde_json::from_str(payload_json).map_err(|_| NetworkError::InvalidEventPayload)?;

    serde_json::to_string(&EventPayload {
        kind: "device.event",
        device_id,
        event: event_name,
        payload,
    })
    .map_err(NetworkError::from)
}
