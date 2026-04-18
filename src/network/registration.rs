use core::fmt;
use std::time::Duration;

use embedded_svc::http::client::Client;
use embedded_svc::io::{Read, Write};
use esp_idf_svc::http::client::{Configuration as HttpConfiguration, EspHttpConnection};
use serde::{Deserialize, Serialize};

use crate::identity::DeviceIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistrationConfig {
    pub endpoint: &'static str,
}

#[derive(Clone, Eq, PartialEq, Serialize)]
pub struct RegistrationRequest {
    pub device_id: String,
    pub device_secret: String,
}

impl fmt::Debug for RegistrationRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegistrationRequest")
            .field("device_id", &self.device_id)
            .field("device_secret", &"<redacted>")
            .finish()
    }
}

impl RegistrationRequest {
    pub fn from_identity(identity: &DeviceIdentity) -> Self {
        Self {
            device_id: identity.device_id().to_hex_string(),
            device_secret: identity.device_secret().to_hex_string(),
        }
    }

    pub fn to_json(&self) -> Result<String, RegistrationError> {
        serde_json::to_string(self).map_err(|_| RegistrationError::InvalidRequest)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct RegistrationResponse {
    pub registered: bool,
    pub session_token: String,
    pub websocket_url: String,
}

impl RegistrationResponse {
    pub fn from_json(json: &str) -> Result<Self, RegistrationError> {
        serde_json::from_str(json).map_err(|_| RegistrationError::InvalidResponse)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistrationError {
    InvalidRequest,
    Transport,
    UnexpectedStatus(u16),
    InvalidResponse,
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest => f.write_str("registration request was invalid"),
            Self::Transport => f.write_str("registration transport failed"),
            Self::UnexpectedStatus(status) => {
                write!(f, "registration server returned HTTP status {status}")
            }
            Self::InvalidResponse => f.write_str("registration response was invalid"),
        }
    }
}

impl std::error::Error for RegistrationError {}

pub fn register_device(
    identity: &DeviceIdentity,
    config: &RegistrationConfig,
) -> Result<RegistrationResponse, RegistrationError> {
    let request = RegistrationRequest::from_identity(identity);
    let request_json = request.to_json()?;
    let content_length = request_json.len().to_string();

    log::info!(
        "prepared registration request for device_id={} endpoint={} body_len={}",
        request.device_id,
        config.endpoint,
        request_json.len()
    );

    let connection = EspHttpConnection::new(&HttpConfiguration {
        timeout: Some(Duration::from_secs(10)),
        ..Default::default()
    })
    .map_err(|_| RegistrationError::Transport)?;
    let mut client = Client::wrap(connection);
    let headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", content_length.as_str()),
    ];

    let mut http_request = client
        .post(config.endpoint, &headers)
        .map_err(|_| RegistrationError::Transport)?;
    write_all(&mut http_request, request_json.as_bytes())?;
    let mut http_response = http_request
        .submit()
        .map_err(|_| RegistrationError::Transport)?;

    let status = http_response.status();
    if !(200..300).contains(&status) {
        return Err(RegistrationError::UnexpectedStatus(status));
    }

    let response_json = read_to_string(&mut http_response)?;
    RegistrationResponse::from_json(&response_json)
}

fn write_all<W>(writer: &mut W, mut bytes: &[u8]) -> Result<(), RegistrationError>
where
    W: Write,
{
    while !bytes.is_empty() {
        let written = writer
            .write(bytes)
            .map_err(|_| RegistrationError::Transport)?;
        if written == 0 {
            return Err(RegistrationError::Transport);
        }
        bytes = &bytes[written..];
    }

    writer.flush().map_err(|_| RegistrationError::Transport)
}

fn read_to_string<R>(reader: &mut R) -> Result<String, RegistrationError>
where
    R: Read,
{
    let mut body = Vec::new();
    let mut buffer = [0_u8; 512];

    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|_| RegistrationError::Transport)?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&buffer[..read]);
    }

    String::from_utf8(body).map_err(|_| RegistrationError::InvalidResponse)
}
