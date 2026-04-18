use core::fmt;

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
    InvalidResponse,
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest => f.write_str("registration request was invalid"),
            Self::Transport => f.write_str("registration transport failed"),
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

    log::info!(
        "prepared registration request for device_id={} endpoint={} body_len={}",
        request.device_id,
        config.endpoint,
        request_json.len()
    );

    Err(RegistrationError::Transport)
}
