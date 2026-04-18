use core::fmt;

use crate::identity::DeviceIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistrationConfig {
    pub endpoint: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistrationRequest {
    pub device_id: String,
    pub device_secret: String,
}

impl RegistrationRequest {
    pub fn from_identity(identity: &DeviceIdentity) -> Self {
        Self {
            device_id: identity.device_id().to_hex_string(),
            device_secret: identity.device_secret().to_hex_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistrationResponse {
    pub registered: bool,
    pub session_token: String,
    pub websocket_url: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistrationError {
    Transport,
    InvalidResponse,
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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

    log::info!(
        "prepared registration request for device_id={} endpoint={}",
        request.device_id,
        config.endpoint
    );

    Err(RegistrationError::Transport)
}
