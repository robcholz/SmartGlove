pub mod protocol;
pub mod status_stream;

pub use protocol::{send_device_info, DeviceInfo, DeviceKind};
pub use status_stream::{
    connect_status_reporter, mock_sensor_sample, FlexReadings, SensorSample, StatusReportConfig,
    StatusReporter,
};

use core::fmt;

#[derive(Debug)]
pub enum NetworkError {
    InvalidConfig(&'static str),
    Http(esp_idf_svc::sys::EspError),
    Io(esp_idf_svc::io::EspIOError),
    HttpStatus(u16),
    WebSocket(esp_idf_svc::sys::EspError),
    WebSocketNotConnected,
    InvalidEventPayload,
    Serialize(serde_json::Error),
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => f.write_str(message),
            Self::Http(err) => write!(f, "http request failed: {err}"),
            Self::Io(err) => write!(f, "i/o failed: {err}"),
            Self::HttpStatus(status) => write!(f, "unexpected http status: {status}"),
            Self::WebSocket(err) => write!(f, "websocket operation failed: {err}"),
            Self::WebSocketNotConnected => f.write_str("websocket client is not connected"),
            Self::InvalidEventPayload => f.write_str("event payload must be valid JSON"),
            Self::Serialize(err) => write!(f, "json serialization failed: {err}"),
        }
    }
}

impl std::error::Error for NetworkError {}

impl From<esp_idf_svc::sys::EspError> for NetworkError {
    fn from(value: esp_idf_svc::sys::EspError) -> Self {
        Self::Http(value)
    }
}

impl From<esp_idf_svc::io::EspIOError> for NetworkError {
    fn from(value: esp_idf_svc::io::EspIOError) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for NetworkError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialize(value)
    }
}
