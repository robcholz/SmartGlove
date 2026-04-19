use std::io::{Error as IoError, ErrorKind};

const NETWORK_CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/network_runtime_config.env"
));

const DEFAULT_SAMPLE_RATE_HZ: u16 = 100;
const DEFAULT_BATCH_SAMPLES: usize = 10;
const DEFAULT_BATCH_COUNT: usize = 10;
const DEFAULT_FLUSH_INTERVAL_MS: u64 = 100;
const DEFAULT_POST_RUN_DELAY_MS: u32 = 250;

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub device_info_url: String,
    pub status_ws_url: String,
    pub event_name: String,
    pub event_payload_json: String,
    pub sample_rate_hz: u16,
    pub batch_samples: usize,
    pub batch_count: usize,
    pub flush_interval_ms: u64,
    pub post_run_delay_ms: u32,
}

pub fn load_runtime_config() -> Result<RuntimeConfig, IoError> {
    let mut device_info_url = None;
    let mut status_ws_url = None;
    let mut event_name = None;
    let mut event_payload_json = None;
    let mut sample_rate_hz = None;
    let mut batch_samples = None;
    let mut batch_count = None;
    let mut flush_interval_ms = None;
    let mut post_run_delay_ms = None;

    for raw_line in NETWORK_CONFIG.lines() {
        let line = raw_line.trim();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            continue;
        };

        let value = value.trim().to_owned();

        match key.trim() {
            "DEVICE_INFO_URL" => device_info_url = Some(value),
            "STATUS_WS_URL" => status_ws_url = Some(value),
            "EVENT_NAME" => event_name = Some(value),
            "EVENT_PAYLOAD_JSON" => event_payload_json = Some(value),
            "SAMPLE_RATE_HZ" => sample_rate_hz = Some(parse_u16("SAMPLE_RATE_HZ", &value)?),
            "BATCH_SAMPLES" => batch_samples = Some(parse_usize("BATCH_SAMPLES", &value)?),
            "BATCH_COUNT" => batch_count = Some(parse_usize("BATCH_COUNT", &value)?),
            "FLUSH_INTERVAL_MS" => {
                flush_interval_ms = Some(parse_u64("FLUSH_INTERVAL_MS", &value)?)
            }
            "POST_RUN_DELAY_MS" => {
                post_run_delay_ms = Some(parse_u32("POST_RUN_DELAY_MS", &value)?)
            }
            _ => {}
        }
    }

    Ok(RuntimeConfig {
        device_info_url: require_config("DEVICE_INFO_URL", device_info_url)?,
        status_ws_url: require_config("STATUS_WS_URL", status_ws_url)?,
        event_name: require_config("EVENT_NAME", event_name)?,
        event_payload_json: require_config("EVENT_PAYLOAD_JSON", event_payload_json)?,
        sample_rate_hz: sample_rate_hz.unwrap_or(DEFAULT_SAMPLE_RATE_HZ),
        batch_samples: batch_samples.unwrap_or(DEFAULT_BATCH_SAMPLES),
        batch_count: batch_count.unwrap_or(DEFAULT_BATCH_COUNT),
        flush_interval_ms: flush_interval_ms.unwrap_or(DEFAULT_FLUSH_INTERVAL_MS),
        post_run_delay_ms: post_run_delay_ms.unwrap_or(DEFAULT_POST_RUN_DELAY_MS),
    })
}

fn require_config(name: &'static str, value: Option<String>) -> Result<String, IoError> {
    let value = value.ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            format!("missing {name} in tests/network_runtime_config.env"),
        )
    })?;

    if value.is_empty() {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("{name} must not be empty"),
        ));
    }

    Ok(value)
}

fn parse_u16(name: &'static str, value: &str) -> Result<u16, IoError> {
    let parsed = value
        .parse::<u16>()
        .map_err(|err| IoError::new(ErrorKind::InvalidInput, format!("invalid {name}: {err}")))?;
    if parsed == 0 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("{name} must be greater than zero"),
        ));
    }
    Ok(parsed)
}

fn parse_usize(name: &'static str, value: &str) -> Result<usize, IoError> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| IoError::new(ErrorKind::InvalidInput, format!("invalid {name}: {err}")))?;
    if parsed == 0 {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("{name} must be greater than zero"),
        ));
    }
    Ok(parsed)
}

fn parse_u64(name: &'static str, value: &str) -> Result<u64, IoError> {
    value
        .parse::<u64>()
        .map_err(|err| IoError::new(ErrorKind::InvalidInput, format!("invalid {name}: {err}")))
}

fn parse_u32(name: &'static str, value: &str) -> Result<u32, IoError> {
    value
        .parse::<u32>()
        .map_err(|err| IoError::new(ErrorKind::InvalidInput, format!("invalid {name}: {err}")))
}
