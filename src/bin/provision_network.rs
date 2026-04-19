use std::io::{Error as IoError, ErrorKind};
use std::time::{Duration, Instant};

use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::log::EspLogger;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use identity::DeviceId;
use network::{connect_status_reporter, mock_sensor_sample, send_device_info, StatusReportConfig};
use provision::{
    BasicProvision, EspProvisionCapabilityProvider, Provision, ProvisionCapabilityProvider,
    ProvisionError,
};

const DEVICE_NAME: &str = "SmartGlove Provision";
const ADVERTISING_WINDOW: Duration = Duration::from_secs(10);
const NETWORK_CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/network_runtime_config.env"
));
const DEFAULT_SAMPLE_RATE_HZ: u16 = 100;
const DEFAULT_BATCH_SAMPLES: usize = 10;
const DEFAULT_BATCH_COUNT: usize = 10;
const DEFAULT_FLUSH_INTERVAL_MS: u64 = 100;
const DEFAULT_POST_RUN_DELAY_MS: u32 = 250;

struct RuntimeConfig {
    device_info_url: String,
    status_ws_url: String,
    event_name: String,
    event_payload_json: String,
    sample_rate_hz: u16,
    batch_samples: usize,
    batch_count: usize,
    flush_interval_ms: u64,
    post_run_delay_ms: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    esp_idf_svc::sys::link_patches();
    EspLogger::initialize_default();

    let config = load_runtime_config()?;
    let device_id = DeviceId::from_factory_mac()?.to_hex_string();

    let peripherals = Peripherals::take()?;
    let sysloop = EspSystemEventLoop::take()?;
    let nvs = EspDefaultNvsPartition::take()?;
    let (wifi_modem, bluetooth_modem) = peripherals.modem.split();

    let mut provider = EspProvisionCapabilityProvider::new(
        DEVICE_NAME,
        wifi_modem,
        bluetooth_modem,
        sysloop,
        Some(nvs),
    )?;

    provider.set_ble_message_callback(Some(Box::new(|payload| {
        log::info!(
            "PROVISION_RX bytes={} payload={}",
            payload.len(),
            String::from_utf8_lossy(payload)
        );
    })));
    provider.set_wifi_status_callback(Some(Box::new(|status| {
        log::info!("PROVISION_STATUS {:?}", status);
    })));
    provider.set_error_callback(Some(Box::new(|error| {
        log::error!("PROVISION_ERROR {error}");
    })));

    let mut provisioner = BasicProvision::new(provider, ADVERTISING_WINDOW);

    log::info!(
        "PROVISION_NETWORK_READY name={} device_id={} device_info_url={} status_ws_url={}",
        DEVICE_NAME,
        device_id,
        config.device_info_url,
        config.status_ws_url
    );

    loop {
        log::info!(
            "PROVISION_WINDOW_OPEN secs={}",
            ADVERTISING_WINDOW.as_secs()
        );

        match provisioner.start() {
            Ok(()) => {
                log::info!("PROVISION_DONE success");
                break;
            }
            Err(ProvisionError::Timeout) => {
                log::warn!("PROVISION_DONE timeout");
            }
            Err(err) => {
                log::error!("PROVISION_DONE error={err}");
            }
        }

        FreeRtos::delay_ms(1000u32);
    }

    let events = [config.event_name.as_str()];
    let status = send_device_info(&config.device_info_url, &device_id, &events)?;
    log::info!("NETWORK_DEVICE_INFO_SENT status={status}");

    let status_config = StatusReportConfig {
        websocket_url: &config.status_ws_url,
        sample_rate_hz: config.sample_rate_hz,
        batch_samples: config.batch_samples,
        flush_interval: Duration::from_millis(config.flush_interval_ms),
    };
    let mut reporter = connect_status_reporter(&status_config, &device_id)?;
    log::info!("NETWORK_WS_CONNECTED");

    reporter.send_online()?;
    log::info!("NETWORK_ONLINE_SENT");

    reporter.send_event(&config.event_name, &config.event_payload_json)?;
    log::info!("NETWORK_EVENT_SENT name={}", config.event_name);

    let send_started_at = Instant::now();
    let mut sequence = 0u64;
    for batch_index in 0..config.batch_count {
        for _ in 0..config.batch_samples {
            let _ = reporter.push_sensor_sample(mock_sensor_sample(sequence))?;
            sequence += 1;

            let sample_delay_ms = 1000u32 / u32::from(config.sample_rate_hz);
            if sample_delay_ms > 0 {
                FreeRtos::delay_ms(sample_delay_ms);
            }
        }

        let _ = reporter.flush()?;
        log::info!(
            "NETWORK_BATCH_SENT batch_index={} samples={}",
            batch_index + 1,
            config.batch_samples
        );
    }

    log::info!(
        "NETWORK_STREAM_DONE sample_rate_hz={} batch_samples={} batches={} total_samples={} elapsed_ms={}",
        config.sample_rate_hz,
        config.batch_samples,
        config.batch_count,
        config.batch_count * config.batch_samples,
        send_started_at.elapsed().as_millis()
    );

    if config.post_run_delay_ms > 0 {
        FreeRtos::delay_ms(config.post_run_delay_ms);
    }

    Ok(())
}

fn load_runtime_config() -> Result<RuntimeConfig, IoError> {
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
