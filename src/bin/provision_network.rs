use std::io::{Error as IoError, ErrorKind};
use std::time::Duration;

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
const SENSOR_BATCH_SAMPLES: usize = 10;
const SENSOR_SAMPLE_RATE_HZ: u16 = 100;
const SENSOR_FLUSH_INTERVAL: Duration = Duration::from_millis(100);

struct RuntimeConfig {
    device_info_url: String,
    status_ws_url: String,
    event_name: String,
    event_payload_json: String,
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
        sample_rate_hz: SENSOR_SAMPLE_RATE_HZ,
        batch_samples: SENSOR_BATCH_SAMPLES,
        flush_interval: SENSOR_FLUSH_INTERVAL,
    };
    let mut reporter = connect_status_reporter(&status_config, &device_id)?;
    log::info!("NETWORK_WS_CONNECTED");

    reporter.send_online()?;
    log::info!("NETWORK_ONLINE_SENT");

    reporter.send_event(&config.event_name, &config.event_payload_json)?;
    log::info!("NETWORK_EVENT_SENT name={}", config.event_name);

    for sample_index in 0..SENSOR_BATCH_SAMPLES {
        let _ = reporter.push_sensor_sample(mock_sensor_sample(sample_index as u64))?;
        FreeRtos::delay_ms(10u32);
    }

    let _ = reporter.flush()?;
    log::info!("NETWORK_BATCH_SENT samples={SENSOR_BATCH_SAMPLES}");

    loop {
        FreeRtos::delay_ms(10_000u32);
    }
}

fn load_runtime_config() -> Result<RuntimeConfig, IoError> {
    let mut device_info_url = None;
    let mut status_ws_url = None;
    let mut event_name = None;
    let mut event_payload_json = None;

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
            _ => {}
        }
    }

    Ok(RuntimeConfig {
        device_info_url: require_config("DEVICE_INFO_URL", device_info_url)?,
        status_ws_url: require_config("STATUS_WS_URL", status_ws_url)?,
        event_name: require_config("EVENT_NAME", event_name)?,
        event_payload_json: require_config("EVENT_PAYLOAD_JSON", event_payload_json)?,
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
