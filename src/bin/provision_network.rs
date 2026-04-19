use std::time::{Duration, Instant};

use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::log::EspLogger;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use identity::DeviceId;
use network::{
    connect_status_reporter, mock_sensor_sample, send_device_info, DeviceKind, StatusReportConfig,
};
use provision::{
    BasicProvision, EspProvisionCapabilityProvider, Provision, ProvisionCapabilityProvider,
    ProvisionError,
};
use smart_glove::runtime_config::load_runtime_config;

const DEVICE_NAME: &str = "SmartGlove Provision";
const DEVICE_KIND: DeviceKind = DeviceKind::Glove;
const ADVERTISING_WINDOW: Duration = Duration::from_secs(10);

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
    let status = send_device_info(&config.device_info_url, &device_id, DEVICE_KIND, &events)?;
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
