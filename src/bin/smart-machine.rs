use std::time::Duration;

use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::log::EspLogger;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use identity::DeviceId;
use network::{send_device_info, DeviceKind};
use provision::{
    BasicProvision, EspProvisionCapabilityProvider, Provision, ProvisionCapabilityProvider,
    ProvisionError, WifiProvisioningStatus,
};
use smart_glove::runtime_config::load_runtime_config;

const DEVICE_NAME: &str = "SmartMachine-LED";
const DEVICE_KIND: DeviceKind = DeviceKind::Machine;
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
        log::info!("received provisioning payload ({} bytes)", payload.len());
    })));
    provider.set_wifi_status_callback(Some(Box::new(|status| {
        log::info!("provisioning status: {}", describe_provision_status(status));
    })));
    provider.set_error_callback(Some(Box::new(|error| {
        log::error!("provisioning error: {error}");
    })));

    let mut provisioner = BasicProvision::new(provider, ADVERTISING_WINDOW);

    log::info!(
        "starting machine firmware with device info url {}",
        config.device_info_url
    );
    log::info!(
        "machine firmware is ready with device name {} and device id {}",
        DEVICE_NAME,
        device_id
    );

    loop {
        log::info!(
            "opening provisioning window for {} seconds",
            ADVERTISING_WINDOW.as_secs()
        );

        match provisioner.start() {
            Ok(()) => {
                log::info!("provisioning completed successfully");
                break;
            }
            Err(ProvisionError::Timeout) => {
                log::warn!("provisioning timed out");
            }
            Err(err) => {
                log::error!("provisioning failed: {err}");
            }
        }

        FreeRtos::delay_ms(1000u32);
    }

    let events: [&str; 0] = [];
    let status = send_device_info(&config.device_info_url, &device_id, DEVICE_KIND, &events)?;
    log::info!(
        "sent machine device info to server with http status {}",
        status
    );

    Ok(())
}

fn describe_provision_status(status: &WifiProvisioningStatus) -> &'static str {
    match status {
        WifiProvisioningStatus::BroadcastStarting => "starting bluetooth provisioning broadcast",
        WifiProvisioningStatus::Broadcasting => "broadcasting bluetooth provisioning service",
        WifiProvisioningStatus::BroadcastStopped => "stopped bluetooth provisioning broadcast",
        WifiProvisioningStatus::CredentialsReceived => "received wi-fi credentials",
        WifiProvisioningStatus::Connecting => "connecting to wi-fi",
        WifiProvisioningStatus::Connected => "connected to wi-fi",
        WifiProvisioningStatus::ConnectionFailed(_) => "wi-fi connection failed",
    }
}
