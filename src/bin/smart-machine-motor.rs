use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{Output, PinDriver};
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::log::EspLogger;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use identity::DeviceId;
use network::{send_device_info, DeviceKind};
use provision::{
    BasicProvision, EspProvisionCapabilityProvider, Provision, ProvisionCapabilityProvider,
    ProvisionError, WifiProvisioningStatus,
};
use remote_control::{
    connect_remote_control, derive_machine_ws_url, RemoteControlClient, RemoteControlConfig,
};
use smart_glove::runtime_config::load_runtime_config;

const DEVICE_NAME: &str = "SmartMachine-Motor";
const DEVICE_KIND: DeviceKind = DeviceKind::Machine;
const ADVERTISING_WINDOW: Duration = Duration::from_secs(10);
const MOTOR_PIN: i32 = 6;
const SUPPORTED_EVENTS: [&str; 2] = ["hw.motor_on", "hw.motor_off"];
const REMOTE_CONTROL_POLL_INTERVAL_MS: u64 = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    esp_idf_svc::sys::link_patches();
    EspLogger::initialize_default();

    let config = load_runtime_config()?;
    let machine_ws_url = derive_machine_ws_url(&config.status_ws_url)?;
    let device_id = DeviceId::from_factory_mac()?.to_hex_string();
    log::info!("using device id: {device_id}");

    let peripherals = Peripherals::take()?;
    let motor = Arc::new(Mutex::new(MotorController::new(PinDriver::output_od(
        peripherals.pins.gpio6,
    )?)?));
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
        "starting machine firmware with device info url {} and machine websocket {}",
        config.device_info_url,
        machine_ws_url
    );
    log::info!(
        "machine firmware is ready with device name {} and device id {}",
        DEVICE_NAME,
        device_id
    );
    log::info!(
        "motor control initialized on GPIO{} in open-drain mode",
        MOTOR_PIN
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

    let status = send_device_info(
        &config.device_info_url,
        &device_id,
        DEVICE_KIND,
        &SUPPORTED_EVENTS,
    )?;
    log::info!(
        "sent machine device info to server with http status {}",
        status
    );

    let remote_config = RemoteControlConfig {
        websocket_url: &machine_ws_url,
    };
    let mut remote_control = connect_remote_control(&remote_config, &device_id)?;
    register_motor_callback(&mut remote_control, &motor, "hw.motor_on", true);
    register_motor_callback(&mut remote_control, &motor, "hw.motor_off", false);
    remote_control.send_online()?;
    log::info!("connected to machine websocket");

    loop {
        let handled = remote_control.poll()?;
        if handled > 0 {
            log::debug!("handled {} remote-control trigger(s)", handled);
        }
        thread::sleep(Duration::from_millis(REMOTE_CONTROL_POLL_INTERVAL_MS));
    }
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

struct MotorController<'d> {
    pin: PinDriver<'d, Output>,
}

impl<'d> MotorController<'d> {
    fn new(motor_pin: PinDriver<'d, Output>) -> Result<Self, esp_idf_svc::sys::EspError> {
        let mut controller = Self { pin: motor_pin };
        controller.set_enabled(false)?;
        Ok(controller)
    }

    fn set_enabled(&mut self, enabled: bool) -> Result<(), esp_idf_svc::sys::EspError> {
        set_motor_output(&mut self.pin, enabled)
    }
}

fn register_motor_callback(
    remote_control: &mut RemoteControlClient,
    motor: &Arc<Mutex<MotorController<'static>>>,
    event_name: &'static str,
    enabled: bool,
) {
    let motor = Arc::clone(motor);
    remote_control.register(event_name, move |_trigger| {
        let mut motor = motor
            .lock()
            .map_err(|_| "failed to lock motor controller".to_owned())?;
        motor
            .set_enabled(enabled)
            .map_err(|err| format!("failed to set motor state: {err}"))?;
        log::info!("set motor {}", if enabled { "on" } else { "off" });
        Ok(None)
    });
}

fn set_motor_output(
    motor_pin: &mut PinDriver<'_, Output>,
    enabled: bool,
) -> Result<(), esp_idf_svc::sys::EspError> {
    if enabled {
        motor_pin.set_low()?;
    } else {
        motor_pin.set_high()?;
    }
    Ok(())
}
