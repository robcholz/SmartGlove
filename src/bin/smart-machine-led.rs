#![allow(deprecated)]

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::rmt::{config::TransmitConfig, TxRmtDriver};
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
use ws2812_esp32_rmt_driver::{Ws2812Esp32Rmt, RGB8};

const DEVICE_NAME: &str = "SmartMachine-LED";
const DEVICE_KIND: DeviceKind = DeviceKind::Machine;
const ADVERTISING_WINDOW: Duration = Duration::from_secs(10);
const STATUS_LED_PIN: i32 = 5;
const STATUS_LED_COUNT: usize = 40;
const SUPPORTED_EVENTS: [&str; 3] = ["hw.rgb_green", "hw.rgb_red", "hw.rgb_off"];
const REMOTE_CONTROL_POLL_INTERVAL_MS: u64 = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    esp_idf_svc::sys::link_patches();
    EspLogger::initialize_default();

    let config = load_runtime_config()?;
    let machine_ws_url = derive_machine_ws_url(&config.status_ws_url)?;
    let device_id = DeviceId::from_factory_mac()?.to_hex_string();
    log::info!("using device id: {device_id}");

    let peripherals = Peripherals::take()?;
    let status_led = Arc::new(Mutex::new(StatusLed::new(
        peripherals.rmt.channel0,
        peripherals.pins.gpio5,
    )?));
    set_status_led(&status_led, StatusColor::Booting)?;
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
    log::info!("ws2812 status led initialized on GPIO{}", STATUS_LED_PIN);

    loop {
        log::info!(
            "opening provisioning window for {} seconds",
            ADVERTISING_WINDOW.as_secs()
        );
        set_status_led(&status_led, StatusColor::Provisioning)?;

        match provisioner.start() {
            Ok(()) => {
                set_status_led(&status_led, StatusColor::Provisioned)?;
                log::info!("provisioning completed successfully");
                break;
            }
            Err(ProvisionError::Timeout) => {
                set_status_led(&status_led, StatusColor::TimedOut)?;
                log::warn!("provisioning timed out");
            }
            Err(err) => {
                set_status_led(&status_led, StatusColor::Error)?;
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
    register_led_callback(
        &mut remote_control,
        &status_led,
        "hw.rgb_green",
        StatusColor::Green,
    );
    register_led_callback(
        &mut remote_control,
        &status_led,
        "hw.rgb_red",
        StatusColor::Red,
    );
    register_led_callback(
        &mut remote_control,
        &status_led,
        "hw.rgb_off",
        StatusColor::Off,
    );

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

struct StatusLed<'d> {
    ws2812: Ws2812Esp32Rmt<'d>,
}

impl<'d> StatusLed<'d> {
    fn new<C, P>(channel: C, pin: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        C: esp_idf_svc::hal::rmt::RmtChannel + 'd,
        P: esp_idf_svc::hal::gpio::OutputPin + 'd,
    {
        let config = TransmitConfig::new().clock_divider(1).mem_block_num(2);
        let driver = TxRmtDriver::new(channel, pin, &config)?;
        let ws2812 = Ws2812Esp32Rmt::new_with_rmt_driver(driver)?;
        Ok(Self { ws2812 })
    }

    fn set(&mut self, color: StatusColor) -> Result<(), Box<dyn std::error::Error>> {
        self.ws2812.write_nocopy([color.rgb(); STATUS_LED_COUNT])?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
enum StatusColor {
    Booting,
    Provisioning,
    Provisioned,
    TimedOut,
    Error,
    Green,
    Red,
    Off,
}

impl StatusColor {
    fn rgb(self) -> RGB8 {
        match self {
            Self::Booting => RGB8::new(8, 8, 0),
            Self::Provisioning => RGB8::new(0, 0, 16),
            Self::Provisioned => RGB8::new(0, 16, 0),
            Self::TimedOut => RGB8::new(16, 4, 0),
            Self::Error => RGB8::new(16, 0, 0),
            Self::Green => RGB8::new(0, 16, 0),
            Self::Red => RGB8::new(16, 0, 0),
            Self::Off => RGB8::new(0, 0, 0),
        }
    }
}

fn set_status_led(
    status_led: &Arc<Mutex<StatusLed<'static>>>,
    color: StatusColor,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut led = status_led.lock().map_err(|_| "failed to lock status led")?;
    led.set(color)
}

fn register_led_callback(
    remote_control: &mut RemoteControlClient,
    status_led: &Arc<Mutex<StatusLed<'static>>>,
    event_name: &'static str,
    color: StatusColor,
) {
    let led = Arc::clone(status_led);
    remote_control.register(event_name, move |_trigger| {
        let mut led = led
            .lock()
            .map_err(|_| "failed to lock status led".to_owned())?;
        led.set(color)
            .map_err(|err| format!("failed to set ws2812 color: {err}"))?;
        log::info!("set color '{:?}'", color);
        Ok(None)
    });
}
