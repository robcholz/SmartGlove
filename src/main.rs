use std::io::Error as IoError;
use std::time::Duration;

use drivers::{AnalogFlexSensor, FlexSensor, Imu, Mpu6050Imu};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::adc::oneshot::AdcDriver;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::i2c::{I2cConfig, I2cDriver};
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::units::*;
use esp_idf_svc::log::EspLogger;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use identity::DeviceId;
use network::{
    connect_status_reporter, send_device_info, FlexReadings, SensorSample, StatusReportConfig,
};
use provision::{
    BasicProvision, EspProvisionCapabilityProvider, Provision, ProvisionCapabilityProvider,
    ProvisionError,
};
use smart_glove::runtime_config::load_runtime_config;

const IMU_I2C_ADDRESS: u8 = 0x68;
const DEVICE_NAME: &str = "SmartGlove Provision";
const ADVERTISING_WINDOW: Duration = Duration::from_secs(10);
const EVENT_NAME: &str = "event.none";
const EVENT_PAYLOAD_JSON: &str = "null";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    esp_idf_svc::sys::link_patches();
    EspLogger::initialize_default();

    let config = load_runtime_config()?;
    let device_id = DeviceId::from_factory_mac()?.to_hex_string();

    let peripherals = Peripherals::take()?;
    let sysloop = EspSystemEventLoop::take()?;
    let nvs = EspDefaultNvsPartition::take()?;
    let (wifi_modem, bluetooth_modem) = peripherals.modem.split();

    // - MPU6050 on I2C0 with SDA=GPIO47, SCL=GPIO48
    // - Five analog flex sensors on ADC1 / GPIO4..GPIO8
    let i2c_config = I2cConfig::new().baudrate(100.kHz().into());
    let i2c = I2cDriver::new(
        peripherals.i2c0,
        peripherals.pins.gpio47,
        peripherals.pins.gpio48,
        &i2c_config,
    )?;

    let mut delay = FreeRtos;
    let mut imu = Mpu6050Imu::new_with_addr(i2c, IMU_I2C_ADDRESS, &mut delay)
        .map_err(|err| IoError::other(format!("failed to init MPU6050: {err:?}")))?;

    let adc = AdcDriver::new(peripherals.adc1)?;
    let mut thumb_finger = AnalogFlexSensor::new_with_pin(&adc, peripherals.pins.gpio4)?;
    let mut index_finger = AnalogFlexSensor::new_with_pin(&adc, peripherals.pins.gpio5)?;
    let mut middle_finger = AnalogFlexSensor::new_with_pin(&adc, peripherals.pins.gpio6)?;
    let mut ring_finger = AnalogFlexSensor::new_with_pin(&adc, peripherals.pins.gpio7)?;
    let mut pinky_finger = AnalogFlexSensor::new_with_pin(&adc, peripherals.pins.gpio8)?;

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
        "SMART_GLOVE_READY name={} device_id={} device_info_url={} status_ws_url={}",
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

    let events = [EVENT_NAME];
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

    reporter.send_event(EVENT_NAME, EVENT_PAYLOAD_JSON)?;
    log::info!("NETWORK_EVENT_SENT name={}", EVENT_NAME);

    log::info!(
        "SENSOR_STREAM_STARTED sample_rate_hz={} batch_samples={} flush_interval_ms={}",
        config.sample_rate_hz,
        config.batch_samples,
        config.flush_interval_ms
    );

    loop {
        let acc = match imu.read_acc() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read accelerometer: {:?}", err);
                FreeRtos::delay_ms(100u32);
                continue;
            }
        };

        let vel = match imu.read_vec() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read vector: {:?}", err);
                FreeRtos::delay_ms(100u32);
                continue;
            }
        };

        let thumb = match thumb_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read thumb flex sensor: {:?}", err);
                FreeRtos::delay_ms(100u32);
                continue;
            }
        };
        let index = match index_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read index flex sensor: {:?}", err);
                FreeRtos::delay_ms(100u32);
                continue;
            }
        };
        let middle = match middle_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read middle flex sensor: {:?}", err);
                FreeRtos::delay_ms(100u32);
                continue;
            }
        };
        let ring = match ring_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read ring flex sensor: {:?}", err);
                FreeRtos::delay_ms(100u32);
                continue;
            }
        };
        let pinky = match pinky_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read pinky flex sensor: {:?}", err);
                FreeRtos::delay_ms(100u32);
                continue;
            }
        };

        let sample = SensorSample {
            imu_acc: acc,
            vel,
            flex: FlexReadings {
                thumb: normalize_flex_reading(thumb),
                index: normalize_flex_reading(index),
                middle: normalize_flex_reading(middle),
                ring: normalize_flex_reading(ring),
                pinky: normalize_flex_reading(pinky),
            },
        };

        if reporter.push_sensor_sample(sample)? {
            log::debug!(
                "NETWORK_BATCH_FLUSHED buffered_samples={}",
                reporter.buffered_samples()
            );
        }

        let sample_delay_ms = 1000u32 / u32::from(config.sample_rate_hz);
        if sample_delay_ms > 0 {
            FreeRtos::delay_ms(sample_delay_ms);
        }
    }

    #[allow(unreachable_code)]
    Ok(())
}

fn normalize_flex_reading(raw: u16) -> f32 {
    f32::from(raw) / 1023.0
}
