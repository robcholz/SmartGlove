use smart_glove::inference::{
    run_quantized_self_test, run_raw_sensor_input_with_scratch, SlidingWindow, MODEL_INPUT_LEN,
    MODEL_WINDOW_SIZE,
};
use std::io::Error as IoError;
use std::thread;
use std::time::Duration;

use drivers::{AnalogFlexSensor, FlexSensor, Imu, Mpu6050Imu};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::adc::oneshot::AdcDriver;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::i2c::{I2cConfig, I2cDriver};
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::units::*;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use identity::DeviceId;
use network::{
    connect_status_reporter, send_device_info, FlexReadings, SensorSample, StatusReportConfig,
};
use provision::{
    BasicProvision, EspProvisionCapabilityProvider, Provision, ProvisionCapabilityProvider,
    ProvisionError,
};
use smart_glove::runtime_config::{load_runtime_config, RuntimeConfig};

const IMU_I2C_ADDRESS: u8 = 0x68;
const SAMPLE_INTERVAL_MS: u32 = 10;
const INFERENCE_INTERVAL_FRAMES: usize = 10;
const LIVE_INFERENCE_STACK_SIZE: usize = 24 * 1024;
const DEVICE_NAME: &str = "SmartGlove Provision";
const ADVERTISING_WINDOW: Duration = Duration::from_secs(10);
const EVENT_NAME: &str = "event.none";
const EVENT_PAYLOAD_JSON: &str = "null";

fn main() {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    let runtime_config = load_runtime_config().expect("failed to load network runtime config");

    log::info!(
        "smart-glove live inference startup device_info_url={} status_ws_url={} sample_rate_hz={} batch_samples={}",
        runtime_config.device_info_url,
        runtime_config.status_ws_url,
        runtime_config.sample_rate_hz,
        runtime_config.batch_samples
    );

    let worker = thread::Builder::new()
        .name("live-inference".into())
        .stack_size(LIVE_INFERENCE_STACK_SIZE)
        .spawn(move || {
            if let Err(err) = run_live_inference(runtime_config) {
                log::error!("live inference worker failed: {err}");
            }
        })
        .expect("failed to spawn live inference worker");

    if let Err(err) = worker.join() {
        panic!("live inference worker panicked: {:?}", err);
    }
}

fn run_live_inference(config: RuntimeConfig) -> Result<(), Box<dyn std::error::Error>> {
    match run_quantized_self_test() {
        Ok(report) => log::info!(
            "esp-dl self-test ok: predicted_label={} exact_quantized_match={} max_dequantized_abs_error={:.6}",
            report.inference.predicted_label,
            report.exact_quantized_match,
            report.max_dequantized_abs_error
        ),
        Err(err) => {
            log::error!("esp-dl self-test failed before live inference: {err}");
            return Ok(());
        }
    }

    if config.sample_rate_hz != 1000u16 / (SAMPLE_INTERVAL_MS as u16) {
        log::warn!(
            "runtime sample_rate_hz={} does not match inference loop cadence={}Hz",
            config.sample_rate_hz,
            1000u16 / (SAMPLE_INTERVAL_MS as u16)
        );
    }

    let device_id = DeviceId::from_factory_mac()?.to_hex_string();
    let peripherals = Peripherals::take().expect("failed to take peripherals");
    let sysloop = EspSystemEventLoop::take()?;
    let nvs = EspDefaultNvsPartition::take()?;
    let (wifi_modem, bluetooth_modem) = peripherals.modem.split();

    let i2c_config = I2cConfig::new().baudrate(100.kHz().into());
    let i2c = I2cDriver::new(
        peripherals.i2c0,
        peripherals.pins.gpio47,
        peripherals.pins.gpio48,
        &i2c_config,
    )
    .expect("failed to initialize I2C");

    let mut delay = FreeRtos;
    let mut imu = match Mpu6050Imu::new_with_addr(i2c, IMU_I2C_ADDRESS, &mut delay) {
        Ok(imu) => imu,
        Err(err) => {
            return Err(IoError::other(format!(
                "failed to init MPU6050 on gpio47/gpio48 at address 0x{IMU_I2C_ADDRESS:02x}: {err:?}"
            ))
            .into());
        }
    };

    let adc = AdcDriver::new(peripherals.adc1).expect("failed to initialize ADC1");
    let mut thumb_finger = AnalogFlexSensor::new_with_pin(&adc, peripherals.pins.gpio4)
        .expect("failed to init thumb sensor");
    let mut index_finger = AnalogFlexSensor::new_with_pin(&adc, peripherals.pins.gpio5)
        .expect("failed to init index sensor");
    let mut middle_finger = AnalogFlexSensor::new_with_pin(&adc, peripherals.pins.gpio6)
        .expect("failed to init middle sensor");
    let mut ring_finger = AnalogFlexSensor::new_with_pin(&adc, peripherals.pins.gpio7)
        .expect("failed to init ring sensor");
    let mut pinky_finger = AnalogFlexSensor::new_with_pin(&adc, peripherals.pins.gpio8)
        .expect("failed to init pinky sensor");

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

    let mut sliding_window = Box::new(SlidingWindow::new());
    let mut feature_input = Box::new([0.0f32; MODEL_INPUT_LEN]);
    let mut normalized_window = Box::new([0.0f32; MODEL_INPUT_LEN]);
    let mut frame_counter = 0usize;

    loop {
        let acc = match imu.read_acc() {
            Ok(acc) => acc,
            Err(err) => {
                log::error!("failed to read accelerometer: {:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };

        let gyro = match imu.read_vec() {
            Ok(vec) => vec,
            Err(err) => {
                log::error!("failed to read vector: {:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };

        let thumb = match thumb_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read thumb flex sensor: {:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };
        let index = match index_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read index flex sensor: {:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };
        let middle = match middle_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read middle flex sensor: {:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };
        let ring = match ring_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read ring flex sensor: {:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };
        let pinky = match pinky_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("failed to read pinky flex sensor: {:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };

        let network_sample = SensorSample {
            imu_acc: acc,
            vel: gyro,
            flex: FlexReadings {
                thumb: thumb as f32,
                index: index as f32,
                middle: middle as f32,
                ring: ring as f32,
                pinky: pinky as f32,
            },
        };
        if reporter.push_sensor_sample(network_sample)? {
            log::debug!(
                "NETWORK_BATCH_FLUSHED buffered_samples={}",
                reporter.buffered_samples()
            );
        }

        let frame = [
            thumb as f32,
            index as f32,
            middle as f32,
            ring as f32,
            pinky as f32,
            acc[0],
            acc[1],
            acc[2],
        ];

        sliding_window.push_frame(frame);
        frame_counter += 1;

        if sliding_window.is_full() && frame_counter % INFERENCE_INTERVAL_FRAMES == 0 {
            if sliding_window.extract_features_into(feature_input.as_mut()) {
                match run_raw_sensor_input_with_scratch(
                    feature_input.as_ref(),
                    normalized_window.as_mut(),
                ) {
                    Ok(result) => {
                        let top3 = result.top_predictions(3);
                        if top3.len() < 3 {
                            log::error!(
                                "gesture inference returned too few predictions: {}",
                                top3.len()
                            );
                            FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                            continue;
                        }
                        log::info!(
                            "gesture classification: label={} top3=[{}:{:.3}, {}:{:.3}, {}:{:.3}]",
                            result.predicted_label,
                            top3[0].label,
                            top3[0].score,
                            top3[1].label,
                            top3[1].score,
                            top3[2].label,
                            top3[2].score,
                        );
                    }
                    Err(err) => log::error!("gesture inference failed: {err}"),
                }
            }
        } else if frame_counter % 25 == 0 {
            log::debug!(
                "collecting samples: buffered={}/{} latest_frame={:?}",
                sliding_window.len(),
                MODEL_WINDOW_SIZE,
                frame
            );
        }

        FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
    }

    #[allow(unreachable_code)]
    Ok(())
}
