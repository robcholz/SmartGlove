use drivers::{AnalogFlexSensor, FlexSensor, Imu, Mpu6050Imu};
use esp_idf_svc::hal::adc::oneshot::AdcDriver;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::i2c::{I2cConfig, I2cDriver};
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::units::*;
use smart_glove::inference::{
    run_quantized_self_test, run_raw_sensor_input_with_scratch, SlidingWindow, MODEL_INPUT_LEN,
    MODEL_WINDOW_SIZE,
};
use std::thread;

const IMU_I2C_ADDRESS: u8 = 0x68;
const SAMPLE_INTERVAL_MS: u32 = 10;
const INFERENCE_INTERVAL_FRAMES: usize = 10;
const LIVE_INFERENCE_STACK_SIZE: usize = 24 * 1024;

fn main() {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    log::info!("smart-glove live inference startup");

    let worker = thread::Builder::new()
        .name("live-inference".into())
        .stack_size(LIVE_INFERENCE_STACK_SIZE)
        .spawn(run_live_inference)
        .expect("failed to spawn live inference worker");

    if let Err(err) = worker.join() {
        panic!("live inference worker panicked: {:?}", err);
    }
}

fn run_live_inference() {
    match run_quantized_self_test() {
        Ok(report) => log::info!(
            "esp-dl self-test ok: predicted_label={} exact_quantized_match={} max_dequantized_abs_error={:.6}",
            report.inference.predicted_label,
            report.exact_quantized_match,
            report.max_dequantized_abs_error
        ),
        Err(err) => {
            log::error!("esp-dl self-test failed before live inference: {err}");
            return;
        }
    }

    let peripherals = Peripherals::take().expect("failed to take peripherals");

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
            log::error!(
                "failed to init MPU6050 on gpio47/gpio48 at address 0x{:02x}: {:?}",
                IMU_I2C_ADDRESS,
                err
            );
            return;
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
            log::info!(
                "collecting samples: buffered={}/{} latest_frame={:?}",
                sliding_window.len(),
                MODEL_WINDOW_SIZE,
                frame
            );
        }

        FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
    }
}
