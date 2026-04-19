use drivers::{AnalogFlexSensor, FlexSensor, Imu, Mpu6050Imu};
use esp_idf_svc::hal::adc::oneshot::AdcDriver;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::i2c::{I2cConfig, I2cDriver};
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::units::*;

const IMU_I2C_ADDRESS: u8 = 0x68;
const SAMPLE_INTERVAL_MS: u32 = 10;

fn main() {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    let peripherals = Peripherals::take().expect("failed to take peripherals");

    // - MPU6050 on I2C0 with SDA=GPIO47, SCL=GPIO48
    // - Five analog flex sensors on ADC1 / GPIO4..GPIO8
    let i2c_config = I2cConfig::new().baudrate(100.kHz().into());
    let i2c = I2cDriver::new(
        peripherals.i2c0,
        peripherals.pins.gpio47,
        peripherals.pins.gpio48,
        &i2c_config,
    )
    .expect("failed to initialize I2C");

    let mut delay = FreeRtos;
    let mut imu = Mpu6050Imu::new_with_addr(i2c, IMU_I2C_ADDRESS, &mut delay)
        .expect("failed to init MPU6050");

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

    log::info!("driver started");

    loop {
        let acc = match imu.read_acc() {
            Ok(acc) => acc,
            Err(err) => {
                log::error!("driver sample failed: accelerometer={:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };

        let vec = match imu.read_vec() {
            Ok(vec) => vec,
            Err(err) => {
                log::error!("driver sample failed: vector={:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };

        let thumb = match thumb_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("driver sample failed: thumb={:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };

        let index = match index_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("driver sample failed: index={:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };

        let middle = match middle_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("driver sample failed: middle={:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };

        let ring = match ring_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("driver sample failed: ring={:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };

        let pinky = match pinky_finger.read_value() {
            Ok(value) => value,
            Err(err) => {
                log::error!("driver sample failed: pinky={:?}", err);
                FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
                continue;
            }
        };

        log::info!(
            "sample acc={:.3},{:.3},{:.3} vec={:.3},{:.3},{:.3} flex={},{},{},{},{}",
            acc[0],
            acc[1],
            acc[2],
            vec[0],
            vec[1],
            vec[2],
            thumb,
            index,
            middle,
            ring,
            pinky
        );

        FreeRtos::delay_ms(SAMPLE_INTERVAL_MS);
    }
}
