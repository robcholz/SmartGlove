#![no_std]

use core::borrow::Borrow;

use embedded_hal_02::blocking::delay::DelayMs;
use embedded_hal_02::blocking::i2c::{Write, WriteRead};
use esp_idf_hal::adc;
use esp_idf_hal::adc::attenuation::DB_12;
use esp_idf_hal::adc::oneshot::config::AdcChannelConfig;
use esp_idf_hal::adc::oneshot::{AdcChannelDriver, AdcDriver};
use esp_idf_hal::gpio::ADCPin;
use esp_idf_hal::sys::EspError;
use mpu6050::{Mpu6050, Mpu6050Error};

pub type Axis3 = [f32; 3];

pub trait Imu {
    type Error;

    fn read_acc(&mut self) -> Result<Axis3, Self::Error>;
    fn read_vec(&mut self) -> Result<Axis3, Self::Error>;
}

pub trait FlexSensor {
    type Error;

    /// Reads the raw sensor value in the inclusive range `0..=1023`.
    fn read_value(&mut self) -> Result<u16, Self::Error>;
}

pub struct Mpu6050Imu<I2C> {
    inner: Mpu6050<I2C>,
}

impl<I2C, E> Mpu6050Imu<I2C>
where
    I2C: Write<Error = E> + WriteRead<Error = E>,
{
    pub fn new<D>(i2c: I2C, delay: &mut D) -> Result<Self, Mpu6050Error<E>>
    where
        D: DelayMs<u8>,
    {
        let mut inner = Mpu6050::new(i2c);
        inner.init(delay)?;

        Ok(Self { inner })
    }

    pub fn new_with_addr<D>(i2c: I2C, address: u8, delay: &mut D) -> Result<Self, Mpu6050Error<E>>
    where
        D: DelayMs<u8>,
    {
        let mut inner = Mpu6050::new_with_addr(i2c, address);
        inner.init(delay)?;

        Ok(Self { inner })
    }
}

impl<I2C, E> Imu for Mpu6050Imu<I2C>
where
    I2C: Write<Error = E> + WriteRead<Error = E>,
{
    type Error = Mpu6050Error<E>;

    fn read_acc(&mut self) -> Result<Axis3, Self::Error> {
        let acc = self.inner.get_acc()?;
        Ok([-acc.x, -acc.y, -acc.z]) // calibrated by the phy placement
    }

    fn read_vec(&mut self) -> Result<Axis3, Self::Error> {
        let gyro = self.inner.get_gyro()?;
        Ok([-gyro.x, -gyro.y, -gyro.z]) // calibrated by the phy placement
    }
}

pub struct AnalogFlexSensor<'d, C, M>
where
    C: adc::AdcChannel,
    M: Borrow<AdcDriver<'d, C::AdcUnit>>,
{
    channel: AdcChannelDriver<'d, C, M>,
}

impl<'d, C, M> AnalogFlexSensor<'d, C, M>
where
    C: adc::AdcChannel,
    M: Borrow<AdcDriver<'d, C::AdcUnit>>,
{
    pub fn new(channel: AdcChannelDriver<'d, C, M>) -> Self {
        Self { channel }
    }

    pub fn new_with_pin(adc: M, pin: impl ADCPin<AdcChannel = C> + 'd) -> Result<Self, EspError> {
        let config = AdcChannelConfig {
            attenuation: DB_12,
            ..Default::default()
        };
        let channel = AdcChannelDriver::new(adc, pin, &config)?;

        Ok(Self::new(channel))
    }

    fn normalize_to_10bit(raw: u16) -> u16 {
        ((u32::from(raw).min(4095) * 1023) / 4095) as u16
    }
}

impl<'d, C, M> FlexSensor for AnalogFlexSensor<'d, C, M>
where
    C: adc::AdcChannel,
    M: Borrow<AdcDriver<'d, C::AdcUnit>>,
{
    type Error = EspError;

    fn read_value(&mut self) -> Result<u16, Self::Error> {
        let raw = self.channel.read_raw()?;
        Ok(Self::normalize_to_10bit(raw))
    }
}
