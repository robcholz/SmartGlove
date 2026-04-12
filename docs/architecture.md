# Architecture

See [Onboarding](onboarding.md) for local environment setup, pinned toolchain details, and ESP flashing tools.

## Hardware

- MPU6050
- Flex Resister
- LiPo
- ESP32-S3

## Driver

- MPU6050
  - mpu6050 = "0.1.6"
- Flex Resister
  - use normal ADC
  - remember to measure the value changes vs the physical change (if the value is linear or non-linear).
- LiPo

- ESP32
  - https://github.com/esp-rs/esp-hal.git
  - Runtime baseline: `esp-hal` + `esp-rtos` with Embassy on `no_std`

## Service

- HTTP
  - https://github.com/drogue-iot/reqwless.git
- Websocket
  -
