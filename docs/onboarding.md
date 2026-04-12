# Onboarding

This project targets `ESP32-S3` and builds with the Espressif Xtensa Rust toolchain, not plain upstream Rust.

The source of truth in the repo today is:

- [`rust-toolchain.toml`](../rust-toolchain.toml): selects the custom `esp` Rust toolchain
- [`.cargo/config.toml`](../.cargo/config.toml): sets the build target to `xtensa-esp32s3-none-elf`
- [Cargo.toml](../Cargo.toml): enables `esp32s3` features for `esp-hal`, `esp-println`, `esp-rtos`, and `esp-backtrace`

## 1. What The Error Means

If you see:

```text
error: custom toolchain 'esp' specified in override file '.../rust-toolchain.toml' is not installed
```

that means the repo is configured correctly, but your machine has not installed the Espressif Rust toolchain yet.

The fix is to install the `esp` toolchain locally with `espup`.

## 2. Required Tooling

Install these CLIs first:

```bash
cargo install espup --locked
cargo install espflash --locked --version 4.3.0
```

Optional but useful:

```bash
cargo install probe-rs-tools --locked
```

What each tool is for:

- `espup`: installs and manages the Espressif Rust toolchain used by this repo
- `espflash`: flashes firmware to the board and opens a serial monitor
- `probe-rs`: optional hardware debugging beyond simple serial flashing

## 3. Install The `esp` Toolchain

Run:

```bash
espup install
```

This creates the custom `esp` toolchain expected by [`rust-toolchain.toml`](../rust-toolchain.toml).

After installation, load the environment in each shell:

```bash
source "$HOME/export-esp.sh"
```

To verify it worked:

```bash
rustc --version --verbose
rustup show active-toolchain
```

From this repo, `rustup show active-toolchain` should report `esp`.

## 4. Locked Toolchain Expectations

This repo intentionally uses:

- custom Rust toolchain: `esp`
- target triple: `xtensa-esp32s3-none-elf`

That selection is locked in the repo through:

- [`rust-toolchain.toml`](../rust-toolchain.toml)
- [`.cargo/config.toml`](../.cargo/config.toml)

CI is aligned to the same Xtensa setup and installs the toolchain with `esp-rs/xtensa-toolchain@v1.5`, so local development should follow the same family of toolchain instead of plain `stable`.

## 5. Build And Flash Workflow

The repo already defines the default target and runner in [`.cargo/config.toml`](../.cargo/config.toml):

- target: `xtensa-esp32s3-none-elf`
- runner: `espflash flash --monitor --chip esp32s3`

That means the normal commands are:

```bash
cargo build
cargo run
```

`cargo run` will build, flash, and attach the serial monitor through `espflash`.

If you want to use `espflash` directly:

```bash
espflash board-info
espflash flash --monitor --chip esp32s3 target/xtensa-esp32s3-none-elf/debug/SmartGlove
```

For release firmware:

```bash
cargo build --release
espflash flash --monitor --chip esp32s3 target/xtensa-esp32s3-none-elf/release/SmartGlove
```

## 6. Verification Checklist

After `espup install` and `source "$HOME/export-esp.sh"`:

```bash
rustup show active-toolchain
cargo build
```

Expected result:

- active toolchain is `esp`
- the firmware builds for `xtensa-esp32s3-none-elf`

Once hardware is connected, verify flashing:

```bash
espflash board-info
cargo run
```

## 7. System Prerequisites

On macOS, make sure you have:

- Xcode Command Line Tools
- a working USB data cable
- any USB serial driver required by your specific ESP32-S3 board

Install Apple developer tools if needed:

```bash
xcode-select --install
```

## 8. Troubleshooting

### `custom toolchain 'esp' is not installed`

Fix:

```bash
cargo install espup --locked
espup install
source "$HOME/export-esp.sh"
```

### `cargo run` does not flash

Check:

- the board is connected over USB data, not charge-only USB
- the board is actually an `ESP32-S3`
- `espflash board-info` can see the device

### shell works in one terminal but not another

You likely installed the toolchain once but did not load the environment in the new shell.

Run:

```bash
source "$HOME/export-esp.sh"
```
