# SmartGlove

See [onboarding.md](docs/onboarding.md)

See [CONTRIBUTION.md](CONTRIBUTION.md)

## Provisioning Test Firmware

Build the dedicated provisioning firmware:

```bash
bash scripts/build.sh release provision
```

Flash the provisioning firmware:

```bash
bash scripts/flash.sh release provision
```

## UV Test Tools

Install Python dependencies with `uv`:

```bash
uv sync
```

Capture serial output:

```bash
uv run provision-serial --port /dev/tty.usbmodem1101
```

Run the BLE provisioning test:

```bash
uv run provision-test \
  --ssid YOUR_WIFI_SSID \
  --password YOUR_WIFI_PASSWORD \
  --serial-port /dev/tty.usbmodem1101 \
  --expect-serial PROVISION_DONE
```

Run the full workflow in one command:

```bash
uv run provision-e2e \
  --ssid YOUR_WIFI_SSID \
  --password YOUR_WIFI_PASSWORD \
  --serial-port /dev/tty.usbmodem1101
```

Useful variants:

```bash
uv run provision-e2e \
  --ssid YOUR_WIFI_SSID \
  --password YOUR_WIFI_PASSWORD \
  --serial-port /dev/tty.usbmodem1101 \
  --serial-log provision.log
```

```bash
uv run provision-e2e \
  --ssid YOUR_WIFI_SSID \
  --password YOUR_WIFI_PASSWORD \
  --serial-port /dev/tty.usbmodem1101 \
  --skip-build
```
