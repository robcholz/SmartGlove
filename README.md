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

## Network API Server

Run the Python server that implements [docs/network-openapi.yaml](docs/network-openapi.yaml):

```bash
uv run server
```

Useful options:

```bash
uv run server --host 0.0.0.0 --port 8000 --log-level debug
```

Config via environment variables:

```bash
SMARTGLOVE_SERVER_HOST=0.0.0.0
SMARTGLOVE_SERVER_PORT=8000
SMARTGLOVE_SERVER_LOG_LEVEL=info
SMARTGLOVE_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```
