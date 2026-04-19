# Provision App

Static Web Bluetooth app for provisioning SmartGlove and SmartMachine devices.

Run it from localhost:

```bash
uv run provision-app
```

Then open:

```text
http://localhost:4173
```

Optional flags:

```bash
uv run provision-app --host 0.0.0.0 --port 8080
```

Notes:

- Use Chrome or Edge with Web Bluetooth enabled.
- The BLE scan path filters on the shared provisioning manufacturer marker `0xFFFF / "SG"`.
- The browser picker fallback uses `acceptAllDevices`, so you can manually pick SmartGlove or SmartMachine provisioners even if the browser does not expose a stable local name prefix during discovery.
