# Provision App

Static Web Bluetooth app for provisioning SmartGlove devices.

Run it from localhost:

```bash
python3 -m http.server 4173 --directory provision-app
```

Then open:

```text
http://localhost:4173
```

Notes:

- Use Chrome or Edge with Web Bluetooth enabled.
- The BLE scan path filters on the SmartGlove manufacturer marker `0xFFFF / "SG"`.
- The browser picker fallback matches by `SmartGlove` name prefix when scan mode is not available.
