const SERVICE_UUID = "000012ff-0000-1000-8000-00805f9b34fb";
const CREDENTIALS_UUID = "0000ff01-0000-1000-8000-00805f9b34fb";
const STATUS_UUID = "0000ff02-0000-1000-8000-00805f9b34fb";
const MANUFACTURER_ID = 0xffff;
const MANUFACTURER_MAGIC = "SG";
const NAME_PREFIXES = ["SmartGlove", "SmartMachine"];
const DEFAULT_DEVICE_LABEL = "Provisioning device";

const state = {
  devices: new Map(),
  selectedDeviceId: null,
  selectedConnection: null,
  scan: null,
};

const ui = {
  scanButton: document.querySelector("#scan-button"),
  pickerButton: document.querySelector("#picker-button"),
  clearLogButton: document.querySelector("#clear-log-button"),
  deviceList: document.querySelector("#device-list"),
  scanState: document.querySelector("#scan-state"),
  selectedTitle: document.querySelector("#selected-title"),
  deviceState: document.querySelector("#device-state"),
  statusText: document.querySelector("#status-text"),
  markerText: document.querySelector("#marker-text"),
  browserHint: document.querySelector("#browser-hint"),
  provisionForm: document.querySelector("#provision-form"),
  ssidInput: document.querySelector("#ssid-input"),
  passwordInput: document.querySelector("#password-input"),
  connectButton: document.querySelector("#connect-button"),
  disconnectButton: document.querySelector("#disconnect-button"),
  logView: document.querySelector("#log-view"),
  rowTemplate: document.querySelector("#device-row-template"),
};

boot();

function boot() {
  if (!("bluetooth" in navigator)) {
    setHint("Web Bluetooth is not available in this browser.");
    ui.scanButton.disabled = true;
    ui.pickerButton.disabled = true;
    return;
  }

  setHint(
    "Use Chrome or Edge on localhost. Scan mode uses the SmartGlove/SmartMachine manufacturer marker."
  );
  renderDevices();
  renderLogEmpty();

  ui.scanButton.addEventListener("click", startBleScan);
  ui.pickerButton.addEventListener("click", addDeviceViaPicker);
  ui.clearLogButton.addEventListener("click", clearLog);
  ui.disconnectButton.addEventListener("click", disconnectSelectedDevice);
  ui.provisionForm.addEventListener("submit", submitProvisioning);
}

async function startBleScan() {
  if (!("requestLEScan" in navigator.bluetooth)) {
    setHint("This browser does not support BLE scan mode. Use the browser picker fallback.");
    return;
  }

  try {
    if (state.scan) {
      state.scan.stop();
      state.scan = null;
    }

    const manufacturerData = {
      companyIdentifier: MANUFACTURER_ID,
      dataPrefix: textEncoder(MANUFACTURER_MAGIC),
      mask: new Uint8Array([0xff, 0xff]),
    };

    state.scan = await navigator.bluetooth.requestLEScan({
      keepRepeatedDevices: true,
      filters: [{ manufacturerData: [manufacturerData] }],
      optionalServices: [SERVICE_UUID],
    });

    navigator.bluetooth.addEventListener("advertisementreceived", handleAdvertisement);
    setScanState("Scanning", "active");
    logEvent("BLE scan started");
  } catch (error) {
    logError(`Unable to start scan: ${error.message}`);
    setScanState("Scan Failed", "error");
  }
}

async function addDeviceViaPicker() {
  try {
    const device = await navigator.bluetooth.requestDevice({
      acceptAllDevices: true,
      optionalServices: [SERVICE_UUID],
    });

    upsertDevice({
      id: device.id,
      name: device.name || DEFAULT_DEVICE_LABEL,
      device,
      rssi: "picker",
      marker: "picker",
    });
    selectDevice(device.id);
    logEvent(`Added ${device.name || device.id} via browser picker`);
  } catch (error) {
    if (error.name !== "NotFoundError") {
      logError(`Picker failed: ${error.message}`);
    }
  }
}

function handleAdvertisement(event) {
  const markerView = event.manufacturerData.get(MANUFACTURER_ID);
  const marker = markerView ? dataViewToAscii(markerView) : "unknown";
  if (marker !== MANUFACTURER_MAGIC) {
    return;
  }

  const device = event.device;
  upsertDevice({
    id: device.id,
    name: device.name || event.name || DEFAULT_DEVICE_LABEL,
    device,
    rssi: typeof event.rssi === "number" ? `${event.rssi} dBm` : "n/a",
    marker,
  });
}

function upsertDevice(record) {
  state.devices.set(record.id, {
    ...state.devices.get(record.id),
    ...record,
  });

  if (!state.selectedDeviceId) {
    state.selectedDeviceId = record.id;
  }

  renderDevices();
  syncSelectionUi();
}

function renderDevices() {
  ui.deviceList.innerHTML = "";

  if (state.devices.size === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No provisioners discovered yet.";
    ui.deviceList.append(empty);
    return;
  }

  for (const deviceRecord of state.devices.values()) {
    const row = ui.rowTemplate.content.firstElementChild.cloneNode(true);
    row.dataset.deviceId = deviceRecord.id;
    row.querySelector(".device-name").textContent = deviceRecord.name || "Unnamed device";
    row.querySelector(".device-id").textContent = deviceRecord.id;
    row.querySelector(".device-rssi").textContent = `Signal: ${deviceRecord.rssi || "n/a"}`;
    row.querySelector(".device-marker").textContent = `Marker: ${deviceRecord.marker || "n/a"}`;
    row.classList.toggle("selected", state.selectedDeviceId === deviceRecord.id);
    row.addEventListener("click", () => selectDevice(deviceRecord.id));
    ui.deviceList.append(row);
  }
}

function selectDevice(deviceId) {
  state.selectedDeviceId = deviceId;
  renderDevices();
  syncSelectionUi();
}

function syncSelectionUi() {
  const deviceRecord = state.devices.get(state.selectedDeviceId);
  const hasDevice = Boolean(deviceRecord);

  ui.selectedTitle.textContent = hasDevice
    ? deviceRecord.name || DEFAULT_DEVICE_LABEL
    : "No device selected";
  ui.markerText.textContent = hasDevice
    ? `${deviceRecord.marker || "unknown"} · ${deviceRecord.id}`
    : "Not scanned yet";
  ui.connectButton.disabled = !hasDevice;
  ui.disconnectButton.disabled = !state.selectedConnection;

  if (!hasDevice) {
    setDeviceState("Disconnected", "idle");
    ui.statusText.textContent = "Waiting for a device.";
  }
}

async function submitProvisioning(event) {
  event.preventDefault();
  const deviceRecord = state.devices.get(state.selectedDeviceId);
  if (!deviceRecord) {
    return;
  }

  try {
    setDeviceState("Connecting BLE", "active");
    ui.statusText.textContent = "Opening BLE session…";
    const connection = await connectToProvisioner(deviceRecord);
    state.selectedConnection = connection;
    ui.disconnectButton.disabled = false;

    const ssid = ui.ssidInput.value.trim();
    const password = ui.passwordInput.value;
    const payload = textEncoder(`${ssid}\n${password}`);
    await connection.credentials.writeValueWithResponse(payload);
    logEvent(`Provisioning payload sent to ${deviceRecord.name || deviceRecord.id}`);
  } catch (error) {
    logError(`Provisioning failed: ${error.message}`);
    setDeviceState("Error", "error");
    ui.statusText.textContent = error.message;
  }
}

async function connectToProvisioner(deviceRecord) {
  const existing = state.selectedConnection;
  if (existing?.device?.id === deviceRecord.id && existing.device.gatt.connected) {
    return existing;
  }

  if (existing) {
    await disconnectConnection(existing);
  }

  const device = deviceRecord.device;
  device.addEventListener("gattserverdisconnected", onGattDisconnected, { once: true });

  const server = await device.gatt.connect();
  const service = await server.getPrimaryService(SERVICE_UUID);
  const credentials = await service.getCharacteristic(CREDENTIALS_UUID);
  const status = await service.getCharacteristic(STATUS_UUID);

  await status.startNotifications();
  status.addEventListener("characteristicvaluechanged", onStatusChanged);

  const connection = { device, server, service, credentials, status };
  state.selectedConnection = connection;
  setDeviceState("BLE Connected", "active");
  ui.statusText.textContent = "Connected. Waiting for device status…";
  logEvent(`BLE connected to ${device.name || device.id}`);
  return connection;
}

async function disconnectSelectedDevice() {
  if (!state.selectedConnection) {
    return;
  }

  await disconnectConnection(state.selectedConnection);
}

async function disconnectConnection(connection) {
  try {
    connection.status.removeEventListener("characteristicvaluechanged", onStatusChanged);
    if (connection.device.gatt.connected) {
      connection.device.gatt.disconnect();
    }
  } finally {
    state.selectedConnection = null;
    setDeviceState("Disconnected", "idle");
    ui.disconnectButton.disabled = true;
    ui.statusText.textContent = "Disconnected.";
    logEvent("BLE disconnected");
  }
}

function onGattDisconnected() {
  state.selectedConnection = null;
  ui.disconnectButton.disabled = true;
  setDeviceState("Disconnected", "idle");
  ui.statusText.textContent = "Device disconnected.";
  logEvent("Device disconnected from BLE");
}

function onStatusChanged(event) {
  const status = new TextDecoder().decode(event.target.value);
  ui.statusText.textContent = status;
  logEvent(`Status: ${status}`);

  if (status === "connected") {
    setDeviceState("Wi-Fi Connected", "success");
  } else if (status.startsWith("connection_failed:")) {
    setDeviceState("Wi-Fi Failed", "error");
  } else {
    setDeviceState(status.replaceAll("_", " "), "active");
  }
}

function clearLog() {
  ui.logView.innerHTML = "";
  renderLogEmpty();
}

function renderLogEmpty() {
  if (ui.logView.children.length > 0) {
    return;
  }
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.textContent = "No events yet.";
  ui.logView.append(empty);
}

function logEvent(message) {
  appendLog(message, false);
}

function logError(message) {
  appendLog(message, true);
}

function appendLog(message, isError) {
  if (ui.logView.firstElementChild?.classList.contains("empty-state")) {
    ui.logView.innerHTML = "";
  }

  const row = document.createElement("div");
  row.className = "log-entry";
  if (isError) {
    row.style.borderColor = "rgba(192, 72, 52, 0.38)";
  }

  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString();
  const text = document.createElement("div");
  text.textContent = message;
  row.append(time, text);
  ui.logView.prepend(row);
}

function setHint(text) {
  ui.browserHint.textContent = text;
}

function setScanState(text, tone) {
  ui.scanState.textContent = text;
  ui.scanState.className = `status-pill ${tone}`;
}

function setDeviceState(text, tone) {
  ui.deviceState.textContent = text;
  ui.deviceState.className = `status-pill ${tone}`;
}

function textEncoder(text) {
  return new TextEncoder().encode(text);
}

function dataViewToAscii(dataView) {
  return new TextDecoder().decode(new Uint8Array(dataView.buffer, dataView.byteOffset, dataView.byteLength));
}
