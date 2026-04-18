use std::fmt;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
use enumset::enum_set;

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
use esp_idf_svc::bt::ble::gap::{AdvConfiguration, AppearanceCategory, BleGapEvent, EspBleGap};
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
use esp_idf_svc::bt::ble::gatt::server::{EspGatts, GattsEvent};
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
use esp_idf_svc::bt::ble::gatt::{
    AutoResponse, GattCharacteristic, GattDescriptor, GattId, GattInterface, GattResponse,
    GattServiceId, GattStatus, Handle, Permission, Property,
};
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
use esp_idf_svc::bt::{Ble, BtDriver, BtStatus, BtUuid};
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
use esp_idf_svc::eventloop::EspSystemEventLoop;
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
use esp_idf_svc::hal::modem::{BluetoothModemPeripheral, WifiModemPeripheral};
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::sys::EspError;
use esp_idf_svc::wifi::{AuthMethod, BlockingWifi, ClientConfiguration, Configuration, EspWifi};
use log::{info, warn};

pub type ProvisionBleMessageCallback = Box<dyn FnMut(&[u8]) + Send>;
pub type ProvisionWifiStatusCallback = Box<dyn FnMut(&WifiProvisioningStatus) + Send>;
pub type ProvisionErrorCallback = Box<dyn FnMut(&ProvisionError) + Send>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvisioningCredentials {
    pub ssid: String,
    pub password: String,
}

impl ProvisioningCredentials {
    pub fn from_payload(payload: &[u8]) -> Result<Self, ProvisionError> {
        if let Some((ssid, password)) = payload
            .splitn(2, |byte| *byte == b'\0')
            .collect::<Vec<_>>()
            .split_first()
            .and_then(|(ssid, rest)| rest.first().map(|password| (*ssid, *password)))
        {
            return Self::from_parts(ssid, password);
        }

        if let Some((ssid, password)) = split_once_bytes(payload, b'\n') {
            return Self::from_parts(ssid, password);
        }

        let message =
            core::str::from_utf8(payload).map_err(|_| ProvisionError::InvalidCredentialsPayload)?;

        if let (Some(ssid), Some(password)) = (
            extract_key_value(message, "ssid"),
            extract_key_value(message, "password"),
        ) {
            return Ok(Self { ssid, password });
        }

        Err(ProvisionError::InvalidCredentialsPayload)
    }

    fn from_parts(ssid: &[u8], password: &[u8]) -> Result<Self, ProvisionError> {
        let ssid = core::str::from_utf8(trim_ascii_whitespace(ssid))
            .map_err(|_| ProvisionError::InvalidCredentialsPayload)?;
        let password = core::str::from_utf8(trim_ascii_whitespace(password))
            .map_err(|_| ProvisionError::InvalidCredentialsPayload)?;

        if ssid.is_empty() {
            return Err(ProvisionError::InvalidCredentialsPayload);
        }

        Ok(Self {
            ssid: ssid.to_owned(),
            password: password.to_owned(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WifiProvisioningStatus {
    BroadcastStarting,
    Broadcasting,
    BroadcastStopped,
    CredentialsReceived,
    Connecting,
    Connected,
    ConnectionFailed(String),
}

impl WifiProvisioningStatus {
    fn as_ble_payload(&self) -> Vec<u8> {
        match self {
            Self::BroadcastStarting => b"broadcast_starting".to_vec(),
            Self::Broadcasting => b"broadcasting".to_vec(),
            Self::BroadcastStopped => b"broadcast_stopped".to_vec(),
            Self::CredentialsReceived => b"credentials_received".to_vec(),
            Self::Connecting => b"connecting".to_vec(),
            Self::Connected => b"connected".to_vec(),
            Self::ConnectionFailed(reason) => format!("connection_failed:{reason}").into_bytes(),
        }
    }
}

#[derive(Debug)]
pub enum ProvisionError {
    Esp(EspError),
    InvalidCredentialsPayload,
    ConnectionFailed(String),
    MissingGattInterface,
    MissingCharacteristicHandle,
    NoBleConnection,
    Poisoned(&'static str),
    Timeout,
    Unsupported(&'static str),
}

impl fmt::Display for ProvisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Esp(err) => write!(f, "{err}"),
            Self::InvalidCredentialsPayload => write!(f, "invalid provisioning credential payload"),
            Self::ConnectionFailed(reason) => write!(f, "wifi connection failed: {reason}"),
            Self::MissingGattInterface => write!(f, "missing GATT interface"),
            Self::MissingCharacteristicHandle => write!(f, "missing GATT characteristic handle"),
            Self::NoBleConnection => write!(f, "no BLE peer is connected"),
            Self::Poisoned(name) => write!(f, "shared state poisoned: {name}"),
            Self::Timeout => write!(f, "provisioning timed out"),
            Self::Unsupported(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ProvisionError {}

impl From<EspError> for ProvisionError {
    fn from(value: EspError) -> Self {
        Self::Esp(value)
    }
}

#[derive(Default)]
pub struct ProvisioningCallbacks {
    pub ble_message: Option<ProvisionBleMessageCallback>,
    pub wifi_status: Option<ProvisionWifiStatusCallback>,
    pub error: Option<ProvisionErrorCallback>,
}

pub trait ProvisionCapabilityProvider {
    fn broadcast(&mut self) -> Result<(), ProvisionError>;

    fn turn_off_broadcast(&mut self) -> Result<(), ProvisionError>;

    fn connect(
        &mut self,
        ssid: &str,
        password: &str,
    ) -> Result<WifiProvisioningStatus, ProvisionError>;

    fn notify_wifi_status(&mut self, status: WifiProvisioningStatus) -> Result<(), ProvisionError>;

    fn set_ble_message_callback(&mut self, callback: Option<ProvisionBleMessageCallback>);

    fn set_wifi_status_callback(&mut self, callback: Option<ProvisionWifiStatusCallback>);

    fn set_error_callback(&mut self, callback: Option<ProvisionErrorCallback>);
}

pub trait Provision {
    fn start(&mut self) -> Result<(), ProvisionError>;

    fn stop(&mut self) -> Result<(), ProvisionError>;
}

pub struct BasicProvision<P> {
    provider: P,
    advertising_window: Duration,
}

impl<P> BasicProvision<P> {
    pub fn new(provider: P, advertising_window: Duration) -> Self {
        Self {
            provider,
            advertising_window,
        }
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }
}

impl<P> Provision for BasicProvision<P>
where
    P: ProvisionCapabilityProvider,
{
    fn start(&mut self) -> Result<(), ProvisionError> {
        let (credentials_tx, credentials_rx) = mpsc::channel::<ProvisioningCredentials>();

        self.provider
            .set_ble_message_callback(Some(Box::new(move |payload| {
                if let Ok(credentials) = ProvisioningCredentials::from_payload(payload) {
                    let _ = credentials_tx.send(credentials);
                }
            })));

        self.provider.broadcast()?;

        let credentials = match credentials_rx.recv_timeout(self.advertising_window) {
            Ok(credentials) => credentials,
            Err(_) => {
                // let _ = self.provider.turn_off_broadcast();
                return Err(ProvisionError::Timeout);
            }
        };

        // self.provider.turn_off_broadcast()?;
        match self
            .provider
            .connect(&credentials.ssid, &credentials.password)?
        {
            WifiProvisioningStatus::Connected => Ok(()),
            WifiProvisioningStatus::ConnectionFailed(reason) => {
                Err(ProvisionError::ConnectionFailed(reason))
            }
            other => Err(ProvisionError::Unsupported(match other {
                WifiProvisioningStatus::BroadcastStarting => {
                    "unexpected broadcast_starting status after Wi-Fi connect"
                }
                WifiProvisioningStatus::Broadcasting => {
                    "unexpected broadcasting status after Wi-Fi connect"
                }
                WifiProvisioningStatus::BroadcastStopped => {
                    "unexpected broadcast_stopped status after Wi-Fi connect"
                }
                WifiProvisioningStatus::CredentialsReceived => {
                    "unexpected credentials_received status after Wi-Fi connect"
                }
                WifiProvisioningStatus::Connecting => {
                    "unexpected connecting status after Wi-Fi connect"
                }
                WifiProvisioningStatus::Connected | WifiProvisioningStatus::ConnectionFailed(_) => {
                    unreachable!()
                }
            })),
        }
    }

    fn stop(&mut self) -> Result<(), ProvisionError> {
        self.provider.turn_off_broadcast()
    }
}

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
const PROVISION_APP_ID: u16 = 0;
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
const PROVISION_SERVICE_UUID: u16 = 0x12FF;
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
const PROVISION_CREDENTIALS_UUID: u16 = 0xFF01;
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
const PROVISION_STATUS_UUID: u16 = 0xFF02;
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
const CLIENT_CONFIG_DESCRIPTOR_UUID: u16 = 0x2902;
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
const PROVISION_MANUFACTURER_ID: u16 = 0xFFFF;
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
const PROVISION_MAGIC: &[u8] = b"SG";
#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
const PROVISION_MANUFACTURER_DATA: [u8; 4] = [
    (PROVISION_MANUFACTURER_ID & 0x00FF) as u8,
    (PROVISION_MANUFACTURER_ID >> 8) as u8,
    PROVISION_MAGIC[0],
    PROVISION_MAGIC[1],
];

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
#[derive(Default)]
struct EspProvisionState {
    gatt_if: Option<GattInterface>,
    service_handle: Option<Handle>,
    credentials_handle: Option<Handle>,
    status_handle: Option<Handle>,
    conn_id: Option<u16>,
    stack_initialized: bool,
    service_started: bool,
    advertising_configured: bool,
    advertising_requested: bool,
}

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
pub struct EspProvisionCapabilityProvider<'d> {
    device_name: String,
    gap: Arc<EspBleGap<'d, Ble, Arc<BtDriver<'d, Ble>>>>,
    gatts: Arc<EspGatts<'d, Ble, Arc<BtDriver<'d, Ble>>>>,
    wifi: Arc<Mutex<BlockingWifi<EspWifi<'d>>>>,
    state: Arc<Mutex<EspProvisionState>>,
    callbacks: Arc<Mutex<ProvisioningCallbacks>>,
}

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
impl<'d> EspProvisionCapabilityProvider<'d> {
    pub fn new<W, B>(
        device_name: impl Into<String>,
        wifi_modem: W,
        bluetooth_modem: B,
        sysloop: EspSystemEventLoop,
        nvs: Option<EspDefaultNvsPartition>,
    ) -> Result<Self, ProvisionError>
    where
        W: WifiModemPeripheral + 'd,
        B: BluetoothModemPeripheral + 'd,
    {
        let wifi = BlockingWifi::wrap(
            EspWifi::new(wifi_modem, sysloop.clone(), nvs.clone())?,
            sysloop,
        )?;
        let bt = Arc::new(BtDriver::new(bluetooth_modem, nvs)?);

        Self::from_parts(
            device_name,
            EspBleGap::new(bt.clone())?,
            EspGatts::new(bt)?,
            wifi,
        )
    }

    pub fn from_parts(
        device_name: impl Into<String>,
        gap: EspBleGap<'d, Ble, Arc<BtDriver<'d, Ble>>>,
        gatts: EspGatts<'d, Ble, Arc<BtDriver<'d, Ble>>>,
        wifi: BlockingWifi<EspWifi<'d>>,
    ) -> Result<Self, ProvisionError> {
        let provider = Self {
            device_name: device_name.into(),
            gap: Arc::new(gap),
            gatts: Arc::new(gatts),
            wifi: Arc::new(Mutex::new(wifi)),
            state: Arc::new(Mutex::new(EspProvisionState::default())),
            callbacks: Arc::new(Mutex::new(ProvisioningCallbacks::default())),
        };

        provider.install_callbacks()?;

        Ok(provider)
    }

    fn install_callbacks(&self) -> Result<(), ProvisionError> {
        let state = Arc::clone(&self.state);
        let callbacks = Arc::clone(&self.callbacks);
        let gatts = Arc::clone(&self.gatts);
        let gap = Arc::clone(&self.gap);
        let device_name = self.device_name.clone();

        unsafe {
            self.gatts.subscribe_nonstatic(move |(gatt_if, event)| {
                if let Err(err) = Self::handle_gatts_event(
                    &gatts,
                    &gap,
                    &state,
                    &callbacks,
                    &device_name,
                    gatt_if,
                    event,
                ) {
                    emit_error(&callbacks, err);
                }
            })?;
        }

        let state = Arc::clone(&self.state);
        let callbacks = Arc::clone(&self.callbacks);
        let gap = Arc::clone(&self.gap);

        unsafe {
            self.gap.subscribe_nonstatic(move |event| {
                if let Err(err) = Self::handle_gap_event(&gap, &state, &callbacks, event) {
                    emit_error(&callbacks, err);
                }
            })?;
        }

        Ok(())
    }

    fn handle_gap_event(
        gap: &Arc<EspBleGap<'d, Ble, Arc<BtDriver<'d, Ble>>>>,
        state: &Arc<Mutex<EspProvisionState>>,
        callbacks: &Arc<Mutex<ProvisioningCallbacks>>,
        event: BleGapEvent,
    ) -> Result<(), ProvisionError> {
        info!("BLE GAP event: {event:?}");

        match event {
            BleGapEvent::AdvertisingConfigured(BtStatus::Success) => {
                {
                    let mut state = lock_mutex(state, "esp_gap_state")?;
                    state.advertising_configured = true;
                }

                Self::maybe_start_advertising(gap, state)?;
            }
            BleGapEvent::AdvertisingConfigured(status) => {
                emit_error(
                    callbacks,
                    ProvisionError::Unsupported(match status {
                        BtStatus::Success => "unexpected advertising configuration status",
                        _ => "failed to configure BLE advertising payload",
                    }),
                );
            }
            BleGapEvent::AdvertisingStarted(BtStatus::Success) => {
                emit_status(callbacks, WifiProvisioningStatus::Broadcasting);
            }
            BleGapEvent::AdvertisingStopped(BtStatus::Success) => {
                emit_status(callbacks, WifiProvisioningStatus::BroadcastStopped);
            }
            BleGapEvent::AdvertisingStarted(_) | BleGapEvent::AdvertisingStopped(_) => {
                emit_error(
                    callbacks,
                    ProvisionError::Unsupported("BLE advertising transition failed"),
                );
            }
            _ => {}
        }

        Ok(())
    }

    fn handle_gatts_event(
        gatts: &Arc<EspGatts<'d, Ble, Arc<BtDriver<'d, Ble>>>>,
        gap: &Arc<EspBleGap<'d, Ble, Arc<BtDriver<'d, Ble>>>>,
        state: &Arc<Mutex<EspProvisionState>>,
        callbacks: &Arc<Mutex<ProvisioningCallbacks>>,
        device_name: &str,
        gatt_if: GattInterface,
        event: GattsEvent,
    ) -> Result<(), ProvisionError> {
        info!("BLE GATTS event: {event:?}");

        match event {
            GattsEvent::ServiceRegistered {
                status: GattStatus::Ok,
                app_id,
            } if app_id == PROVISION_APP_ID => {
                {
                    let mut state = lock_mutex(state, "esp_gatts_state")?;
                    state.gatt_if = Some(gatt_if);
                }

                info!("BLE setup: set device name");
                gap.set_device_name(device_name)?;
                info!("BLE setup: configure advertising payload");
                gap.set_adv_conf(&adv_configuration())?;
                info!("BLE setup: create GATT service");
                gatts.create_service(gatt_if, &provision_service_id(), 8)?;
            }
            GattsEvent::ServiceCreated {
                status: GattStatus::Ok,
                service_handle,
                ..
            } => {
                {
                    let mut state = lock_mutex(state, "esp_gatts_state")?;
                    state.service_handle = Some(service_handle);
                }

                gatts.add_characteristic(service_handle, &credentials_characteristic(), &[])?;
            }
            GattsEvent::CharacteristicAdded {
                status: GattStatus::Ok,
                service_handle,
                attr_handle,
                char_uuid,
            } if char_uuid == BtUuid::uuid16(PROVISION_CREDENTIALS_UUID) => {
                {
                    let mut state = lock_mutex(state, "esp_gatts_state")?;
                    state.credentials_handle = Some(attr_handle);
                }

                gatts.add_characteristic(service_handle, &status_characteristic(), b"idle")?;
            }
            GattsEvent::CharacteristicAdded {
                status: GattStatus::Ok,
                service_handle,
                attr_handle,
                char_uuid,
            } if char_uuid == BtUuid::uuid16(PROVISION_STATUS_UUID) => {
                {
                    let mut state = lock_mutex(state, "esp_gatts_state")?;
                    state.status_handle = Some(attr_handle);
                }

                gatts.add_descriptor(service_handle, &status_descriptor())?;
            }
            GattsEvent::DescriptorAdded {
                status: GattStatus::Ok,
                service_handle,
                ..
            } => {
                gatts.start_service(service_handle)?;
            }
            GattsEvent::ServiceStarted {
                status: GattStatus::Ok,
                ..
            } => {
                {
                    let mut state = lock_mutex(state, "esp_gatts_state")?;
                    state.service_started = true;
                }

                Self::maybe_start_advertising(gap, state)?;
            }
            GattsEvent::PeerConnected { conn_id, .. } => {
                let mut state = lock_mutex(state, "esp_gatts_state")?;
                state.conn_id = Some(conn_id);
            }
            GattsEvent::PeerDisconnected { .. } => {
                let mut state = lock_mutex(state, "esp_gatts_state")?;
                state.conn_id = None;
            }
            GattsEvent::Write {
                conn_id,
                trans_id,
                handle,
                need_rsp,
                value,
                ..
            } => {
                let credentials_handle = {
                    let state = lock_mutex(state, "esp_gatts_state")?;
                    state.credentials_handle
                };

                if need_rsp {
                    let mut response = GattResponse::new();
                    response.attr_handle(handle);
                    gatts.send_response(
                        gatt_if,
                        conn_id,
                        trans_id,
                        GattStatus::Ok,
                        Some(&response),
                    )?;
                }

                if Some(handle) == credentials_handle {
                    emit_status(callbacks, WifiProvisioningStatus::CredentialsReceived);
                    emit_ble_message(callbacks, value);
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn maybe_start_advertising(
        gap: &Arc<EspBleGap<'d, Ble, Arc<BtDriver<'d, Ble>>>>,
        state: &Arc<Mutex<EspProvisionState>>,
    ) -> Result<(), ProvisionError> {
        let should_start = {
            let state = lock_mutex(state, "esp_gap_state")?;
            state.advertising_requested && state.service_started && state.advertising_configured
        };

        if should_start {
            gap.start_advertising()?;
        }

        Ok(())
    }

    fn configure_stack_if_needed(&mut self) -> Result<(), ProvisionError> {
        let should_configure = {
            let mut state = lock_mutex(&self.state, "esp_provider_state")?;
            if state.stack_initialized {
                false
            } else {
                state.stack_initialized = true;
                true
            }
        };

        if should_configure {
            self.gatts.register_app(PROVISION_APP_ID)?;
        }

        Ok(())
    }
}

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
impl<'d> ProvisionCapabilityProvider for EspProvisionCapabilityProvider<'d> {
    fn broadcast(&mut self) -> Result<(), ProvisionError> {
        self.configure_stack_if_needed()?;

        {
            let mut state = lock_mutex(&self.state, "esp_provider_state")?;
            state.advertising_requested = true;
        }

        emit_status(&self.callbacks, WifiProvisioningStatus::BroadcastStarting);
        Self::maybe_start_advertising(&self.gap, &self.state)?;

        Ok(())
    }

    fn turn_off_broadcast(&mut self) -> Result<(), ProvisionError> {
        {
            let mut state = lock_mutex(&self.state, "esp_provider_state")?;
            state.advertising_requested = false;
        }

        self.gap.stop_advertising()?;
        Ok(())
    }

    fn connect(
        &mut self,
        ssid: &str,
        password: &str,
    ) -> Result<WifiProvisioningStatus, ProvisionError> {
        self.notify_wifi_status(WifiProvisioningStatus::Connecting)?;

        let auth_method = if password.is_empty() {
            AuthMethod::None
        } else {
            AuthMethod::WPA2Personal
        };

        let client = ClientConfiguration {
            ssid: ssid
                .try_into()
                .map_err(|_| ProvisionError::InvalidCredentialsPayload)?,
            password: password
                .try_into()
                .map_err(|_| ProvisionError::InvalidCredentialsPayload)?,
            auth_method,
            ..Default::default()
        };

        let mut wifi = lock_mutex(&self.wifi, "wifi")?;

        wifi.set_configuration(&Configuration::Client(client))?;

        if !wifi.is_started()? {
            wifi.start()?;
        }

        if wifi.is_connected()? {
            wifi.disconnect()?;
        }

        let result = match (|| -> Result<(), ProvisionError> {
            wifi.connect()?;
            wifi.wait_netif_up()?;
            Ok(())
        })() {
            Ok(()) => WifiProvisioningStatus::Connected,
            Err(err) => {
                let message = err.to_string();
                WifiProvisioningStatus::ConnectionFailed(message)
            }
        };

        drop(wifi);

        self.notify_wifi_status(result.clone())?;
        Ok(result)
    }

    fn notify_wifi_status(&mut self, status: WifiProvisioningStatus) -> Result<(), ProvisionError> {
        emit_status(&self.callbacks, status.clone());

        let payload = status.as_ble_payload();
        let (gatt_if, status_handle, conn_id) = {
            let state = lock_mutex(&self.state, "esp_provider_state")?;
            (state.gatt_if, state.status_handle, state.conn_id)
        };

        if let (Some(gatt_if), Some(status_handle), Some(conn_id)) =
            (gatt_if, status_handle, conn_id)
        {
            self.gatts.set_attr(status_handle, &payload)?;
            self.gatts
                .notify(gatt_if, conn_id, status_handle, &payload)?;
        }

        Ok(())
    }

    fn set_ble_message_callback(&mut self, callback: Option<ProvisionBleMessageCallback>) {
        if let Ok(mut callbacks) = self.callbacks.lock() {
            callbacks.ble_message = callback;
        }
    }

    fn set_wifi_status_callback(&mut self, callback: Option<ProvisionWifiStatusCallback>) {
        if let Ok(mut callbacks) = self.callbacks.lock() {
            callbacks.wifi_status = callback;
        }
    }

    fn set_error_callback(&mut self, callback: Option<ProvisionErrorCallback>) {
        if let Ok(mut callbacks) = self.callbacks.lock() {
            callbacks.error = callback;
        }
    }
}

fn split_once_bytes(bytes: &[u8], delimiter: u8) -> Option<(&[u8], &[u8])> {
    bytes
        .iter()
        .position(|byte| *byte == delimiter)
        .map(|index| {
            let (left, right) = bytes.split_at(index);
            (left, &right[1..])
        })
}

fn trim_ascii_whitespace(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map(|index| index + 1)
        .unwrap_or(start);

    &bytes[start..end]
}

fn extract_key_value(message: &str, key: &str) -> Option<String> {
    extract_json_value(message, key).or_else(|| extract_assignment_value(message, key))
}

fn extract_json_value(message: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let start = message.find(&pattern)?;
    let rest = &message[start + pattern.len()..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}

fn extract_assignment_value(message: &str, key: &str) -> Option<String> {
    for segment in message.split(['\n', ';', '&']) {
        if let Some((candidate_key, candidate_value)) = segment.split_once('=') {
            if candidate_key.trim() == key {
                return Some(candidate_value.trim().to_owned());
            }
        }

        if let Some((candidate_key, candidate_value)) = segment.split_once(':') {
            if candidate_key.trim() == key {
                return Some(candidate_value.trim().to_owned());
            }
        }
    }

    None
}

fn lock_mutex<'a, T>(
    mutex: &'a Mutex<T>,
    name: &'static str,
) -> Result<std::sync::MutexGuard<'a, T>, ProvisionError> {
    mutex.lock().map_err(|_| ProvisionError::Poisoned(name))
}

fn emit_ble_message(callbacks: &Arc<Mutex<ProvisioningCallbacks>>, payload: &[u8]) {
    match callbacks.lock() {
        Ok(mut callbacks) => {
            if let Some(callback) = callbacks.ble_message.as_mut() {
                callback(payload);
            }
        }
        Err(_) => warn!("failed to acquire BLE message callback lock"),
    }
}

fn emit_status(callbacks: &Arc<Mutex<ProvisioningCallbacks>>, status: WifiProvisioningStatus) {
    match callbacks.lock() {
        Ok(mut callbacks) => {
            if let Some(callback) = callbacks.wifi_status.as_mut() {
                callback(&status);
            }
        }
        Err(_) => warn!("failed to acquire Wi-Fi status callback lock"),
    }
}

fn emit_error(callbacks: &Arc<Mutex<ProvisioningCallbacks>>, error: ProvisionError) {
    match callbacks.lock() {
        Ok(mut callbacks) => {
            if let Some(callback) = callbacks.error.as_mut() {
                callback(&error);
            } else {
                warn!("provisioning error: {error}");
            }
        }
        Err(_) => warn!("failed to acquire provisioning error callback lock"),
    }
}

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
fn provision_service_id() -> GattServiceId {
    GattServiceId {
        id: GattId {
            uuid: BtUuid::uuid16(PROVISION_SERVICE_UUID),
            inst_id: 0,
        },
        is_primary: true,
    }
}

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
fn credentials_characteristic() -> GattCharacteristic {
    GattCharacteristic::new(
        BtUuid::uuid16(PROVISION_CREDENTIALS_UUID),
        enum_set!(Permission::Read | Permission::Write),
        enum_set!(Property::Write | Property::WriteNoResponse),
        128,
        AutoResponse::ByApp,
    )
}

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
fn status_characteristic() -> GattCharacteristic {
    GattCharacteristic::new(
        BtUuid::uuid16(PROVISION_STATUS_UUID),
        enum_set!(Permission::Read),
        enum_set!(Property::Read | Property::Notify),
        128,
        AutoResponse::ByGatt,
    )
}

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
fn status_descriptor() -> GattDescriptor {
    GattDescriptor::new(
        BtUuid::uuid16(CLIENT_CONFIG_DESCRIPTOR_UUID),
        enum_set!(Permission::Read | Permission::Write),
    )
}

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
fn adv_configuration() -> AdvConfiguration<'static> {
    AdvConfiguration {
        include_name: true,
        include_txpower: false,
        appearance: AppearanceCategory::Unknown,
        flag: 2,
        service_uuid: None,
        manufacturer_data: Some(&PROVISION_MANUFACTURER_DATA),
        ..Default::default()
    }
}

#[cfg(not(esp_idf_btdm_ctrl_mode_br_edr_only))]
pub fn log_provider_ready() {
    info!("ESP provisioning capability provider ready");
}
