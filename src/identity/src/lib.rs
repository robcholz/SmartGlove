use core::fmt;

use esp_idf_svc::nvs::{EspDefaultNvs, EspDefaultNvsPartition};
use esp_idf_svc::sys::{self, esp, EspError};

pub const DEVICE_ID_LEN: usize = 6;
pub const DEVICE_ID_HEX_LEN: usize = DEVICE_ID_LEN * 2;
pub const DEVICE_SECRET_LEN: usize = 32;

const NAMESPACE: &str = "identity";
const DEVICE_SECRET_KEY: &str = "device_secret";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceId([u8; DEVICE_ID_LEN]);

impl DeviceId {
    pub fn from_factory_mac() -> Result<Self, EspError> {
        let mut mac = [0_u8; DEVICE_ID_LEN];

        esp!(unsafe { sys::esp_efuse_mac_get_default(mac.as_mut_ptr()) })?;

        Ok(Self(mac))
    }

    pub fn as_bytes(&self) -> &[u8; DEVICE_ID_LEN] {
        &self.0
    }

    pub fn to_hex_string(&self) -> String {
        encode_hex(&self.0)
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex_string())
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct DeviceSecret([u8; DEVICE_SECRET_LEN]);

impl DeviceSecret {
    fn load_or_create(nvs: &EspDefaultNvs) -> Result<Self, EspError> {
        if let Some(secret_len) = nvs.blob_len(DEVICE_SECRET_KEY)? {
            if secret_len != DEVICE_SECRET_LEN {
                return Err(EspError::from_infallible::<{ sys::ESP_ERR_INVALID_SIZE }>());
            }

            let mut secret = [0_u8; DEVICE_SECRET_LEN];
            let stored = nvs
                .get_blob(DEVICE_SECRET_KEY, &mut secret)?
                .ok_or_else(|| EspError::from_infallible::<{ sys::ESP_FAIL }>())?;

            if stored.len() != DEVICE_SECRET_LEN {
                return Err(EspError::from_infallible::<{ sys::ESP_ERR_INVALID_SIZE }>());
            }

            Ok(Self(secret))
        } else {
            let mut secret = [0_u8; DEVICE_SECRET_LEN];
            fill_device_secret(&mut secret);
            nvs.set_blob(DEVICE_SECRET_KEY, &secret)?;

            Ok(Self(secret))
        }
    }

    pub fn as_bytes(&self) -> &[u8; DEVICE_SECRET_LEN] {
        &self.0
    }

    pub fn to_hex_string(&self) -> String {
        encode_hex(&self.0)
    }

    pub fn fingerprint(&self) -> String {
        encode_hex(&self.0[..4])
    }
}

impl fmt::Debug for DeviceSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceSecret")
            .field("len", &DEVICE_SECRET_LEN)
            .field("fingerprint", &self.fingerprint())
            .finish()
    }
}

pub struct DeviceIdentity {
    device_id: DeviceId,
    device_secret: DeviceSecret,
}

impl DeviceIdentity {
    pub fn load() -> Result<Self, EspError> {
        let device_id = DeviceId::from_factory_mac()?;
        let nvs_partition = EspDefaultNvsPartition::take()?;
        let nvs = EspDefaultNvs::new(nvs_partition, NAMESPACE, true)?;
        let device_secret = DeviceSecret::load_or_create(&nvs)?;

        Ok(Self {
            device_id,
            device_secret,
        })
    }

    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    pub fn device_secret(&self) -> &DeviceSecret {
        &self.device_secret
    }

    pub fn log_label(&self) -> String {
        format!(
            "device_id={}, secret_fp={}",
            self.device_id,
            self.device_secret.fingerprint()
        )
    }
}

fn fill_device_secret(secret: &mut [u8; DEVICE_SECRET_LEN]) {
    unsafe {
        sys::esp_fill_random(secret.as_mut_ptr().cast(), secret.len());
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let mut out = String::with_capacity(bytes.len() * 2);

    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }

    out
}
