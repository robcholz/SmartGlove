use identity::DeviceIdentity;
use smart_glove::network::registration::{register_device, RegistrationConfig};

const REGISTRATION_ENDPOINT: &str = "http://192.168.1.100:3000/devices/register";

fn main() {
    // It is necessary to call this function once. Otherwise, some patches to the runtime
    // implemented by esp-idf-sys might not link properly. See https://github.com/esp-rs/esp-idf-template/issues/71
    esp_idf_svc::sys::link_patches();

    // Bind the log crate to the ESP Logging facilities
    esp_idf_svc::log::EspLogger::initialize_default();

    let identity = DeviceIdentity::load().expect("failed to load device identity");
    log::info!("Loaded {}", identity.log_label());

    let config = RegistrationConfig {
        endpoint: REGISTRATION_ENDPOINT,
    };
    let response = register_device(&identity, &config).expect("failed to register device");

    log::info!(
        "registration response: registered={}, websocket_url={}, session_token_len={}",
        response.registered,
        response.websocket_url,
        response.session_token.len()
    );
}
