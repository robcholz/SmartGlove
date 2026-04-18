use std::thread;
use std::time::{Duration, Instant};

use esp_idf_svc::ws::{
    client::{EspWebSocketClient, EspWebSocketClientConfig, EspWebSocketTransport},
    FrameType,
};
use serde::Serialize;

use crate::{
    protocol::{build_event_json, build_online_json},
    NetworkError,
};

pub const DEFAULT_SAMPLE_RATE_HZ: u16 = 100;
pub const DEFAULT_BATCH_SAMPLES: usize = 10;
pub const DEFAULT_FLUSH_INTERVAL_MS: u64 = 100;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FlexReadings {
    pub thumb: f32,
    pub index: f32,
    pub middle: f32,
    pub ring: f32,
    pub pinky: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SensorSample {
    pub imu_acc: [f32; 3],
    pub vel: [f32; 3],
    pub flex: FlexReadings,
}

#[derive(Serialize)]
struct BufferedSensorSample {
    imu_acc: [f32; 3],
    vel: [f32; 3],
    flex: FlexReadings,
}

#[derive(Serialize)]
struct SensorBatchPayload<'a> {
    kind: &'static str,
    device_id: &'a str,
    sample_rate_hz: u16,
    start_tick_ms: u64,
    samples: &'a [BufferedSensorSample],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StatusReportConfig<'a> {
    pub websocket_url: &'a str,
    pub sample_rate_hz: u16,
    pub batch_samples: usize,
    pub flush_interval: Duration,
}

impl<'a> Default for StatusReportConfig<'a> {
    fn default() -> Self {
        Self {
            websocket_url: "",
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            batch_samples: DEFAULT_BATCH_SAMPLES,
            flush_interval: Duration::from_millis(DEFAULT_FLUSH_INTERVAL_MS),
        }
    }
}

pub struct StatusReporter {
    device_id: String,
    websocket: EspWebSocketClient<'static>,
    sample_rate_hz: u16,
    batch_samples: usize,
    flush_interval: Duration,
    started_at: Instant,
    last_flush_at: Instant,
    batch_start_tick_ms: Option<u64>,
    buffer: Vec<SensorSample>,
}

pub fn connect_status_reporter(
    config: &StatusReportConfig<'_>,
    device_id: &str,
) -> Result<StatusReporter, NetworkError> {
    if config.websocket_url.is_empty() {
        return Err(NetworkError::InvalidConfig(
            "websocket url must not be empty",
        ));
    }

    if config.batch_samples == 0 {
        return Err(NetworkError::InvalidConfig(
            "batch_samples must be greater than zero",
        ));
    }

    if config.sample_rate_hz == 0 {
        return Err(NetworkError::InvalidConfig(
            "sample_rate_hz must be greater than zero",
        ));
    }

    let transport = if config.websocket_url.starts_with("wss://") {
        EspWebSocketTransport::TransportOverSSL
    } else {
        EspWebSocketTransport::TransportOverTCP
    };

    let client_config = EspWebSocketClientConfig {
        transport,
        buffer_size: 4096,
        network_timeout_ms: Duration::from_secs(10),
        reconnect_timeout_ms: Duration::from_secs(5),
        ping_interval_sec: Duration::from_secs(30),
        pingpong_timeout_sec: Duration::from_secs(10),
        ..Default::default()
    };

    let websocket = EspWebSocketClient::new(
        config.websocket_url,
        &client_config,
        Duration::from_secs(10),
        |event| {
            if let Err(err) = event {
                log::warn!("websocket callback error: {err}");
            }
        },
    )
    .map_err(NetworkError::Io)?;

    wait_until_connected(&websocket)?;

    Ok(StatusReporter {
        device_id: device_id.to_owned(),
        websocket,
        sample_rate_hz: config.sample_rate_hz,
        batch_samples: config.batch_samples,
        flush_interval: config.flush_interval,
        started_at: Instant::now(),
        last_flush_at: Instant::now(),
        batch_start_tick_ms: None,
        buffer: Vec::with_capacity(config.batch_samples),
    })
}

impl StatusReporter {
    pub fn send_online(&mut self) -> Result<(), NetworkError> {
        let payload = build_online_json(&self.device_id);
        self.send_text_json(&payload)
    }

    pub fn send_event(&mut self, event_name: &str, payload_json: &str) -> Result<(), NetworkError> {
        let payload = build_event_json(&self.device_id, event_name, payload_json)?;
        self.send_text_json(&payload)
    }

    pub fn push_sensor_sample(&mut self, sample: SensorSample) -> Result<bool, NetworkError> {
        if self.buffer.is_empty() {
            self.batch_start_tick_ms = Some(self.started_at.elapsed().as_millis() as u64);
        }

        self.buffer.push(sample);

        let due_by_size = self.buffer.len() >= self.batch_samples;
        let due_by_time = self.last_flush_at.elapsed() >= self.flush_interval;

        if due_by_size || due_by_time {
            return self.flush();
        }

        Ok(false)
    }

    pub fn flush(&mut self) -> Result<bool, NetworkError> {
        if self.buffer.is_empty() {
            return Ok(false);
        }

        let payload = build_sensor_batch_json(
            &self.device_id,
            self.sample_rate_hz,
            self.batch_start_tick_ms.unwrap_or(0),
            &self.buffer,
        );
        self.send_text_json(&payload)?;
        self.buffer.clear();
        self.last_flush_at = Instant::now();
        self.batch_start_tick_ms = None;

        Ok(true)
    }

    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }

    fn send_text_json(&mut self, payload: &str) -> Result<(), NetworkError> {
        if !self.websocket.is_connected() {
            return Err(NetworkError::WebSocketNotConnected);
        }

        self.websocket
            .send(FrameType::Text(false), payload.as_bytes())
            .map_err(NetworkError::WebSocket)?;

        Ok(())
    }
}

pub fn mock_sensor_sample(sequence: u64) -> SensorSample {
    let t = sequence as f32 / DEFAULT_SAMPLE_RATE_HZ as f32;
    let wave = |scale: f32, phase: f32| -> f32 { ((t * scale) + phase).sin() };

    SensorSample {
        imu_acc: [wave(2.0, 0.1), wave(1.3, 0.7), 0.98 + wave(0.7, 1.2) * 0.05],
        vel: [
            wave(1.1, 0.0) * 0.2,
            wave(0.9, 0.5) * 0.25,
            wave(1.7, 1.1) * 0.15,
        ],
        flex: FlexReadings {
            thumb: 0.4 + wave(0.8, 0.0) * 0.1,
            index: 0.5 + wave(0.9, 0.3) * 0.1,
            middle: 0.45 + wave(1.0, 0.6) * 0.1,
            ring: 0.35 + wave(1.1, 0.9) * 0.1,
            pinky: 0.3 + wave(1.2, 1.2) * 0.1,
        },
    }
}

fn wait_until_connected(websocket: &EspWebSocketClient<'static>) -> Result<(), NetworkError> {
    for _ in 0..20 {
        if websocket.is_connected() {
            return Ok(());
        }

        thread::sleep(Duration::from_millis(100));
    }

    Err(NetworkError::WebSocketNotConnected)
}

fn build_sensor_batch_json(
    device_id: &str,
    sample_rate_hz: u16,
    start_tick_ms: u64,
    samples: &[SensorSample],
) -> String {
    let buffered_samples: Vec<BufferedSensorSample> = samples
        .iter()
        .map(|sample| BufferedSensorSample {
            imu_acc: sample.imu_acc,
            vel: sample.vel,
            flex: sample.flex,
        })
        .collect();

    serde_json::to_string(&SensorBatchPayload {
        kind: "sensor.batch",
        device_id,
        sample_rate_hz,
        start_tick_ms,
        samples: &buffered_samples,
    })
    .expect("sensor batch json")
}
