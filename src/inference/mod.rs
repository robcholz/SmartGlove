#[allow(dead_code)]
mod generated;

use core::ffi::CStr;
use std::cmp::Ordering;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub use generated::{
    ESPDL_INPUT_EXPONENT, ESPDL_INPUT_SHAPE, ESPDL_OUTPUT_EXPONENT, ESPDL_OUTPUT_SHAPE,
    ESPDL_TEST_INPUT, ESPDL_TEST_OUTPUT, FEATURE_MEANS, FEATURE_SCALES, MODEL_FEATURE_COUNT,
    MODEL_LABELS, MODEL_OUTPUT_COUNT, MODEL_WINDOW_SIZE,
};

#[derive(Debug)]
pub enum InferenceError {
    ComponentUnavailable,
    Esp(i32),
    InvalidResult(&'static str),
}

impl Display for InferenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ComponentUnavailable => write!(f, "esp-dl experiment component is not enabled"),
            Self::Esp(err) => write!(f, "ESP error: {err}"),
            Self::InvalidResult(message) => write!(f, "{message}"),
        }
    }
}

impl Error for InferenceError {}

pub const MODEL_INPUT_LEN: usize = MODEL_WINDOW_SIZE * MODEL_FEATURE_COUNT;
pub type SensorFrame = [f32; MODEL_FEATURE_COUNT];

#[derive(Clone, Copy, Debug)]
pub struct RankedPrediction {
    pub index: usize,
    pub label: &'static str,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct InferenceResult {
    pub predicted_index: usize,
    pub predicted_label: &'static str,
    pub input_exponent: i32,
    pub output_exponent: i32,
    pub quantized_output: [i8; MODEL_OUTPUT_COUNT],
    pub dequantized_output: [f32; MODEL_OUTPUT_COUNT],
    pub message: String,
    pub output_preview: String,
}

#[derive(Clone, Debug)]
pub struct SelfTestReport {
    pub inference: InferenceResult,
    pub exact_quantized_match: bool,
    pub max_dequantized_abs_error: f32,
}

pub struct SlidingWindow {
    frames: [SensorFrame; MODEL_WINDOW_SIZE],
    len: usize,
    next: usize,
}

impl SlidingWindow {
    pub fn new() -> Self {
        Self {
            frames: [[0.0; MODEL_FEATURE_COUNT]; MODEL_WINDOW_SIZE],
            len: 0,
            next: 0,
        }
    }

    pub fn push_frame(&mut self, frame: SensorFrame) {
        self.frames[self.next] = frame;
        self.next = (self.next + 1) % MODEL_WINDOW_SIZE;
        if self.len < MODEL_WINDOW_SIZE {
            self.len += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_full(&self) -> bool {
        self.len == MODEL_WINDOW_SIZE
    }

    pub fn flatten_if_full(&self) -> Option<[f32; MODEL_INPUT_LEN]> {
        if !self.is_full() {
            return None;
        }

        let mut flattened = [0.0f32; MODEL_INPUT_LEN];
        self.copy_into_flattened(&mut flattened);

        Some(flattened)
    }

    pub fn copy_into_flattened(&self, flattened: &mut [f32; MODEL_INPUT_LEN]) -> bool {
        if !self.is_full() {
            return false;
        }

        let oldest = self.next;
        for window_index in 0..MODEL_WINDOW_SIZE {
            let frame = self.frames[(oldest + window_index) % MODEL_WINDOW_SIZE];
            let base = window_index * MODEL_FEATURE_COUNT;
            flattened[base..base + MODEL_FEATURE_COUNT].copy_from_slice(&frame);
        }

        true
    }
}

impl Default for SlidingWindow {
    fn default() -> Self {
        Self::new()
    }
}

pub fn normalize_flat_input(input: &[f32; MODEL_INPUT_LEN]) -> [f32; MODEL_INPUT_LEN] {
    let mut normalized = [0.0f32; MODEL_INPUT_LEN];
    normalize_flat_input_into(input, &mut normalized);
    normalized
}

pub fn normalize_flat_input_into(
    input: &[f32; MODEL_INPUT_LEN],
    normalized: &mut [f32; MODEL_INPUT_LEN],
) {
    for (index, value) in input.iter().copied().enumerate() {
        let feature_index = index % MODEL_FEATURE_COUNT;
        normalized[index] = (value - FEATURE_MEANS[feature_index]) / FEATURE_SCALES[feature_index];
    }
}

impl InferenceResult {
    pub fn top_predictions(&self, count: usize) -> Vec<RankedPrediction> {
        let mut ranked: Vec<_> = self
            .dequantized_output
            .iter()
            .copied()
            .enumerate()
            .map(|(index, score)| RankedPrediction {
                index,
                label: MODEL_LABELS[index],
                score,
            })
            .collect();
        ranked.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        ranked.truncate(count.min(ranked.len()));
        ranked
    }
}

#[cfg(esp_idf_comp_espdl_experiment_enabled)]
fn parse_c_string(buffer: &[u8]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr().cast()) }
        .to_string_lossy()
        .trim_end_matches('\0')
        .to_string()
}

#[cfg(esp_idf_comp_espdl_experiment_enabled)]
unsafe fn reconfigure_task_wdt(timeout_ms: u32, idle_core_mask: u32) -> i32 {
    let config = esp_idf_svc::sys::esp_task_wdt_config_t {
        timeout_ms,
        idle_core_mask,
        trigger_panic: true,
    };
    esp_idf_svc::sys::esp_task_wdt_reconfigure(&config)
}

#[cfg(esp_idf_comp_espdl_experiment_enabled)]
fn run_with_relaxed_task_wdt<T>(
    f: impl FnOnce() -> Result<T, InferenceError>,
) -> Result<T, InferenceError> {
    unsafe {
        let _ = reconfigure_task_wdt(30_000, 0);
    }

    let result = f();

    unsafe {
        let _ = reconfigure_task_wdt(5_000, (1 << 0) | (1 << 1));
    }

    result
}

#[cfg(esp_idf_comp_espdl_experiment_enabled)]
fn map_ffi_result(
    raw: &esp_idf_svc::sys::espdl_experiment::espdl_inference_result_t,
) -> Result<InferenceResult, InferenceError> {
    let logical_output_size = usize::try_from(raw.logical_output_size)
        .map_err(|_| InferenceError::InvalidResult("logical output size does not fit in usize"))?;
    if logical_output_size < MODEL_OUTPUT_COUNT {
        return Err(InferenceError::InvalidResult(
            "esp-dl returned fewer outputs than the trained model expects",
        ));
    }

    let predicted_index = usize::try_from(raw.predicted_index)
        .map_err(|_| InferenceError::InvalidResult("predicted index does not fit in usize"))?;
    if predicted_index >= MODEL_OUTPUT_COUNT {
        return Err(InferenceError::InvalidResult(
            "predicted index is out of range",
        ));
    }

    let mut quantized_output = [0i8; MODEL_OUTPUT_COUNT];
    quantized_output.copy_from_slice(&raw.quantized_output[..MODEL_OUTPUT_COUNT]);

    let mut dequantized_output = [0.0f32; MODEL_OUTPUT_COUNT];
    dequantized_output.copy_from_slice(&raw.dequantized_output[..MODEL_OUTPUT_COUNT]);

    Ok(InferenceResult {
        predicted_index,
        predicted_label: MODEL_LABELS[predicted_index],
        input_exponent: raw.input_exponent,
        output_exponent: raw.output_exponent,
        quantized_output,
        dequantized_output,
        message: parse_c_string(&raw.message),
        output_preview: parse_c_string(&raw.output_preview),
    })
}

#[cfg(esp_idf_comp_espdl_experiment_enabled)]
pub fn run_normalized_input(
    input: &[f32; MODEL_INPUT_LEN],
) -> Result<InferenceResult, InferenceError> {
    run_with_relaxed_task_wdt(|| {
        use core::mem::MaybeUninit;

        let mut raw =
            MaybeUninit::<esp_idf_svc::sys::espdl_experiment::espdl_inference_result_t>::zeroed();
        let err = unsafe {
            esp_idf_svc::sys::espdl_experiment::espdl_experiment_run_float_input(
                input.as_ptr(),
                input.len(),
                raw.as_mut_ptr(),
            )
        };
        let raw = unsafe { raw.assume_init() };
        if err != 0 {
            return Err(InferenceError::Esp(err));
        }
        map_ffi_result(&raw)
    })
}

#[cfg(not(esp_idf_comp_espdl_experiment_enabled))]
pub fn run_normalized_input(
    _input: &[f32; MODEL_INPUT_LEN],
) -> Result<InferenceResult, InferenceError> {
    Err(InferenceError::ComponentUnavailable)
}

pub fn run_raw_sensor_input(
    input: &[f32; MODEL_INPUT_LEN],
) -> Result<InferenceResult, InferenceError> {
    let normalized = normalize_flat_input(input);
    run_normalized_input(&normalized)
}

pub fn run_raw_sensor_input_with_scratch(
    input: &[f32; MODEL_INPUT_LEN],
    normalized: &mut [f32; MODEL_INPUT_LEN],
) -> Result<InferenceResult, InferenceError> {
    normalize_flat_input_into(input, normalized);
    run_normalized_input(normalized)
}

#[cfg(esp_idf_comp_espdl_experiment_enabled)]
pub fn run_quantized_self_test() -> Result<SelfTestReport, InferenceError> {
    run_with_relaxed_task_wdt(|| {
        let inference = run_normalized_input(&ESPDL_TEST_INPUT)?;
        let exact_quantized_match = inference.quantized_output == ESPDL_TEST_OUTPUT;
        let scale = 2f32.powi(ESPDL_OUTPUT_EXPONENT);
        let mut max_dequantized_abs_error = 0.0f32;
        for (actual, expected_quantized) in inference
            .dequantized_output
            .iter()
            .copied()
            .zip(ESPDL_TEST_OUTPUT.iter().copied())
        {
            let expected = f32::from(expected_quantized) * scale;
            max_dequantized_abs_error = max_dequantized_abs_error.max((actual - expected).abs());
        }

        Ok(SelfTestReport {
            inference,
            exact_quantized_match,
            max_dequantized_abs_error,
        })
    })
}

#[cfg(not(esp_idf_comp_espdl_experiment_enabled))]
pub fn run_quantized_self_test() -> Result<SelfTestReport, InferenceError> {
    Err(InferenceError::ComponentUnavailable)
}
