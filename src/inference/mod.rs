#[allow(dead_code)]
mod generated;

use core::ffi::CStr;
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

pub fn normalize_flat_input(input: &[f32; MODEL_INPUT_LEN]) -> [f32; MODEL_INPUT_LEN] {
    let mut normalized = [0.0f32; MODEL_INPUT_LEN];
    for (index, value) in input.iter().copied().enumerate() {
        let feature_index = index % MODEL_FEATURE_COUNT;
        normalized[index] = (value - FEATURE_MEANS[feature_index]) / FEATURE_SCALES[feature_index];
    }
    normalized
}

#[cfg(esp_idf_comp_espdl_experiment_enabled)]
fn parse_c_string(buffer: &[u8]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr().cast()) }
        .to_string_lossy()
        .trim_end_matches('\0')
        .to_string()
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

#[cfg(esp_idf_comp_espdl_experiment_enabled)]
pub fn run_quantized_self_test() -> Result<SelfTestReport, InferenceError> {
    use core::mem::MaybeUninit;

    let mut raw =
        MaybeUninit::<esp_idf_svc::sys::espdl_experiment::espdl_inference_result_t>::zeroed();
    let err = unsafe {
        esp_idf_svc::sys::espdl_experiment::espdl_experiment_run_quantized_input(
            ESPDL_TEST_INPUT.as_ptr(),
            ESPDL_TEST_INPUT.len(),
            ESPDL_INPUT_EXPONENT,
            raw.as_mut_ptr(),
        )
    };
    let raw = unsafe { raw.assume_init() };
    if err != 0 {
        return Err(InferenceError::Esp(err));
    }

    let inference = map_ffi_result(&raw)?;
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
}

#[cfg(not(esp_idf_comp_espdl_experiment_enabled))]
pub fn run_quantized_self_test() -> Result<SelfTestReport, InferenceError> {
    Err(InferenceError::ComponentUnavailable)
}
