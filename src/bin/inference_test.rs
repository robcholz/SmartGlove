use smart_glove::inference::{
    run_quantized_self_test, ESPDL_INPUT_EXPONENT, ESPDL_INPUT_SHAPE, ESPDL_OUTPUT_EXPONENT,
    ESPDL_OUTPUT_SHAPE,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    log::info!(
        "gesture inference self-test starting: input_shape={:?} input_exponent={} output_shape={:?} output_exponent={}",
        ESPDL_INPUT_SHAPE,
        ESPDL_INPUT_EXPONENT,
        ESPDL_OUTPUT_SHAPE,
        ESPDL_OUTPUT_EXPONENT
    );

    let report = run_quantized_self_test()?;
    log::info!(
        "gesture inference self-test result: predicted_index={} predicted_label={} exact_quantized_match={} max_dequantized_abs_error={:.6} preview={} message={}",
        report.inference.predicted_index,
        report.inference.predicted_label,
        report.exact_quantized_match,
        report.max_dequantized_abs_error,
        report.inference.output_preview,
        report.inference.message
    );
    log::info!("quantized_output={:?}", report.inference.quantized_output);

    if !report.exact_quantized_match {
        return Err("embedded .espdl output did not match the exported test vector".into());
    }

    Ok(())
}
