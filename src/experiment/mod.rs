#[cfg(esp_idf_comp_espdl_experiment_enabled)]
pub fn run_espdl_probe() {
    use core::ffi::CStr;
    use core::mem::MaybeUninit;
    use esp_idf_svc::sys;

    unsafe {
        let mut result = MaybeUninit::<sys::espdl_experiment::espdl_experiment_result_t>::zeroed();
        let err = sys::espdl_experiment::espdl_experiment_run(result.as_mut_ptr());

        if err == 0 {
            let result = result.assume_init();
            let message = CStr::from_ptr(result.message.as_ptr())
                .to_str()
                .unwrap_or("ESP-DL probe returned a non-UTF8 message");

            log::info!(
                "ESP-DL experiment passed: enabled={}, model_size={}, context_size={}, memory_manager_size={}, input_count={}, output_count={}, output_checksum={}, message={}",
                result.component_enabled,
                result.model_class_size,
                result.context_class_size,
                result.memory_manager_size,
                result.input_count,
                result.output_count,
                result.output_checksum,
                message
            );
        } else {
            log::error!("ESP-DL experiment failed with esp_err_t={err}");
        }
    }
}

#[cfg(not(esp_idf_comp_espdl_experiment_enabled))]
pub fn run_espdl_probe() {
    log::warn!("ESP-DL experiment component is not enabled in this build");
}
