#[cfg(esp_idf_comp_espdl_experiment_enabled)]
pub fn run_espdl_probe() {
    run_espdl_probe_inner();
}

#[cfg(not(esp_idf_comp_espdl_experiment_enabled))]
pub fn run_espdl_probe() {
    log::warn!("ESP-DL experiment component is not enabled in this build");
}

#[cfg(esp_idf_comp_espdl_experiment_enabled)]
fn run_espdl_probe_inner() {
    use core::ffi::CStr;
    use core::mem::MaybeUninit;
    use esp_idf_svc::sys;

    unsafe fn reconfigure_task_wdt(timeout_ms: u32, idle_core_mask: u32) -> sys::esp_err_t {
        let config = sys::esp_task_wdt_config_t {
            timeout_ms,
            idle_core_mask,
            trigger_panic: true,
        };
        sys::esp_task_wdt_reconfigure(&config)
    }

    unsafe {
        let wdt_disabled = reconfigure_task_wdt(30_000, 0);
        if wdt_disabled != 0 {
            log::warn!(
                "failed to relax task watchdog before ESP-DL inference, esp_err_t={wdt_disabled}"
            );
        }

        let mut result = MaybeUninit::<sys::espdl_experiment::espdl_experiment_result_t>::zeroed();
        let err = sys::espdl_experiment::espdl_experiment_run(result.as_mut_ptr());
        let wdt_restored = reconfigure_task_wdt(5_000, (1 << 0) | (1 << 1));

        if wdt_restored != 0 {
            log::warn!(
                "failed to restore task watchdog after ESP-DL inference, esp_err_t={wdt_restored}"
            );
        }

        if err == 0 {
            let result = result.assume_init();
            let message = CStr::from_ptr(result.message.as_ptr())
                .to_str()
                .unwrap_or("ESP-DL probe returned a non-UTF8 message");

            log::info!(
                "ESP-DL experiment passed: enabled={}, model_size={}, context_size={}, memory_manager_size={}, input_count={}, output_count={}, output_checksum={}, output_preview={}, message={}",
                result.component_enabled,
                result.model_class_size,
                result.context_class_size,
                result.memory_manager_size,
                result.input_count,
                result.output_count,
                result.output_checksum,
                CStr::from_ptr(result.output_preview.as_ptr())
                    .to_str()
                    .unwrap_or("ESP-DL output preview returned a non-UTF8 message"),
                message
            );
        } else {
            let result = result.assume_init();
            let message = CStr::from_ptr(result.message.as_ptr())
                .to_str()
                .unwrap_or("ESP-DL probe returned a non-UTF8 message");
            log::error!(
                "ESP-DL experiment failed: esp_err_t={}, input_count={}, output_count={}, message={}",
                err,
                result.input_count,
                result.output_count,
                message
            );
        }
    }
}
