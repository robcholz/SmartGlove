fn main() {
    println!("cargo::rustc-check-cfg=cfg(esp_idf_comp_espdl_experiment_enabled)");
    println!("cargo::rustc-check-cfg=cfg(esp_idf_comp_espressif__esp_dl_enabled)");
    embuild::espidf::sysenv::output();
}
