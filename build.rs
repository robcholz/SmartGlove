fn main() {
    println!("cargo::rustc-check-cfg=cfg(esp_idf_comp_espdl_experiment_enabled)");
    println!("cargo::rustc-check-cfg=cfg(esp_idf_comp_espressif__esp_dl_enabled)");
    println!("cargo::rustc-link-arg=-Wl,-u,_ZSt11_Hash_bytesPKvjj");
    println!(
        "cargo::rustc-link-arg=-Wl,-u,_ZNKSt8__detail20_Prime_rehash_policy14_M_need_rehashEjjj"
    );
    embuild::espidf::sysenv::output();
}
