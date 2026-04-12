#![cfg_attr(not(test), no_std)]

pub const fn firmware_name() -> &'static str {
    "SmartGlove"
}

#[cfg(test)]
mod tests {
    use super::firmware_name;

    #[test]
    fn firmware_name_is_stable() {
        assert_eq!(firmware_name(), "SmartGlove");
    }
}
