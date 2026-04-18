# Onboarding

Run this project inside the **devcontainer** and mount the current directory into it.

On the **host** machine, install:

```bash
cargo install espflash
```

## Flash & Monitor

Run this command in your **host**.

Debug:

```bash
espflash flash --monitor target/xtensa-esp32s3-none-elf/debug/smart-glove
```

Release:

```bash
espflash flash --monitor target/xtensa-esp32s3-none-elf/release/smart-glove
```

## Notes

If IDE failed to detect toolchain, you might need to configure the right toolchain (you can get the path by running
`where cargo`)
