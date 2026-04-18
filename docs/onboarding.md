# Onboarding

Run this project inside the **devcontainer** and mount the current directory into it.

On the **host** machine, install:

```bash
cargo install espflash
```

## Flash & Monitor

Run this command in your **host**.

Dev:

```bash
uv run --script uv_tasks.py
```

Flash:

```bash
cargo run
```

## Notes

If IDE failed to detect toolchain, you might need to configure the right toolchain (you can get the path by running
`where cargo`)
