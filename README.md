_# SmartGlove

See [onboarding.md](docs/onboarding.md)

See [CONTRIBUTION.md](CONTRIBUTION.md)

## Environment Setup

We highly recommend using devcontainer.

Or, make sure you have everything for `esp-idf-svc`, and you should have `uv`.

Install Python dependencies with `uv`:

```bash
uv sync
```

## Building

Training the model (also do a test e2e inference)

```shell
uv run --group training python training/data_process.py
uv run --group training python training/train.py
uv run --group training python training/benchmark.py
uv run --group training python training/inference.py --port /dev/cu.SLAB_USBtoUART --verbose-stream
```

Flash

```shell
cargo run --bin smart-glove --release
```

## Testing

Create a `.env.local`, see `.env.example` for the template.
