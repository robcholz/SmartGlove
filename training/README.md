```shell
uv run --group training python training/data_process.py
uv run --group training python training/train.py
uv run --group training python training/benchmark.py
uv run --group training python training/inference.py --port /dev/cu.SLAB_USBtoUART --verbose-stream
```