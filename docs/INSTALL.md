# Install

EVA runs on Windows, Linux, and macOS. The default install assumes:

* Python 3.10+ (3.12 recommended).
* ≥8 GiB of system RAM (the default brain budget).
* No GPU required — CUDA/MPS are used automatically when present.

## Quick start (Linux / macOS)

```bash
bash install.sh
source .venv/bin/activate
python run.py ui   # open http://127.0.0.1:8765
```

Flags:

* `bash install.sh --gpu` — installs the CUDA wheel of torch.
* `bash install.sh --no-ui` — skips FastAPI / uvicorn.

## Quick start (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
.\.venv\Scripts\Activate.ps1
python run.py ui
```

Flags: `-Gpu`, `-NoUi`.

## Manual install

```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -e ".[ui]"
```

## Start scripts

* `bash start.sh [cmd]` / `./start.ps1 [cmd]` — activate the venv and run
  `python run.py <cmd>`. Defaults to `ui`. Available commands:
  * `ui` — interactive web interface (voice + chat + live inspection).
  * `train` — headless curiosity-driven training.
  * `interact` — CLI chat with a trained checkpoint.
  * `evolve` — self-evolution demo (watch the brain grow).
  * `tools` — list built-in tools.

## GPU / accelerator notes

The brain autodetects CUDA → MPS → CPU. Override with the `--device`
flag on `python run.py ui`, or set `hardware.device` in
`configs/default.yaml`.

## Pre-commit hooks

If `.pre-commit-config.yaml` is added to the repo, both `install.sh` and
`install.ps1` will run `pre-commit install` automatically.
