# Ultra-Slow Dark Energy Modelling

Minimal Python toolkit to evaluate a low-frequency oscillatory dark-energy template (Model A) against a \(\Lambda\)CDM baseline.

## Features

- \(\Lambda\)CDM background expansion: \(E(z), H(z)\)
- Model A oscillatory equation of state
- Observable computation: \(H(z), D_L(z), q(z)\)
- Residuals: \(\Delta_H(z), \Delta_{D_L}(z)\)
- CLI demo and pytest suite

## Quickstart

1. Create and activate a virtual environment
2. Install in editable mode with dev dependencies
3. Run tests
4. Run demo

Example command sequence (PowerShell):

- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`
- `python -m pip install --upgrade pip`
- `pip install -e .[dev]`
- `pytest -q`
- `python -m ultra_slow_de.demo --z-max 2.0 --n-z 300 --w0 -0.98 --A 0.05 --omega 0.8 --phi 0.0`

## Docker runtime for forced CLASS modeling

When local Windows builds of `classy` fail, use the provided Linux container:

- Build image: `docker compose -f docker-compose.class.yml build`
- Run forced CLASS growth check: `docker compose -f docker-compose.class.yml run --rm class-runner`

Notes:

- The repository root is bind-mounted to `/workspace`.
- The host `output/` folder is bind-mounted to `/workspace/output`, so logs and JSON artifacts written in the container appear directly in your local `output/` directory.
- Default container command writes: `output/fsig8_class_docker_log.txt`.