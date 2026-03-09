# Adaptive Matching Simulation (Group 39)

Discrete-event simulation of the **Asymmetric(a, δ) expert system** under two matching policies:
- **Random**
- **Backpressure(ε)**

Tasks arrive according to a Poisson process, are served by two experts with type-dependent success probabilities, and are assigned from a shared task pool without observing true type. Beliefs are updated on failure using the posterior update rule.

Based on the model in *"Adaptive matching for expert systems with uncertain task types"* (Shah, Gulikers, Massoulié, Vojnović, 2018).

## Requirements

- **Python 3.11+**
- Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib` (see `pyproject.toml` / `requirements.txt`)

## Hardware and OS assumptions

- **OS**: macOS, Linux, or Windows (with PowerShell/WSL for shell commands). CI runs on Linux (GitHub Actions).
- **Hardware**: No GPU or special hardware. Single-threaded, standard laptop/desktop sufficient for the provided experiments.
- **Docker** (optional): If using the provided Dockerfile, any host OS supported by Docker (Linux, macOS, Windows with WSL2) is assumed.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

For development (pytest, mypy, ruff, black):

```bash
pip install -e ".[dev]"
```

## Project structure

```
.
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── expert_sim/
│       ├── __init__.py      # run_simulation, run_replications
│       ├── config.py       # SimulationConfig
│       ├── models.py       # Task, Expert, Event, SystemState
│       ├── probabilities.py # Asymmetric(a,δ) matrix, belief update φ_s(z)
│       ├── policies.py     # Policy protocol, RandomPolicy, BackpressurePolicy
│       ├── simulator.py    # Discrete-event Simulator, run_simulation, run_replications
│       ├── stats.py        # Warm-up filtering, t-CI, aggregate_replications
│       ├── experiments.py  # Grid experiments, CSV output
│       └── cli.py          # single / reps / grid subcommands
├── tests/
│   ├── conftest.py
│   ├── test_beliefs.py
│   ├── test_experiments.py
│   ├── test_policy.py
│   ├── test_simulator.py
│   ├── test_reproducibility.py
│   └── test_stats.py
├── scripts/
│   ├── run_experiments.py  # Default Task 1 + Task 2 grids → results/
│   └── plot_results.py     # Plots from results/ → figures/ 
```

## Usage

Use the CLI via `python -m expert_sim.cli` or the installed command `expert-sim`. The **default policy is Random**; for Backpressure(ε) add `--algorithm backpressure --eps <ε>` (e.g. `--eps 0.025` for ε = 1/40).

**Subcommands:**

- **single** — one simulation run; prints metrics (mean sojourn time, completions, utilization, etc.).
- **reps** — multiple replications with 95% confidence intervals for mean sojourn time.
- **grid** — full (λ, δ) parameter grid; writes CSV tables to `--outdir`.
- **grid** can run with adaptive replication control for Task D via:
  `--adaptive-reps --min-reps ... --max-reps ... --rep-batch-size ... --min-significant-digits ...`

All subcommands accept `--algorithm {random|backpressure}`; Backpressure requires `--eps`.

| Goal | Command (outline) |
|------|-------------------|
| One run, Random | `expert-sim single --lam ... --a ... --delta ... --horizon ... --warmup ... --seed ...` |
| One run, Backpressure | Same + `--algorithm backpressure --eps 0.025` |
| Replications + CI | `expert-sim reps ... --n-reps 30 --base-seed 42` (+ optional `--algorithm backpressure --eps ...`) |
| Full grid, Random | `expert-sim grid --a 0.5 ... --outdir results/` |
| Full grid, Backpressure | Same + `--algorithm backpressure --eps 0.025 --outdir results/backpressure_eps_1_40` |
| Full grid, adaptive precision | Add `--adaptive-reps --min-reps 30 --max-reps 300 --rep-batch-size 10 --min-significant-digits 2` |
| Both grids (script) | `python scripts/run_experiments.py` |
| Run in Docker | `docker run --rm expert-sim <subcommand> <args>` (same args as above) |

### Command-line interface

**Random (default)** — single run:

```bash
python -m expert_sim.cli single --lam 0.8 --a 0.5 --delta 0.2 --horizon 10000 --warmup 1000 --seed 42
```

**Random** — multiple replications with 95% confidence intervals:

```bash
python -m expert_sim.cli reps --lam 0.8 --a 0.5 --delta 0.2 --horizon 10000 --warmup 1000 --n-reps 30 --base-seed 42
```

**Backpressure** — single run (`--eps` required):

```bash
python -m expert_sim.cli single --lam 0.8 --a 0.5 --delta 0.2 --horizon 10000 --warmup 1000 --seed 42 --algorithm backpressure --eps 0.025
```

**Random grid (default)** — full (λ, δ) grid and write CSVs:

```bash
python -m expert_sim.cli grid --a 0.5 --horizon 10000 --warmup 1000 --n-reps 30 --base-seed 42 --outdir results/
```

**Backpressure(ε) grid**:

```bash
python -m expert_sim.cli grid --a 0.5 --horizon 10000 --warmup 1000 --n-reps 30 --base-seed 42 --algorithm backpressure --eps 0.025 --outdir results/backpressure_eps_1_40
```

**Adaptive Task D grid** (example):

```bash
python -m expert_sim.cli grid --a 0.5 --horizon 10000 --warmup 1000 --n-reps 30 --base-seed 42 --algorithm backpressure --eps 0.025 --adaptive-reps --min-reps 30 --max-reps 300 --rep-batch-size 10 --min-significant-digits 2 --outdir results/backpressure_eps_1_40
```

**Both grids via script** — runs both Random and Backpressure grids and writes to `results/random/` and `results/backpressure_eps_1_40/`:

```bash
python scripts/run_experiments.py
```

### Python API

```python
from expert_sim import run_simulation, run_replications

# Single run (Random by default)
metrics = run_simulation(lam=0.8, a=0.5, delta=0.2, horizon=10_000, warmup=1_000, seed=42)
# metrics: mean_sojourn_time, n_completed_after_warmup, total_arrivals, tasks_in_system_at_horizon, expert_utilization

# Single run (Backpressure, epsilon = 1/40)
metrics_bp = run_simulation(
    lam=0.8,
    a=0.5,
    delta=0.2,
    horizon=10_000,
    warmup=1_000,
    seed=42,
    algorithm="backpressure",
    eps=1/40,
)

# Replications with CIs
result = run_replications(lam=0.8, a=0.5, delta=0.2, horizon=10_000, warmup=1_000, n_reps=30, base_seed=42)
# result["mean_sojourn_time"]: mean, std, half_width, low, high
```

## Experiments

- **Warm-up**: Tasks completed at or before `warmup` are excluded from mean-sojourn-time statistics; only completion time \> warmup counts.
- **Confidence intervals**: 95% t-based CIs from replication-level means (not pooled over tasks).
- **Default grid**: λ ∈ {0.75, 0.77, 0.79, 0.81, 0.83, 0.85}, δ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}, a = 0.5.
- **Task 2 setting**: Backpressure with ε = 1/40.
- **Task D integration**: Optional adaptive replication budget to hit a target significant-digit precision per cell, with diagnostics written to `numeric.csv` (`n_reps_used`, `precision_met`, `achieved_significant_digits`, `relative_half_width`).

### Report figures

To generate the figures (sojourn vs λ, heatmaps, bar comparison), run the grid experiments first, then the plotting script:

```bash
python scripts/run_experiments.py
python scripts/plot_results.py
```

## Docker

Build and run the CLI in a container:

```bash
docker build -t expert-sim .
docker run --rm expert-sim --help
docker run --rm expert-sim single --lam 0.8 --a 0.5 --delta 0.2 --horizon 10000 --warmup 1000 --seed 42
```

Same options as the CLI: use `--algorithm backpressure --eps <value>` for Backpressure. Example:

```bash
docker run --rm expert-sim single --lam 0.8 --a 0.5 --delta 0.2 --horizon 10000 --warmup 1000 --seed 42 --algorithm backpressure --eps 0.025
```

To write results to the host:

```bash
docker run --rm -v "$(pwd)/results:/app/results" expert-sim grid --a 0.5 --horizon 10000 --warmup 1000 --n-reps 30 --base-seed 42 --outdir /app/results
```

## Testing and quality

Run tests:

```bash
pytest
```

Lint, format, and type-check:

```bash
ruff check .
black --check .
mypy src
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full dev setup and checks.

## CI (GitHub Actions)

The main workflow (`.github/workflows/ci.yml`) runs on Python 3.11 and executes:

- `python -m ruff check .`
- `python -m black --check .`
- `python -m mypy src`
- `python -m pytest`

## License

MIT. See [LICENSE](LICENSE).

### Citation

If you use this code or build on this work, please cite:

BibTeX:

```bibtex
@inproceedings{ogul2026simulation,
  author    = {Ogul, Mert and Georgopoulos, Andrew and Molin, Micah and Buring, Lou-Corto},
  title     = {A Simulation Study of Random and Backpressure Matching in Asymmetric Expert Systems},
  booktitle = {TU/e 2MBS40 Stochastic Simulation and Modeling Course Assignment of Adaptive Matching for Expert Systems with Uncertain Task Types},
  year      = {2026}
}


