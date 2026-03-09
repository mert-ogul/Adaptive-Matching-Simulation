## Contributing

- **Environment**: create a virtual environment and install dependencies with:
  - `pip install -r requirements.txt`
  - `pip install -e .[dev]`
- **Quality checks**:
  - `ruff check .`
  - `black .`
  - `mypy src`
  - `pytest`
- **Experiments**:
  - Run the default grid experiment with `python -m expert_sim.cli grid` or `python scripts/run_experiments.py`.

Please keep changes small and focused, with tests for new behavior.

