# Repository Guidelines

## Project Structure & Module Organization
Core MAPF logic lives in `pycam/`, where planners (`planner_*.py`) combine heuristics from `lacam*.py`, allocators, and utilities such as `mapf_utils.py`. Root scripts (`app.py`, `app_test_*`, `test_gvpo_pipelines.py`) provide CLIs and regression entrypoints. Sample grids stay in `assets/` and the numerous `Datas*/images*/` bundles; treat them as read-only inputs and keep new datasets outside Git. Visualization notebooks sit in `notebooks/`, while generated traces land in `output.txt` and plots under `assets/`.

## Build, Test, and Development Commands
- `poetry config virtualenvs.in-project true && poetry install`: create the pinned Poetry environment defined by `pyproject.toml`.
- `poetry run python app.py -m assets/tunnel.map -i assets/tunnel.scen -N 4 --time_limit_ms 5000 --verbose 2`: run the reference scenario and write `output.txt`.
- `poetry run python app.py -m assets/tunnel.map -i assets/tunnel.scen -N 2 --no-flg_star`: faster suboptimal run useful when iterating on heuristics.
- `poetry run python test_gvpo_pipelines.py`: exercise the quick regression plus full GVPO benchmarks exposed in `pycam/allo_util_RL`.
- `poetry run pytest`: execute any unit or smoke suites you add under `tests/` or alongside modules.
- `poetry run jupyter lab`: launch notebook experiments inside the project virtual environment.

## Coding Style & Naming Conventions
Use 4-space indentation, `snake_case` for modules and functions, and CapWords for classes. Install the hooks once via `pre-commit install`, then rely on Black (line length 88) and isort (`--profile black`) to keep formatting consistent; run them manually with `pre-commit run --all-files` before pushing. Group imports as stdlib, third-party, then local. Prefer explicit type hints, short docstrings describing CLI flags, and descriptive filenames such as `lacam_star_single.py`. Keep logging lightweight through `loguru` and gate notebook-specific helpers behind `if __name__ == "__main__":`.

## Testing Guidelines
Author tests with `pytest`, naming files and functions `test_*.py` so `poetry run pytest` discovers them automatically. Reuse deterministic `assets/*.map` fixtures and validate that planners emit the expected `output.txt` structure or path lengths. The GVPO suite exposes `_quick_gvpo_regression_test()` for smoke coverage and `_full_gvpo_test_pipeline()` for longer benchmarks; run them through `test_gvpo_pipelines.py` and capture any failures in the PR description. Store bulky artifacts under temporary folders and clean them before committing to keep the repository lean.

## Commit & Pull Request Guidelines
Existing history favors short lowercase summaries (`init`, `test`), so continue with imperative lines under 50 characters, then add optional detail in the body. Each pull request should explain the motivation, list touched modules (for example `pycam/allocate.py`, `assets/tunnel.scen`), and paste the exact commands you ran (`poetry run python test_gvpo_pipelines.py`, `poetry run pytest`). Link issues or research notes, attach screenshots for visualization or notebook changes, and confirm that new data stays out of version control. Highlight any config toggles or environment variables reviewers must set to reproduce your results.
