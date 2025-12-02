# Repository Guidelines

## Project Structure & Module Organization
The entry script `app.py` parses CLI flags, loads grid maps from `assets/`, and calls planners in `pycam/`. Core solvers live in files such as `lacam.py`, `allocate.py`, and `mapf_utils.py`, while experimental baselines remain in sibling directories (`PIBT/`, `LNS2/`, `MA_CBS/`). Jupyter prototypes sit in `notebooks/`, and shared benchmarks plus visualization sprites stay inside the various `Datas*`, `images*`, and `assets*` folders. Keep generated artifacts like `output.txt` or plots in the root or `assets/` only when they document reproducible runs.

## Build, Test, and Development Commands
`poetry config virtualenvs.in-project true && poetry install` provisions the pinned Python 3.11 toolchain defined in `pyproject.toml`. Run `poetry run python app.py -m assets/tunnel.map -i assets/tunnel.scen -N 4 --time_limit_ms 5000 --verbose 2` to validate a baseline MAPF solve. Use `poetry run pytest pycam PIBT/tests` for automated regression checks and `poetry run pre-commit run --all-files` before opening a PR. Formatting is kept consistent via `poetry run black pycam app.py` and `poetry run isort pycam app.py`.

## Coding Style & Naming Conventions
Python modules use 4-space indentation, descriptive `snake_case` for functions, and `PascalCase` only for classes or typed configs. File names should reflect the algorithm implemented (`lacam_star_single.py`, `dist_table_4.py`). Favor side-effect-free helpers, funnel logging through `loguru`, and keep CLI parsing isolated to `app.py`.

## Testing Guidelines
Pytest is the default harness. Place fast unit tests next to their module (e.g., `pycam/test01.py`) and algorithmic parity tests under `PIBT/tests/` with names like `test_conflict_resolution`. Scenario-based regressions should reference concrete assets (`assets/tunnel.scen`) and assert on the serialized plan written to `output.txt`. Include coverage for new heuristics or reward functions by seeding deterministic RNGs.

## Commit & Pull Request Guidelines
Use short, imperative subjects patterned after upstream work, for example `lacam: tighten node ordering`. Each commit message should mention the impacted module and why the change matters; elaborate context belongs in the body. Pull requests must list reproduction commands, summarize solver impact (speed, optimality, memory), cite any new assets or notebooks, and attach screenshots or GIFs whenever visualization code changes.

## Security & Configuration Tips
Large archives under `images*/` or `Datas*/` should not be modified directly¡ªlink external hosting or Git LFS if replacements are unavoidable. Do not commit secrets or proprietary scenarios; instead, describe their provenance in the PR. When experimenting with new planners, prefer `.env` files or CLI arguments rather than editing source defaults so configs remain reviewable.
