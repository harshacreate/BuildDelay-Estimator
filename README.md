# BuildDelay-Estimator — Construction Schedule Risk Prototype

**Sector:** Construction  
**Task:** Binary classification: synthetic schedule delay

Estimates a synthetic delay label from project size, change orders, weather days, crew size, material delay, budget variance, and contractor experience. Contractor groups are held out.

## Data and provenance

The offline demo uses a deterministic synthetic generator in `builddelay/data.py`. It does not use or download real records, and no external data license applies. Synthetic output is only a benchmark for this simulator, not evidence of real-world performance.

## Method

- **Baseline:** majority-class prior classifier.
- **Stronger model:** histogram gradient-boosting classifier.
- **Split:** grouped holdout by machine, pipe, camera, or contractor (IDs are excluded from the model inputs).
- **Metrics:** accuracy, balanced accuracy, macro F1, confusion matrix, ROC-AUC, average precision, precision, and recall.
- **Error analysis:** confusion matrix and total misclassification count.

## Limitations

Construction timelines depend on contract terms, site conditions, labor, permits, supply chains, and local practices. Generated benchmarks do not justify project or contractor decisions.

## Run

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m builddelay.train --seed 42 --samples 2400 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` are shortcuts. Python 3.9 or newer is required.

## Layout and license

- `builddelay/data.py`: synthetic generator and validation.
- `builddelay/split.py`: entity-group split.
- `builddelay/model.py`: baseline, stronger model, metrics, and subgroup diagnostics.
- `builddelay/train.py`: CLI.
- `tests/`: data, split, model, metric, and CLI checks.
- `.github/workflows/ci.yml`: CI.

No code license has been selected for this new project; default copyright rules apply. Choose a license before public distribution.
