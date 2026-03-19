# Contributing

Thanks for your interest in improving this audit project.

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Typical Development Flows

- Rebuild notebook template after editing generator:

```bash
python scripts/build_audit_notebook.py
```

- Rebuild paper assets (figures/tables):

```bash
python scripts/make_paper_assets.py --root .
```

- Run local model evaluation for benchmark-impact CSV outputs:

```bash
python scripts/run_binary_choice_eval.py --help
```

## Pull Requests

- Keep changes focused and minimal.
- Update docs when behavior/output paths change.
- Avoid committing large one-off prediction dumps.
- Keep generated artifacts reproducible from scripts in this repo.

## Reproducibility Expectations

- Prefer fixed random seeds where available.
- Document any hardware-sensitive behavior (CPU/CUDA/MPS differences).
- If changing thresholds or feature definitions, include rationale and sensitivity notes.
