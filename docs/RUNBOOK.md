# Runbook

`dysphonia-detection` is the project root. Run commands from this directory unless a command states otherwise.

```text
dysphonia-detection/
  Data/
  metadata/subjects/
  pretrained_models/
  docs/
  tools/
  scripts/
  voice_disorder_torch/
  pyproject.toml
  setup_env.sh
```

## Environment

Create or sync the uv environment:

```bash
./setup_env.sh
```

For an existing environment, sync dependencies with:

```bash
uv sync
```

Check that the Python environment can import the package and run small forward passes:

```bash
uv run --no-sync python tools/smoke_check.py
```

## Inputs

```text
Data/EENT_processed/pickle_files
Data/SVD_processed/pickle_files
metadata/subjects/EENT_subjects_share_decrypted.xlsx
metadata/subjects/SVD.xlsx
pretrained_models/SSAST-Base-Patch-400.pth
```


## Centralized CNN

Train the `/a/` + `/i/` pair:

```bash
uv run --no-sync python scripts/train.py \
  --vowel both \
  --dev-test-seed 8 \
  --train-val-seed 100 \
  --model-init-seed 2718
```

Evaluate a trained pair:

```bash
uv run --no-sync python scripts/eval_launcher.py --results-json saved_models/eval_results_fixed.json
```

## Split Learning CNN

Train the `/a/` + `/i/` pair:

```bash
uv run --no-sync python scripts/split_learning/train_split.py \
  --vowel both \
  --model-type cnn \
  --dev-test-seed 8 \
  --train-val-seed 100 \
  --model-init-seed 2718 \
  --n-partitions 5 \
  --partition-seed 42
```

Evaluate a trained pair:

```bash
uv run --no-sync python scripts/eval_launcher.py --results-json saved_models/eval_results_split_fixed.json
```

## Split Learning SSAST

Train the `/a/` + `/i/` pair:

```bash
uv run --no-sync python scripts/split_learning/train_split.py \
  --vowel both \
  --model-type ssast \
  --dev-test-seed 8 \
  --train-val-seed 100 \
  --model-init-seed 2718 \
  --n-partitions 5 \
  --partition-seed 42 \
  --n-client-blocks 2
```

Evaluate a trained pair:

```bash
uv run --no-sync python scripts/eval_launcher.py --results-json saved_models/eval_results_split_ssast_fixed.json
```

## Result Summary

Print any evaluation JSON produced by `--results-json` above. With `--patient-eval-strategy fixed`
and the default seeds, the three flows line up as follows:

```bash
uv run --no-sync python scripts/print_eval_results.py saved_models/eval_results_fixed.json -v
uv run --no-sync python scripts/print_eval_results.py saved_models/eval_results_split_fixed.json -v
uv run --no-sync python scripts/print_eval_results.py saved_models/eval_results_split_ssast_fixed.json -v
```
