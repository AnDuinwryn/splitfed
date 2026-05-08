# Paper 2601 Split-MAE Experimental Interface

This document describes the isolated implementation files for the Split Learning
adaptation of the paper method. All files are intentionally kept at the
repository root and are not wired into the existing package.

## Files

| File | Purpose |
| --- | --- |
| `paper2601_standard_ast.py` | Standard AST/timm backbone builder based on the public AST implementation pattern, not the repository SSAST model. |
| `paper2601_splitmae_utils.py` | Patch grid utilities, patch-wise normalization, content-aware masking, MA-Error loss, `SmashedData`, state-dict averaging helpers. |
| `paper2601_splitmae_client.py` | Client-side preprocessing, patch normalization, masking, AST patch embedding, and first AST encoder blocks. |
| `paper2601_splitmae_server.py` | Server-side remaining AST encoder blocks, lightweight MAE decoder, and Attention-FFNN classifier. |
| `paper2601_splitmae_training.py` | Standalone SplitFed-style Stage 1 and Stage 2 training helpers. |
| `paper2601_splitmae_cli.py` | Root-level command-line entry point for inspect, Stage 1, and Stage 2 runs. |
| `paper2601_splitmae_smoke.py` | Minimal tiny-model smoke example for one Stage 1 and Stage 2 forward/backward pass. |

## Import Model

The files use flat root-level imports:

```python
from paper2601_standard_ast import StandardASTBackbone, StandardASTConfig
from paper2601_splitmae_client import Paper2601SplitMAEClient, SplitMAEClientConfig
from paper2601_splitmae_server import Paper2601SplitMAEServer, SplitMAEServerConfig
from paper2601_splitmae_training import Paper2601SplitServerPool
from paper2601_splitmae_utils import SmashedData
```

This works when running commands from the repository root because Python adds
the current working directory to its module search path. No packaging or
installation step is required for this isolated workflow.

The implementation uses `timm==0.4.5` to instantiate the standard AST/DeiT
backbone. It does not import the repository's SSAST model.

## Client Interface

Create the client:

```python
from paper2601_splitmae_client import Paper2601SplitMAEClient, SplitMAEClientConfig

client = Paper2601SplitMAEClient(
    SplitMAEClientConfig(
        input_fdim=128,
        input_tdim=259,
        model_size="base384",
        n_client_blocks=2,
        mask_ratio=0.75,
        mask_strategy="content",
    )
)
```

Supported input shapes:

| Shape | Meaning |
| --- | --- |
| `(B, T, F)` | Current `SsastMelDataset` format. |
| `(B, 1, F, T)` | Channel-first spectrogram. |
| `(B, F, T, 1)` | Channel-last spectrogram. |

Stage 1 client call:

```python
smashed = client(x, mode="pretrain")
```

Stage 2 client call:

```python
smashed = client(x, mode="finetune", static_features=static_features)
```

`static_features` is optional. If supplied, it should normally be shaped
`(B, 131)` to match the paper's static feature branch.

## Server Interface

Create the server:

```python
from paper2601_splitmae_server import Paper2601SplitMAEServer, SplitMAEServerConfig

server = Paper2601SplitMAEServer(
    SplitMAEServerConfig(
        input_fdim=128,
        input_tdim=259,
        model_size="base384",
        n_client_blocks=2,
        num_labels=1,
        static_feature_dim=131,
        pooling="cls",
    )
)
```

Stage 1 server call:

```python
out = server.forward_pretrain(smashed)
loss = out["loss"]
pred_patches = out["pred_patches"]
encoded_tokens = out["encoded_tokens"]
```

Stage 2 server call:

```python
out = server.forward_finetune(smashed, return_attention=True)
logits = out["logits"]
attention = out["attention"]
audio_features = out["audio_features"]
```

The classifier returns logits. Use `BCEWithLogitsLoss` for binary or
multi-label targets.

## SmashedData Contract

`SmashedData` is the communication payload between client and server.

| Field | Stage | Description |
| --- | --- | --- |
| `tokens` | Both | Gradient-carrying smashed tensor. |
| `mode` | Both | `"pretrain"` or `"finetune"`. |
| `cls_token_count` | Both | Number of AST special tokens. |
| `patch_grid` | Both | Frequency/time patch layout. |
| `ids_keep` | Stage 1 | Visible patch indices after masking. |
| `ids_restore` | Stage 1 | Restore order for MAE decoder. |
| `mask` | Stage 1 | `(B, N)` mask, where `1` means masked. |
| `target_patches` | Stage 1 | Patch-wise normalized reconstruction targets. |
| `static_features` | Stage 2 | Optional tabular/static features. |

Only `tokens` should carry gradients back to the client. Metadata tensors are
detached by the client and consumed by the server.

## Training Helpers

The training helper mirrors the existing SplitFed style: one server replica per
client partition, then optional server averaging after a round.

```python
from paper2601_splitmae_training import (
    Paper2601SplitServerPool,
    run_stage1_splitfed_round,
    run_stage2_splitfed_round,
)

server_pool = Paper2601SplitServerPool(
    server_template=server,
    n_partitions=5,
    server_lr=5e-5,
    device=device,
)

stage1_stats = run_stage1_splitfed_round(
    client_base=client,
    server_pool=server_pool,
    train_loaders=train_loaders,
    n_local_epochs=5,
    client_lr=5e-5,
    device=device,
)

stage2_stats = run_stage2_splitfed_round(
    client_base=client,
    server_pool=server_pool,
    train_loaders=train_loaders,
    n_local_epochs=5,
    client_lr=5e-5,
    device=device,
)
```

Current batch formats:

| Batch | Use |
| --- | --- |
| `(x, y)` | Compatible with current SSAST loaders. |
| `(x, y, static_features)` | Enables the static feature branch. |

## Running a Smoke Check

From the repository root:

```bash
uv run --no-sync python paper2601_splitmae_cli.py smoke
```

If `uv` is unavailable but the current Python environment has the project
dependencies installed:

```bash
python paper2601_splitmae_cli.py smoke
```

The smoke script uses `model_size="tiny"` and shorter `input_tdim` to keep the
test cheap. It checks:

- Stage 1 client/server forward and backward.
- Stage 2 client/server forward and backward.
- Static feature concatenation with `static_feature_dim=131`.

## Recommended Run Sequence

Run all commands from the repository root.

1. Syntax check:

```bash
python -m compileall ^
  paper2601_standard_ast.py ^
  paper2601_splitmae_utils.py ^
  paper2601_splitmae_client.py ^
  paper2601_splitmae_server.py ^
  paper2601_splitmae_training.py ^
  paper2601_splitmae_cli.py ^
  paper2601_splitmae_smoke.py
```

2. Cheap synthetic smoke run:

```bash
uv run --no-sync python paper2601_splitmae_cli.py smoke
```

3. Inspect split tensor shapes before touching real data:

```bash
uv run --no-sync python paper2601_splitmae_cli.py inspect ^
  --model-size tiny ^
  --input-fdim 128 ^
  --input-tdim 64 ^
  --n-client-blocks 1 ^
  --static-feature-dim 131
```

4. Run a very small Stage 1 MAE pass on the real loaders:

```bash
uv run --no-sync python paper2601_splitmae_cli.py train-stage1 ^
  --vowel a ^
  --model-size tiny ^
  --input-fdim 128 ^
  --input-tdim 64 ^
  --n-client-blocks 1 ^
  --n-partitions 2 ^
  --batch-size 8 ^
  --n-global-rounds 1 ^
  --n-local-epochs 1 ^
  --run-name debug_stage1_a
```

5. Run a matching Stage 2 classifier pass, loading the Stage 1 weights:

```bash
uv run --no-sync python paper2601_splitmae_cli.py train-stage2 ^
  --vowel a ^
  --model-size tiny ^
  --input-fdim 128 ^
  --input-tdim 64 ^
  --n-client-blocks 1 ^
  --n-partitions 2 ^
  --batch-size 8 ^
  --n-global-rounds 1 ^
  --n-local-epochs 1 ^
  --num-labels 1 ^
  --load-client paper2601_splitmae_runs/debug_stage1_a_client.pt ^
  --load-server paper2601_splitmae_runs/debug_stage1_a_server.pt ^
  --run-name debug_stage2_a
```

6. Move to a runbook-compatible AST setup after the debug commands pass:

```bash
uv run --no-sync python paper2601_splitmae_cli.py train-stage1 ^
  --vowel a ^
  --model-size base384 ^
  --input-fdim 128 ^
  --input-tdim 259 ^
  --n-client-blocks 2 ^
  --n-partitions 5 ^
  --batch-size 16 ^
  --n-global-rounds 10 ^
  --n-local-epochs 5 ^
  --run-name ast_stage1_a
```

7. Fine-tune from that Stage 1 checkpoint:

```bash
uv run --no-sync python paper2601_splitmae_cli.py train-stage2 ^
  --vowel a ^
  --model-size base384 ^
  --input-fdim 128 ^
  --input-tdim 259 ^
  --n-client-blocks 2 ^
  --n-partitions 5 ^
  --batch-size 16 ^
  --n-global-rounds 20 ^
  --n-local-epochs 5 ^
  --num-labels 1 ^
  --load-client paper2601_splitmae_runs/ast_stage1_a_client.pt ^
  --load-server paper2601_splitmae_runs/ast_stage1_a_server.pt ^
  --run-name ast_stage2_a
```

For the current binary dysphonia labels, keep `--num-labels 1`, which uses
`BCEWithLogitsLoss` internally. If a true multi-label target matrix is
introduced later, set `--num-labels` to the label count and make the loader emit
labels shaped `(B, num_labels)`.

The commands above use the default data paths from the project runbook. Override
them with `--pickle-dir`, or with both `--pickle-dir-eent` and
`--pickle-dir-svd`, if needed.

## pyproject.toml

No `pyproject.toml` change is required for the current isolated workflow.

Reason: the implementation is intentionally not part of the installable package.
The current project configuration only includes:

```toml
include = ["voice_disorder_torch*"]
```

That is correct for the existing package. The experimental files can be run from
the repository root without package installation changes.

Modify `pyproject.toml` only if this experiment is promoted into the main
package, for example by moving the code under `voice_disorder_torch/` or by
creating a formal package namespace.

## Known Integration Notes

- The current implementation instantiates a standard AST-style DeiT distilled
  backbone with `timm`, adapted to single-channel spectrograms.
- It intentionally does not use the repository's SSAST model.
- `imagenet_pretrain=False` by default to avoid implicit network downloads.
- If a local official AST/AudioSet checkpoint is available, pass
  `audioset_checkpoint_path=...`.
- Stage 1 uses patch-wise normalized reconstruction targets and an L1-style
  masked absolute error.
- Content-aware masking follows the paper's high-variance candidate-pool rule:
  select the top `max(70% * N_mask, 50% * N_total)` patches by variance, sample
  70% of masked patches from that group, and sample the rest from remaining
  patches.
- The lightweight MAE decoder defaults to `decoder_embed_dim=256`,
  `decoder_depth=4`, and `decoder_num_heads=4`.
- The server classifier supports the paper-style static feature branch through
  `static_feature_dim`, but current repository loaders do not yet emit static
  features.
- The files are intentionally root-level and flat. This keeps import behavior
  simple for review and preserves isolation from the main pipeline.
