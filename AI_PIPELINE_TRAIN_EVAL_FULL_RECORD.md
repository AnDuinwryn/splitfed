# Training And Evaluation Protocol Record

Temporary analysis note. Focus: how each experiment was trained/evaluated,
with hyperparameters and model choices. No code interface details.

## 0. One-Page Overview

### Final Comparison Target

| Item | Setting |
| --- | --- |
| Unit | patient-level |
| Primary block | combined `/a+i/` |
| In-domain test | EENT held-out test |
| External test | SVD German |
| Default patient rule | fixed threshold |
| Patient threshold | `0.5` |
| Primary metrics | AUC, Accuracy, F1, Sensitivity, Specificity, Confusion Matrix |

### Shared Data And Seeds

| Category | Value |
| --- | --- |
| Train/val/test source | EENT Chinese |
| External test | SVD German |
| EENT test split | 20% patient-level held-out test |
| EENT validation split | 20% of develop set |
| Label mapping | `0 -> 0`, nonzero -> `1` |
| Mel preprocessing | first channel + per-sample normalization |
| `/a/` and `/i/` | trained separately, evaluated jointly |
| `dev_test_seed` | `8` |
| `train_val_seed` | `100` |
| `model_init_seed` | `2718` |
| `partition_seed` | `42` |

### Hyperparameter Summary

| Model / Stage | Optimizer | LR | Weight Decay | LR Decay | Loss | Selection |
| --- | --- | ---: | ---: | --- | --- | --- |
| Centralized CNN | Adam | `5e-5` | none | none | BCEWithLogits | best val loss |
| Split CNN | Adam | `5e-5` client/server | none | none | BCEWithLogits | best val loss |
| Split SSAST | Adam | `5e-5` client/server | none | none | CrossEntropy | best val loss |
| Paper2601 Stage 1 | AdamW | `1.5e-4` client/server | `0.05` | none | MA-Error | last round |
| Paper2601 Stage 2 | AdamW | `1.5e-4` client/server | `0.05` | none | Binary focal | best val Macro F1 |

### Epoch / Round Summary

| Model / Stage | Batch | Max Epochs / Rounds | Local Epochs | Patience |
| --- | ---: | ---: | ---: | ---: |
| Centralized CNN | `256` | `250` epochs | n/a | `10` |
| Split CNN | `256` | `250` global epochs | `5` | `10` |
| Split SSAST | `256` | `250` global epochs | `5` | `10` |
| Paper2601 Stage 1 | `256` | `120` global rounds | `1` | none |
| Paper2601 Stage 2 | `256` | `250` global rounds | `1` | `10` |

### Pretraining / Backbone Summary

| Model | Backbone | Pretrained Source |
| --- | --- | --- |
| Centralized CNN | custom 2D CNN | none |
| Split CNN | same custom 2D CNN, split after second pool | none |
| Split SSAST | SSAST Base | `SSAST-Base-Patch-400.pth` |
| Paper2601 Split-MAE | standard AST / DeiT `base384` | none by default; optional AST/AudioSet checkpoint only if supplied |

## 1. Centralized CNN

### Model Card

| Field | Value |
| --- | --- |
| Task | binary dysphonia classification |
| Models trained | one `/a/`, one `/i/` |
| Input | mel image `(C,H,W)`, usually one channel |
| Output | one binary logit |
| Pretraining | none |
| Initialization | deterministic He-style uniform, seed `2718` |

### Architecture

| Block | Layers |
| --- | --- |
| Conv block 1 | Conv2D `C -> 32`, `3x3`; ReLU; MaxPool `(2,3)` |
| Conv block 2 | Conv2D `32 -> 32`, `3x3`; ReLU; MaxPool `(2,3)`; Dropout `0.25` |
| Conv block 3 | Conv2D `32 -> 64`, `3x3`; ReLU; MaxPool `(2,3)` |
| Conv block 4 | Conv2D `64 -> 64`, `3x3`; ReLU; MaxPool `(2,3)`; Dropout `0.5` |
| Head | Flatten; Dense `128`; ReLU; Dropout `0.5`; Dense `1` |

### Training

| Field | Value |
| --- | --- |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | Adam |
| Learning rate | `5e-5` |
| LR decay / scheduler | none |
| Batch size | `256` |
| Max epochs | `250` |
| Early stopping | validation loss, patience `10` |
| Saved checkpoint | best validation-loss weights |
| Training display | epoch, train loss/acc, val loss/acc, patience |

### Evaluation

| Field | Value |
| --- | --- |
| Segment probability | sigmoid(logit) |
| Patient probability | mean segment positive probability |
| Patient decision | positive if probability >= `0.5` |
| Final report | `/a/`, `/i/`, combined `/a+i/` on EENT and SVD |

## 2. Split CNN

### Model Card

| Field | Value |
| --- | --- |
| Task | binary dysphonia classification |
| Models trained | one `/a/`, one `/i/` |
| Input | same mel image as centralized CNN |
| Output | one binary logit |
| Pretraining | none |
| Initialization | same deterministic CNN init, then split |

### Split Boundary

| Side | Layers |
| --- | --- |
| Client | Conv block 1 + Conv block 2 |
| Smashed data | tensor after second pool/dropout |
| Server | Conv block 3 + Conv block 4 + dense head |

### SplitFed Setup

| Field | Value |
| --- | --- |
| Client partitions | `5` |
| Partitioning | by patient ID |
| Server replicas | one per client partition |
| Client aggregation | uniform state-dict average |
| Server aggregation | uniform state-dict average |
| Partition seed | `42` |

### Training

| Field | Value |
| --- | --- |
| Loss | `BCEWithLogitsLoss` |
| Client optimizer | Adam |
| Server optimizer | Adam |
| Client LR | `5e-5` |
| Server LR | `5e-5` |
| LR decay / scheduler | none |
| Batch size | `256` |
| Global epochs | `250` |
| Local epochs | `5` |
| Early stopping | validation loss, patience `10` |
| Saved checkpoint | best validation-loss client/server weights |
| Training display | global epoch, val acc/loss, patience, per-partition train acc/loss |

### Evaluation

| Field | Value |
| --- | --- |
| Segment probability | sigmoid(server logit) |
| Patient probability | mean segment positive probability |
| Patient decision | positive if probability >= `0.5` |
| Final report | `/a/`, `/i/`, combined `/a+i/` on EENT and SVD |

## 3. Split SSAST

### Model Card

| Field | Value |
| --- | --- |
| Task | binary dysphonia classification as 2-class classification |
| Models trained | one `/a/`, one `/i/` |
| Input | mel sequence `(T,F)` |
| Default `F` | `128` |
| Default `T` | `259` |
| Output | 2-class logits |
| Pretrained model | `SSAST-Base-Patch-400.pth` |
| Model size | `base` |

### Patch / Transformer Settings

| Field | Value |
| --- | --- |
| Patch frequency shape | `16` |
| Patch time shape | `16` |
| Patch frequency stride | `10` |
| Patch time stride | `10` |
| Client transformer blocks | first `2` |
| Server transformer blocks | remaining blocks |
| Pooling | mean over non-class tokens |
| Classifier | LayerNorm + Linear to 2 classes |

### SplitFed Setup

| Field | Value |
| --- | --- |
| Client partitions | `5` |
| Partitioning | by patient ID |
| Server replicas | one per client partition |
| Client aggregation | uniform state-dict average |
| Server aggregation | uniform state-dict average |
| Partition seed | `42` |

### Training

| Field | Value |
| --- | --- |
| Loss | `CrossEntropyLoss` |
| Client optimizer | Adam |
| Server optimizer | Adam |
| Client LR | `5e-5` |
| Server LR | `5e-5` |
| LR decay / scheduler | none |
| Batch size | `256` |
| Global epochs | `250` |
| Local epochs | `5` |
| Early stopping | validation loss, patience `10` |
| Saved checkpoint | best validation-loss client/server weights |
| Training display | global epoch, val acc/loss, patience, per-partition train acc/loss |

### Evaluation

| Field | Value |
| --- | --- |
| Segment probability | softmax positive-class probability |
| Patient probability | mean segment positive probability |
| Patient decision | positive if probability >= `0.5` |
| Final report | `/a/`, `/i/`, combined `/a+i/` on EENT and SVD |

## 4. Paper2601 Split-MAE Standard-AST

### Base Model Card

| Field | Value |
| --- | --- |
| Backbone | standard AST / timm DeiT distilled ViT |
| Default model | `base384` |
| Concrete model | `vit_deit_base_distilled_patch16_384` |
| Required timm | `0.4.5` |
| Input | single-channel spectrogram |
| Frequency bins | `128` |
| Time frames | `259`, padded to `272` |
| Patch size | non-overlapping `16x16` |
| Patch grid | `8 x 17 = 136` patches |
| Patch dimension | `256` |
| ImageNet pretraining | off by default |
| AudioSet/AST checkpoint | optional only if explicitly supplied |
| Static features | supported, but current runs use `0` static features |

Supported model sizes:

| Alias | Concrete model |
| --- | --- |
| `tiny` | `vit_deit_tiny_distilled_patch16_224` |
| `small` | `vit_deit_small_distilled_patch16_224` |
| `base224` | `vit_deit_base_distilled_patch16_224` |
| `base384` | `vit_deit_base_distilled_patch16_384` |

### Stage 1 Card: Domain-Adaptive MAE

| Field | Value |
| --- | --- |
| Purpose | self-supervised domain adaptation |
| Labels used | no |
| Models trained | one `/a/`, one `/i/` |
| Client blocks | first `2` AST blocks |
| Server blocks | remaining AST blocks |
| Mask ratio | `0.75` |
| Mask strategy | content-aware masking |
| Reconstruction target | patch-wise normalized patches |
| Loss | MA-Error masked absolute reconstruction loss |
| Display metric | `ma_error` |

Content-aware masking:

| Step | Behavior |
| --- | --- |
| Rank | patches ranked by variance |
| High-information quota | 70% of masked patches |
| High-information pool | high-variance candidate pool |
| Remaining quota | sampled from lower-variance remainder |

MAE decoder:

| Field | Value |
| --- | --- |
| Decoder embedding dim | `256` |
| Decoder depth | `4` |
| Decoder heads | `8` |
| Decoder MLP ratio | `4.0` |
| Dropout | `0.1` |

Stage 1 training:

| Field | Value |
| --- | --- |
| Optimizer | AdamW |
| Client LR | `1.5e-4` |
| Server LR | `1.5e-4` |
| AdamW betas | `(0.9, 0.95)` |
| Weight decay | `0.05` |
| LR decay / scheduler | none |
| Batch size | `256` |
| Global rounds | `120` |
| Local epochs | `1` |
| Client partitions | `5` |
| Server replicas | one per partition |
| Aggregation | uniform client/server averaging |
| Early stopping | none |
| Saved checkpoint | weights after round `120` |

### Stage 2 Card: Attention-FFNN Classifier

| Field | Value |
| --- | --- |
| Purpose | supervised binary classification |
| Initialization | loads same-vowel Stage 1 client/server weights |
| Models trained | one `/a/`, one `/i/` |
| Client blocks | first `2` AST blocks |
| Server blocks | remaining AST blocks |
| Pooling | class-token pooling; distilled tokens averaged |
| Output | one binary logit |

Classifier:

| Component | Value |
| --- | --- |
| Audio feature dim | `768` for base model |
| Static feature dim | `0` in current runs |
| Attention | feature-wise sigmoid attention |
| FFNN hidden 1 | `512` |
| FFNN hidden 2 | `128` |
| Dropout | `0.1` |
| Output dim | `1` |

Stage 2 training:

| Field | Value |
| --- | --- |
| Loss | binary focal loss with logits |
| Focal gamma | `2.0` |
| Focal alpha | `None` |
| Primary validation metric | Macro F1 |
| Binary Macro F1 | average of negative-class F1 and positive-class F1 |
| Optimizer | AdamW |
| Client LR | `1.5e-4` |
| Server LR | `1.5e-4` |
| AdamW betas | `(0.9, 0.95)` |
| Weight decay | `0.05` |
| LR decay / scheduler | none |
| Batch size | `256` |
| Max global rounds | `250` |
| Local epochs | `1` |
| Client partitions | `5` |
| Server replicas | one per partition |
| Aggregation | uniform client/server averaging |
| Early stopping | validation Macro F1, patience `10` |
| Saved checkpoint | best validation-Macro-F1 client/server weights |
| Training display | global epoch, val Macro F1/loss, patience, per-partition train Macro F1/loss |

### Paper2601 Evaluation

| Field | Value |
| --- | --- |
| Segment probability | sigmoid of classifier logit |
| Patient probability | mean segment positive probability |
| Patient decision | positive if probability >= `0.5` |
| Final report | `/a/`, `/i/`, combined `/a+i/` on EENT and SVD |
| Extra diagnostics | segment-level `/a/` and `/i/` loss + Macro F1 |

## 5. Cross-Model Interpretation Tables

### Checkpoint Selection

| Model / Stage | Selection Rule |
| --- | --- |
| Centralized CNN | best validation loss |
| Split CNN | best validation loss |
| Split SSAST | best validation loss |
| Paper2601 Stage 1 | no selection, final pretraining round |
| Paper2601 Stage 2 | best validation Macro F1 |

### Losses

| Model / Stage | Loss |
| --- | --- |
| Centralized CNN | BCEWithLogitsLoss |
| Split CNN | BCEWithLogitsLoss |
| Split SSAST | CrossEntropyLoss |
| Paper2601 Stage 1 | MA-Error |
| Paper2601 Stage 2 | binary focal loss with logits |

### Split Learning Differences

| Model / Stage | Partitions | Local Epochs | Aggregation | Server Replicas |
| --- | ---: | ---: | --- | --- |
| Split CNN | `5` | `5` | uniform client/server averaging | one per partition |
| Split SSAST | `5` | `5` | uniform client/server averaging | one per partition |
| Paper2601 Stage 1 | `5` | `1` | uniform client/server averaging | one per partition |
| Paper2601 Stage 2 | `5` | `1` | uniform client/server averaging | one per partition |

### Key Confounders

| Confounder | Why It Matters |
| --- | --- |
| Paper2601 selects by Macro F1 | Other models select by validation loss. |
| Paper2601 has Stage 1 pretraining | Extra training cost and different initialization. |
| Paper2601 has no static features in current run | Paper-style static-feature branch is not active. |
| SSAST is pretrained, Paper2601 is not by default | Pretraining source differs. |
| Split models use patient partitions | Client data heterogeneity can affect convergence. |
| EENT vs SVD | EENT is in-domain held-out; SVD is external-domain. |

## 6. Analysis Questions

| Question | Compare |
| --- | --- |
| Does Split Learning hurt CNN performance? | Centralized CNN vs Split CNN |
| Does SSAST help over Split CNN? | Split SSAST vs Split CNN |
| Does Paper2601 justify Stage 1 cost? | Paper2601 vs Split SSAST and Split CNN |
| Does a model overfit EENT? | EENT combined vs SVD combined gap |
| Is a gain threshold-dependent? | AUC vs fixed-threshold F1/accuracy |
| Is there a sensitivity/specificity tradeoff? | sensitivity and specificity columns |
| Is Paper2601 comparison confounded? | validation Macro F1 selection vs val-loss selection |

