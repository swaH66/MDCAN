# MDCAN: Skin Lesion Classification

PyTorch implementation of **MDCAN (Multi-Scale Dual-Path Cascaded Attention
Network)** for skin lesion classification with a DenseNet121 backbone.

## Architecture

| Manuscript module | Python class | Full name |
|---|---|---|
| DCHA | `DCHA` | Dual-Context Hybrid Attention |
| ESA | `ESA` | Efficient Spatial Attention |
| MSFB | `MSFB` | Multi-Scale Feature Block |
| MSCA | `MSCA` | Multi-Scale Cascaded Attention |
| MDCAN | `MDCAN` | Multi-Scale Dual-Path Cascaded Attention Network |

MDCAN uses a DenseNet121 encoder and two independently instantiated MSCA
blocks. The global path uses adaptive average pooling, while the local path uses
3 x 3 max pooling. Each MSCA applies `DCHA -> ESA -> MSFB`. The two
`1024 x 7 x 7` path outputs are concatenated, projected to 1024 channels,
globally averaged, and classified.

## Repository structure

```text
MDCAN-release/
├── models/
│   ├── __init__.py
│   └── mdcan.py
├── utils/
│   ├── __init__.py
│   └── checkpoint.py
├── tests/
│   ├── __init__.py
│   └── test_mdcan.py
├── test.py
├── train.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The tests use `pretrained=False`, so they do not download ImageNet weights.
The custom dataset and external checkpoint tests are optional.

## Run the model tests

Run all model-only unit tests on CPU:

```bash
python test.py --device cpu --num-classes 3 -v
```

Test a custom dataset on CUDA:

```bash
python test.py \
  --data-path "D:/datasets/ISIC2019" \
  --num-classes 8 \
  --device cuda:0 \
  --max-samples 8 \
  -v
```

`--data-path` accepts either the dataset root:

```text
D:/datasets/ISIC2019/
└── test/
    ├── AK/
    ├── BCC/
    └── ...
```

or the split directory itself:

```text
D:/datasets/ISIC2019/test/
├── AK/
├── BCC/
└── ...
```

Each class directory must contain its images. Class names are sorted to create
deterministic label indices. The unit tests verify:

- the forward output shape for `--num-classes`;
- the public model and module names used in the manuscript;
- independent parameters in the global and local MSCA blocks;
- strict loading of the training checkpoint layout;
- loading of `DataParallel` state dictionaries;
- opening, transforming, batching, and forwarding real images from the custom
  dataset path;
- optional strict loading of an actual MDCAN checkpoint.

If `--data-path` or `--checkpoint` is omitted, only the associated optional
test is skipped. All remaining tests still run.

## Use MDCAN

```python
import torch
from models import create_mdcan

model = create_mdcan(pretrained=False, num_classes=8)
model.eval()

with torch.inference_mode():
    logits = model(torch.randn(1, 3, 224, 224))

print(logits.shape)  # torch.Size([1, 8])
```

Use `num_classes=8` for ISIC2019, `7` for HAM10000, and `3` for ISIC2017.

## Load a trained checkpoint

The training script saves parameters under the `model` key. The loader also
accepts a raw state dictionary or parameters stored under `state_dict` or
`model_state_dict`. Prefixes added by `DataParallel` and `torch.compile` are
removed automatically.

```python
from models import create_mdcan
from utils import load_checkpoint

model = create_mdcan(pretrained=False, num_classes=8)
metadata = load_checkpoint(model, "best_model_f1.pth", strict=True)

assert metadata["missing_keys"] == []
assert metadata["unexpected_keys"] == []
```

The model structure, forward computation, parameter-bearing attribute names,
and state-dict keys are unchanged from the implementation used to train the
reported MDCAN checkpoints. Therefore, those state dictionaries can be loaded
with `strict=True` when `num_classes` matches the dataset.

Test a real checkpoint and a custom dataset before publishing the repository:

```bash
python test.py \
  --device cuda:0 \
  --data-path "D:/datasets/ISIC2019" \
  --checkpoint "D:/checkpoints/best_model_f1.pth" \
  --num-classes 8 \
  --max-samples 8 \
  -v
```

## Train MDCAN

```bash
python train.py \
  --model mdcan \
  --data_path data/isic2019 \
  --num_classes 8 \
  --epochs 100 \
  --batch_size 32 \
  --seed 42 \
  --save_dir checkpoints/isic2019/seed42
```

The dataset directory must contain `train`, `val`, and `test`, with one
subdirectory per class:

```text
data/isic2019/
├── train/class_name/
├── val/class_name/
└── test/class_name/
```

Dataset images are not included in this repository.

## License

This project is released under the MIT License. See `LICENSE`.
