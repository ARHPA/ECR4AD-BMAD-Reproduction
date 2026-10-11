# Reproduction of ECR4AD (Camelyon16-BMAD)

Independent reproduction of the patch-level anomaly detection framework proposed in **ECR4AD** using the pretrained **CONCH ViT-B/16** foundation model on the Camelyon16-BMAD benchmark.

---

## Reproduction Results

Direct comparison between the reported figures in the paper and this reproduction run on Camelyon16-BMAD:

| Metric | Paper | Reproduction |
| :--- | :---: | :---: |
| **AUROC** | 94.62% | **94.53%** |
| **AP** (Average Precision) | 93.72% | **93.41%** |
| **F1-max** | 87.72% | **87.68%** |

---

## Architecture & Parameters

- **Pretrained Frozen Backbone:** CONCH ViT-B/16 (~395.2M frozen parameters).
- **Trainable Modules:** Bottleneck MLP + 4 ViT Decoder blocks (~64.6M trainable parameters).
- **Modulation:** FiLM conditioning is frozen.
- **Attention:** Standard QKV Attention (Linear Attention disabled, Dropout removed).

---

## Setup & Execution

### 1. Environment & Dependencies

Install required dependencies:

```bash
pip install torch torchvision timm einops wandb scipy scikit-learn
```

### 2. Dataset Preparation

Download [`Histopathology_AD.zip`](https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw) and extract it to form the expected directory hierarchy:

```text
camelyon16_256/
├── train/
│   └── good/
├── valid/
│   ├── good/
│   └── Ungood/
└── test/
    ├── good/
    └── Ungood/
```

### 3. CONCH Pretrained Weights

CONCH weights require access from Hugging Face (`MahmoodLab/CONCH`). Specify the path to `pytorch_model.bin` either by setting the environment variable or updating `networks/vit_encoder.py`:

```bash
export CONCH_CKPT="/path/to/conch/pytorch_model.bin"
```

### 4. Configuration & Training

Set `data_root` and `save_path` in `configs/conch/conch_bmad_config_ecr4ad.yaml`, then run:

```bash
python patch_train.py
```

Alternatively, run end-to-end cloud training using `colab/ecr4ad_bmad_colab.ipynb`.

> Training can also run without W&B logging by setting `export WANDB_MODE=disabled`.

### 5. Evaluation

Evaluate checkpoints using `evaluation.py` / `inference7visualize.py` or inspect the end-to-end evaluation workflow in `colab/evaluation_pipeline.ipynb`.

---

## Citations & Credits

If you use this codebase, please cite the original ECR4AD publication:

```bibtex
@article{xiao2026revisiting,
  title={Revisiting Reconstruction-based Anomaly Detection for Whole Slide Image},
  author={Xiao, Bin and Wangulu, Collins and van der Kwast, Theodorus and Yousef, George M and Zabihollahy, Fatemeh},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  publisher={IEEE}
}
```

