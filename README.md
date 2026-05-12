# RetinexMamba 🌙→☀️
### Low-Light Image Enhancement via Retinex-Guided State Space Models

<p align="center">
  <!-- Replace the src below with your actual result image once you have it -->
  <!-- <img src="assets/results/teaser.png" width="900px"> -->
  <em>Results coming soon — paste your before/after images here</em>
</p>

---

## Overview

**RetinexMamba** is a deep learning architecture for low-light image enhancement that fuses two powerful ideas:

1. **Retinex Theory** — decomposes an image into reflectance and illumination. By explicitly estimating the illumination map, the model knows *where* an image is dark and by how much.
2. **State Space Models (Mamba)** — models long-range pixel dependencies across the entire image sequence without the quadratic cost of self-attention, enabling global context at scale.

The result is a lightweight (~XM parameters) U-Net-style network that enhances dark images with high fidelity, natural color, and sharp structure.

---

## Architecture

```
Input (B, 3, H, W)
    │
    ├──► IlluminationEstimator ──► Illumination Map L (B, 3, H, W)
    │         (Retinex branch)
    │
    ▼
ShallowExtract (2× Conv)
    │
    ▼
Encoder L1  ──[ RetinexMambaBlock(dim=48)    ]──┐  skip
    ↓ ×2 downsample                              │
Encoder L2  ──[ RetinexMambaBlock(dim=96)    ]──┤  skip
    ↓ ×2 downsample                              │
Bottleneck  ──[ RetinexMambaBlock(dim=192)   ]  │
    ↑ ×2 upsample                                │
Decoder L2  ──[ RetinexMambaBlock(dim=96)    ]◄─┤  + skip_attn
    ↑ ×2 upsample                                │
Decoder L1  ──[ RetinexMambaBlock(dim=48)    ]◄─┘  + skip_attn
    │
    ▼
Refinement  ──[ RetinexMambaBlock(dim=48) × 2 ]
    │
    ▼
HVIColorCorrection  (intensity / hue separation)
    │
    ▼
Output Conv  +  Global Residual (inp)
    │
    ▼
Enhanced Image (B, 3, H, W)  ·  Illumination Map (B, 3, H, W)
```

### Key Modules

| Module | Role |
|---|---|
| `IlluminationEstimator` | Lightweight CNN (1×1 → depthwise 5×5 → 1×1) that estimates a per-pixel illumination map from the mean+RGB input. Grounded in Retinex: *I = R × L*. |
| `MambaBlock` | Flattens spatial tokens → runs a selective SSM (Mamba) for global context → reshapes back → gated FFN. |
| `RetinexMambaBlock` | Projects the illumination map into feature space and fuses it with image features before passing through a stack of MambaBlocks. Low-illumination regions receive stronger correction. |
| `HVIColorCorrection` | Parallel intensity and hue branches with learnable blend weights (α, β). Prevents the color distortion common in brightness-only enhancement methods. |

---

## Results

<!-- Paste your result images below. Suggested layout: -->
<!--
### Quantitative (LOL Benchmark)

| Method | PSNR ↑ | SSIM ↑ | Params |
|---|---|---|---|
| RetinNet | 16.77 | 0.560 | — |
| KinD | 20.87 | 0.800 | — |
| SNR-Aware | 21.48 | 0.849 | — |
| **RetinexMamba (ours)** | **XX.XX** | **0.XXX** | **X.XM** |

### Qualitative

<p align="center">
  <img src="assets/results/comparison_1.png" width="900px">
  <br>
  <img src="assets/results/comparison_2.png" width="900px">
</p>
-->

> 📌 **Results will be added here.** Paste your PSNR/SSIM table and before/after comparison images into the section above once training is complete.

---

## Dataset — LOL (Low-Light)

The model is trained and evaluated on the **LOL dataset** ([Wei et al., 2018](https://arxiv.org/abs/1808.04560)):

- **Train:** 485 low/normal-light image pairs (`our485/`)
- **Test:** 15 image pairs (`eval15/`)

Download from [the official source](https://daooshee.github.io/BMVC2018website/) and place it as:

```
data/
  lol/
    our485/
      low/
      high/
    eval15/
      low/
      high/
```

---
