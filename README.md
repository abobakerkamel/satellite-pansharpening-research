# Satellite Pansharpening Research

Complete research repository for **six-band ×4 PAN-guided satellite super-resolution**.

The uploaded research notebooks are preserved with their real saved outputs where available.

## Task

```text
6-band LR-MS + high-resolution PAN → 6-band HR-MS
```

## Latest main comparison

| Model | Parameters | PSNR ↑ | SSIM ↑ | SAM ↓ | ERGAS ↓ | L1 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| WV3-Transfer HAT | 21,087,156 | 25.3574 | 0.9210 | 1.9107° | 2.1864 | 0.03421 |
| **PanTiny small** | **53,034** | **25.6589** | **0.9778** | **1.8958°** | **2.1034** | **0.03308** |

For the exact uploaded checkpoints, PanTiny uses about **397.6× fewer parameters** than the project HAT-PAN model.

## Try the models

**PanTiny:** `examples/PanTiny_Quick_Inference_Colab.ipynb`  
The PanTiny checkpoint is included in the repository.

**HAT:** `examples/HAT_Quick_Inference_Colab.ipynb`  
Use the GitHub Release asset `best_wv3_transfer_hat_6band.pth`.

## Structure

```text
notebooks/
├── 00_data_pipeline/
├── 01_baselines/
├── 02_hat/
├── 03_fusion/
├── 04_final_evaluation/
├── 05_arnet/
├── 06_long_training/
├── 07_sentinel_benchmark/
├── 08_pantiny/
└── 90_archive/

models/
├── common/
├── hat/
└── pantiny/

examples/
results/
presentations/
weights/
data/
demo/
docs/
assets/
```

## Recommended notebooks

- PanTiny training/results: `notebooks/08_pantiny/24_PanTiny_6Band_FULL_50_Epochs_Colab.ipynb`
- PanTiny full-scene inference: `notebooks/08_pantiny/31_PanTiny_FULL_3_TIFF_Input_PAN_Result_RECOMMENDED.ipynb`
- HAT transfer: `notebooks/02_hat/08_transfer_WV3_pretrained_HAT_to_6band_data_colab.ipynb`
- Final HAT evaluation/full scene: `notebooks/04_final_evaluation/13_final_SSIM_and_full_resolution_HAT_colab_FIXED.ipynb`

## Presentations

- `presentations/01_satellite_super_resolution_DETAILED_EXPLAINER.pptx`
- `presentations/02_satellite_super_resolution_HAT_WITH_SENTINEL_GAN.pptx`

## Scientific limitation

The full-resolution original scene has no native six-band HR-MS ground truth at PAN resolution. Quantitative metrics are therefore based on the reduced-resolution Wald protocol; full-scene inference is a practical/qualitative product.
