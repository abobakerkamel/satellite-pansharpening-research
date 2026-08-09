# WV3-Transfer HAT-PAN Fusion — 6-Band ×4

Project-specific HAT-PAN architecture, not vanilla HAT.  
Parameters reconstructed from the uploaded checkpoint: **21,087,156**.

Flow: 6-band LR-MS + downsampled PAN → 7-channel HAT → ×4 residual → coarse HR-MS → 13-channel PAN-guided refiner → final 6-band HR-MS.

The HAT checkpoint is a GitHub Release asset: `best_wv3_transfer_hat_6band.pth`.

Quick Colab: `examples/HAT_Quick_Inference_Colab.ipynb`

Historical reproducibility limitation: the original training notebook cloned HAT without recording a fixed git commit.
