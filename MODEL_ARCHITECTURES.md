# Model Architectures

This document describes the exact project architectures used for the two primary six-band pansharpening models.

---

# PanTiny Small 6-Band

## Configuration

| Setting | Value |
|---|---:|
| Scale | ×4 |
| MS bands | 6 |
| Feature dimension | 24 |
| Transformer depth | 4 |
| Attention heads | 3 |
| FFN expansion | 2.0 |
| Parameters | 53,034 |
| Reconstruction | Bicubic + learned residual |

## Architecture

```mermaid
flowchart TD
    A["LR-MS<br/>6 × H × W"] --> B["Bicubic ×4<br/>6 × 4H × 4W"]
    B --> C["MS Encoder<br/>Conv 3×3, 6→24<br/>GELU"]

    P["HR-PAN<br/>1 × 4H × 4W"] --> D["PAN Projection<br/>Conv 3×3, 1→24"]

    C --> E["Concat<br/>48 channels"]
    D --> E

    E --> F["Fusion<br/>Conv 3×3, 48→24<br/>GELU"]
    F --> G["Transformer Block ×4"]

    subgraph TB["Each Transformer Block"]
        T1["LayerNorm2d"]
        T2["Channel Attention<br/>QKV 1×1 + DWConv 3×3<br/>3 heads"]
        T3["Residual Add"]
        T4["LayerNorm2d"]
        T5["GDFN<br/>Expansion 2.0<br/>Depthwise 3×3"]
        T6["Residual Add"]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6
    end

    G --> H["Body residual add"]
    H --> I["Enhanced Conv<br/>3×3 → GELU → 3×3<br/>+ skip"]
    I --> J["Output Conv<br/>24→6"]
    J --> K["6-band residual"]

    B --> L["Final Add"]
    K --> L
    L --> M["HR-MS<br/>6 × 4H × 4W"]
```

## Forward equation

```text
B = Bicubic(LR-MS)

Fms  = MS_Encoder(B)
Fpan = PAN_Projection(PAN)

F = Fusion([Fms, Fpan])

Fbody = F + TransformerBody(F)

R = Output(EnhancedConv(Fbody))

HR-MS = clamp(B + R, 0, 1)
```

## Loss

```text
L =
1.5 × Charbonnier
+
4.0 × SSIM
+
1.5 × Focal Regression
```

---

# Project WV3-Transfer HAT-PAN

## Configuration

| Setting | Value |
|---|---:|
| Scale | ×4 |
| HAT local input channels | 7 |
| Output bands | 6 |
| HAT image size | 64 |
| Window size | 16 |
| Embed dimension | 180 |
| Depths | [6, 6, 6, 6, 6, 6] |
| Heads | [6, 6, 6, 6, 6, 6] |
| MLP ratio | 2 |
| Compress ratio | 3 |
| Squeeze factor | 30 |
| Convolution scale | 0.01 |
| Overlap ratio | 0.5 |
| Upsampler | PixelShuffle |
| Residual connection | 1conv |
| Refiner input | 13 channels |
| Refiner width | 64 |
| Refiner residual blocks | 4 |
| Released checkpoint parameters | 21,087,156 |

## Architecture

```mermaid
flowchart TD
    A["LR-MS<br/>6 × H × W"] --> B["Bicubic ×4<br/>6 × 4H × 4W"]
    P["HR-PAN<br/>1 × 4H × 4W"] --> C["Area Downsample<br/>1 × H × W"]

    A --> D["Concat"]
    C --> D
    D --> E["7-channel input"]

    E --> F["HAT Backbone"]
    subgraph HAT["HAT Backbone"]
        H1["Conv / shallow features"]
        H2["6 Hybrid Attention Groups"]
        H3["6 blocks per group"]
        H4["Window Attention<br/>window=16"]
        H5["Channel Attention"]
        H6["Overlapping Cross-Attention"]
        H7["embed_dim=180<br/>6 heads"]
        H8["PixelShuffle ×4"]
        H1 --> H2 --> H3 --> H4 --> H5 --> H6 --> H7 --> H8
    end

    F --> G["6-band HAT residual"]
    B --> H["Add"]
    G --> H
    H --> I["Coarse HR-MS<br/>6 bands"]

    I --> J["Concat"]
    B --> J
    P --> J
    J --> K["13 channels<br/>6 coarse + 6 bicubic + 1 PAN"]

    K --> L["Refine Head<br/>Conv 3×3, 13→64"]
    L --> M["Residual Refinement Block ×4<br/>64→64"]
    M --> N["Refine Tail<br/>Conv 3×3, 64→6"]
    N --> O["6-band refinement residual"]

    I --> Q["Final Add"]
    O --> Q
    Q --> R["Final HR-MS<br/>6 × 4H × 4W"]
```

## Forward equation

```text
B = Bicubic(LR-MS)

PAN_lr = AreaDownsample(PAN_hr)

R_hat = HAT([LR-MS, PAN_lr])

Coarse = B + R_hat

RefineInput = [Coarse, B, PAN_hr]   # 6 + 6 + 1 = 13 channels

R_refine = Refiner(RefineInput)

HR-MS = clamp(Coarse + R_refine, 0, 1)
```

---

# Transfer Learning

## Source WV3 model

```text
8 MS bands + PAN
→ 9 input channels

8-band HR-MS
→ 8 output channels

Refiner input
→ 17 channels
```

## Local model

```text
6 MS bands + PAN
→ 7 input channels

6-band HR-MS
→ 6 output channels

Refiner input
→ 13 channels
```

```mermaid
flowchart LR
    S["WV3 source model<br/>9 in / 8 out<br/>17-ch refiner"] --> T["Transfer compatible internal tensors"]
    T --> A["Adapt input layer<br/>9→7"]
    T --> B["Adapt output layer<br/>8→6"]
    T --> C["Adapt refiner input<br/>17→13"]
    A --> L["Local six-band HAT-PAN"]
    B --> L
    C --> L
```

The internal transformer structure is retained where tensor shapes are compatible, while the channel-dependent layers are adapted to the six-band local task.

---

# Direct Architecture Comparison

| Component | HAT-PAN | PanTiny |
|---|---|---|
| Main backbone | Hybrid Attention Transformer | Compact channel-attention body |
| Base features | 180 embedding channels | 24 feature channels |
| Main depth | 6 groups × 6 blocks | 4 blocks |
| Heads | 6 | 3 |
| PAN first use | Downsampled PAN concatenated with LR-MS | HR-PAN projected directly |
| PAN second use | Original HR-PAN in CNN refiner | Same PAN feature stream used in main fusion |
| Upsampling | Learned PixelShuffle ×4 | Bicubic ×4 |
| Residual strategy | HAT residual + second CNN residual | Single learned residual over Bicubic |
| CNN refiner | 13→64, 4 residual blocks, 64→6 | Enhanced 3×3 conv block |
| Transfer learning | WV3 → local 6-band | No source-domain transfer in final run |
| Parameters | 21,087,156 | 53,034 |
| Relative size | 1.0× | 397.6× smaller |
