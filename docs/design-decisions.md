# Design Decisions

Decisions behind the key hyperparameters and architectural choices in PhotoMatch V2.

---

## Hybrid Vector Weights — `CLIP_WEIGHT = 1.0`, `DEFECT_WEIGHT = 5.0`

The hybrid query vector concatenates a 512-dim CLIP embedding with a 5-dim defect score vector. Before concatenation both parts are L2-normalized, which would normally put them on equal footing — but the defect subspace is 512/5 ≈ 100× smaller, so after re-normalization of the full 517-dim vector, the defect component contributes almost nothing to the final cosine similarity.

`DEFECT_WEIGHT = 5.0` re-amplifies the defect subspace so it has meaningful influence on retrieval. The value was chosen empirically: lower values (1–2) produced negligible improvement over CLIP-only; values above 8 began over-penalizing semantically similar images that happened to differ in quality profile. At 5.0, retrieval accuracy on the FiveK evaluation set jumped from 85.0% (CLIP-only) to 98.8%.

`CLIP_WEIGHT = 1.0` is left as a multiplier baseline — only the ratio between the two weights matters.

---

## Defect Score Amplification — Why Not Just Concatenate Raw?

Without weighting, the 5-dim defect vector is dominated by the 512-dim CLIP vector in the L2 norm. After final normalization:

- CLIP contributes ~99.05% of the vector magnitude
- Defect contributes ~0.95%

The ×5.0 weight raises defect contribution to roughly ~32%, enough to meaningfully differentiate images with similar content but different quality profiles (e.g. two landscape shots where one is blurry and the other sharp).

---

## LUT Size — `LUT_SIZE = 17`

A 3D colour LUT is a 17×17×17 grid of RGB→RGB mappings (4,913 grid points). The choice of 17 is a standard in ICC/DCI colour management and a direct trade-off:

| LUT size | Grid points | Memory | Interpolation error |
|----------|-------------|--------|---------------------|
| 9        | 729         | ~9 KB  | high banding         |
| 17       | 4,913       | ~58 KB | low, smooth gradients|
| 33       | 35,937      | ~432 KB| negligible           |
| 65       | 274,625     | ~3.3 MB| near-zero            |

At 17, each LUT is ~58 KB in memory. With 3,499 reference images pre-cached at startup, total LUT cache ≈ 200 MB — acceptable for a server process. Size 33 would push that to ~1.5 GB; size 9 produced visible posterization on smooth gradients (sky, skin tones).

---

## LUT Blend Strength — `LUT_STRENGTH = 0.2`

The final output blends the CLAHE-corrected image with the LUT-corrected version:

```
final = original_clahe × 0.8 + lut_corrected × 0.2
```

A strength of 1.0 (full LUT) produced over-correction artifacts when the retrieved reference had a different colour temperature than the user's image. The LUT captures the expert's per-image artistic intent, not a universal correction — applying it at full strength assumes a closer RAW→edited mapping than is guaranteed by cosine similarity alone.

0.2 was chosen as the highest value that consistently improved colour without introducing hue shifts or saturation clipping on out-of-distribution images. It functions as a conservative nudge toward the reference style rather than a full regrade.

---

## Semantic Guard Threshold — `clip_sim < 0.55`

In style-constrained retrieval, after filtering candidates to the style region, a semantic guard rejects the match if the CLIP-only cosine similarity falls below 0.55 — falling back to the global FAISS top-1 instead.

CLIP ViT-B-32 cosine similarities between unrelated images typically fall in the 0.2–0.45 range. A threshold of 0.55 is above that noise floor while leaving room for images that are stylistically consistent but not identical in subject. Values below 0.5 allowed semantically unrelated images to pass the guard; values above 0.65 triggered too many false fallbacks on legitimate style matches.

---

## CLAHE Parameters — `clipLimit=1.5`, `tileGridSize=(4,4)`

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to the L channel in LAB space.

- **`clipLimit=1.5`**: Controls contrast amplification cap per tile. Standard values are 2.0–4.0, but these amplified noise in underexposed images. 1.5 gives a visible exposure lift without introducing haloing or noise amplification on clean images.
- **`tileGridSize=(4,4)`**: Divides the image into a 4×4 grid (16 tiles). Larger grids (8×8) caused local contrast variations that looked unnatural on uniform regions (clear sky, flat walls). 4×4 is a coarser adaptation that feels closer to a global tone curve.
- **Bilateral filter pre-step**: Before CLAHE, a bilateral filter (`d=5, sigmaColor=20, sigmaSpace=20`) smooths noise in the L channel. Without it, CLAHE amplifies sensor noise in shadows into visible grain.

---

## MobileNetV3Small — Why This Backbone for Defect Detection

The defect detector needs to run on a mobile device (Android, TFLite export path) and produce 5 soft scores rather than a hard classification. MobileNetV3Small was chosen because:

1. **Size**: ~2.5M parameters, 4.1 MB checkpoint — fits in mobile RAM budget.
2. **Speed**: Hardswish activations and Squeeze-Excitation blocks are efficient on ARM.
3. **Transfer quality**: ImageNet-pretrained features generalize well to low-level quality artefacts (blur, noise, compression) because these manifest as texture patterns that early conv layers capture directly.

The custom head reduces 576 backbone features to 128 before the 5 output neurons, adding capacity for the multi-label regression task without inflating the model size.

---

## Why Defect Scores Use Sigmoid, Not Softmax

Defect types are not mutually exclusive — an image can be simultaneously blurry AND underexposed AND noisy. Softmax would force the model to allocate probability mass across categories as if only one defect were present at a time. Sigmoid treats each score as an independent binary probability, allowing multi-label output. All five scores can be simultaneously high (e.g. a dark, blurry, compressed image).

---

## Aesthetic Re-ranking Weight — `aesthetic_weight = 0.3`

After FAISS returns top-k candidates, they are re-ranked by:

```
score = 0.7 × faiss_similarity + 0.3 × aesthetic_score
```

The aesthetic score is a weighted combination of sharpness (0.4), exposure (0.3), contrast (0.2), and noise (0.1) computed on the edited reference image. The 0.3 weight ensures retrieval is still primarily driven by visual similarity, with aesthetics as a tiebreaker. At 0.5+, lower-similarity but aesthetically cleaner references were selected, reducing style coherence.
