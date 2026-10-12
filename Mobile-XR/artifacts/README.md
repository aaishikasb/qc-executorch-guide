# Mobile + XR Artifacts

Validated on Samsung Galaxy XR (SXR2230P, firmware I610UEU2AZF3) with ExecuTorch v1.5.1.
Run any file with the shared `executor_runner` (built with XNNPACK and Vulkan). Full metadata: `models.json`.

| File | Backend | Precision | Measured on device | Accuracy vs PyTorch | SHA-256 |
|---|---|---|---|---|---|
| `mv2_vulkan.pte` | Vulkan (Adreno GPU) | FP16 (--force_fp16) | load ~440-1300 ms, steady state ~5.4 ms | matches PyTorch top-1 (cosine 0.9999) | `4763ca1ddf2df102…` |
| `mv2_xnnpack_fp32.pte` | XNNPACK (CPU) | FP32 | load ~18-22 ms, steady state ~4.6 ms | matches PyTorch (cosine 1.0000) | `a2ed996c2ce553bb…` |
| `mv2_xnnpack_q8.pte` | XNNPACK (CPU) | INT8 static per-channel | load ~9 ms, steady state ~3.3 ms | SMOKE TEST ONLY: calibrated on one random tensor; top-1 matched PyTorch on 1/3 images | `72a2c49526e43911…` |

**Input:** `float32` tensor `1x3x224x224` (NCHW, 602,112 bytes), ImageNet preprocessing (resize 232, center-crop 224, normalize mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
**Output:** `float32` `1x1000` ImageNet-1k logits (4,000 bytes).

Verify a download: `sha256sum -c SHA256SUMS`
