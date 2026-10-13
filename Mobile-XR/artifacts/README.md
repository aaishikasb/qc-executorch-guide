# Mobile + XR Artifacts

Validated on Samsung Galaxy XR with ExecuTorch v1.5.1.
Run any file with the shared `executor_runner` (built with XNNPACK and Vulkan).

| File                   | Backend             | Precision               |
| ---------------------- | ------------------- | ----------------------- |
| `mv2_vulkan.pte`       | Vulkan (Adreno GPU) | FP16 (--force_fp16)     |
| `mv2_xnnpack_fp32.pte` | XNNPACK (CPU)       | FP32                    |
| `mv2_xnnpack_q8.pte`   | XNNPACK (CPU)       | INT8 static per-channel |

**Input:** `float32` tensor `1x3x224x224` (NCHW, 602,112 bytes), ImageNet preprocessing (resize 232, center-crop 224, normalize mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
**Output:** `float32` `1x1000` ImageNet-1k logits (4,000 bytes).

Verify a download: `sha256sum -c SHA256SUMS`
