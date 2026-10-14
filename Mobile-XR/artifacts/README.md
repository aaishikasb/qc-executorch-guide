# Mobile + XR Artifacts

[Mobile + XR guide](../README.md) · [All tracks](../../README.md)

ExecuTorch models will be available to install for Mobile and XR. Participants do not need to download and compile ExecuTorch from scratch.

## XR download

Download [xr-model-bundle.zip](xr-model-bundle.zip), then follow [installation and device setup](../README.md#install-the-xr-demo-files). The ZIP contains the three models below, `models.json`, `SHA256SUMS`, and a README. It does **not** contain `executor_runner` or an Android AAR; obtain the compatible runtime from the track team.

Validated on Samsung Galaxy XR with ExecuTorch v1.5.1.
Run any file with the shared `executor_runner` (built with XNNPACK and Vulkan).

| File                   | Backend             | Precision               |
| ---------------------- | ------------------- | ----------------------- |
| `mv2_vulkan.pte`       | Vulkan (Adreno GPU) | FP16 (--force_fp16)     |
| `mv2_xnnpack_fp32.pte` | XNNPACK (CPU)       | FP32                    |
| `mv2_xnnpack_q8.pte`   | XNNPACK (CPU)       | INT8 static per-channel |

**Input:** `float32` tensor `1x3x224x224` (NCHW, 602,112 bytes), ImageNet preprocessing (resize 232, center-crop 224, normalize mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
**Output:** `float32` `1x1000` ImageNet-1k logits (4,000 bytes).

After extracting, verify from inside `xr-model-bundle`: `sha256sum -c SHA256SUMS`.

The INT8 model is a smoke-test artifact calibrated on a random tensor; use FP32 or Vulkan FP16 for the real-image demo. See the guide for the recorded accuracy limits.

## Mobile installation

The Galaxy S25 Ultra SM8750 QNN model/runtime bundle and installation link are not yet included here. Obtain the model, matching Android runner or demo app, QNN libraries, sample inputs/reference outputs, and launch instructions from the track team. For app integration, obtain a backend-compatible AAR as well. The XR model ZIP does not supply a mobile QNN demo.

Every distributed bundle should identify its target, backend, model hash, runtime versions, preprocessing, output interpretation, and expected result.
