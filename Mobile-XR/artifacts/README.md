# XR NPU Artifacts

[Mobile + XR guide](../README.md) · [All tracks](../../README.md)

DeepLab V3 (ResNet-101, INT8) precompiled for the Galaxy XR NPU (Hexagon HTP V69), with the matching QNN runtime. Participants do not need to build ExecuTorch or install the QAIRT SDK; only `adb` is required.

## Download

Download [xr-qnn-bundle.zip](xr-qnn-bundle.zip). The ZIP contains the model, `qnn_executor_runner`, the QNN libraries below, `run_on_galaxy_xr.sh`, `SHA256SUMS`, and this README.

Validated on Samsung Galaxy XR firmware `I610UEU1AYIA` with ExecuTorch v1.5.1 and QAIRT 2.51.0.260929. Firmware `I610UEU2AZF3` blocks NPU access; on those units fall back to the guide's [XNNPACK CPU path](../README.md#xnnpack-fallback--cpu-not-recommended). Check with `adb shell getprop ro.build.version.incremental`.

| File                           | Runs on       | Purpose                              |
| ------------------------------ | ------------- | ------------------------------------ |
| `dlv3_qnn.pte`                 | NPU           | DeepLab V3 model, `--soc_model SXR2230P` |
| `qnn_executor_runner`          | CPU           | Runner for QNN `.pte` files          |
| `libqnn_executorch_backend.so` | CPU           | ExecuTorch QNN delegate              |
| `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpPrepare.so`, `libQnnHtpV69Stub.so` | CPU | QAIRT 2.51 runtime |
| `libQnnHtpV69Skel.so`          | NPU           | NPU-side library (unsigned)          |

**Input:** `float32` tensor `1x3x224x224` (NCHW, 602,112 bytes): resize to 224×224, normalize mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
**Output:** output 0 is `float32` `1x21x224x224` per-pixel logits for the 21 PASCAL VOC classes (argmax for the mask); output 1 is an auxiliary head and can be ignored.

## Run

After extracting, from inside `xr-qnn-bundle`:

```bash
sha256sum -c SHA256SUMS
DEVICE=<headset-ip>:<port> ./run_on_galaxy_xr.sh
```

Expected: `Model executed successfully.` and about **7.2 ms** per inference (50 runs after 10 warm-ups). To confirm the NPU was used, `adb logcat -d -b all | grep remote_handle64_open` should show `Successfully opened handle ... libQnnHtpV69Skel.so`.

The model and libraries only work together: do not mix them with libraries from another QAIRT release. The `libQnn*.so` files are distributed under Qualcomm's QAIRT license terms.
