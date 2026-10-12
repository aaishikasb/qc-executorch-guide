# Mobile + XR Guide: Samsung Galaxy S25 Ultra and Galaxy XR
## Developer Guide

This technical guide provides a comprehensive workflow for developers participating in the **Qualcomm × Meta ExecuTorch Hackathon**.

---

## Contents

1. [Core Architectural Concepts](#1-core-architectural-concepts)
   - [Why ExecuTorch?](#why-executorch)
   - [PyTorch to ExecuTorch Workflow](#pytorch-to-executorch-workflow)
   - [Backend Stack Hierarchy](#backend-stack-hierarchy)
2. [ExecuTorch Runtime Overview](#2-executorch-runtime-overview)
3. [Qualcomm AI Engine Backend QNN](#3-qualcomm-ai-engine-backend-qnn)
4. [Setup](#4-setup)
5. [Models and Deployment Strategies](#5-models-and-deployment-strategies)
   - [Choose Your Model Workflow](#choose-your-model-workflow)
   - [Recommended Hackathon Path: Use a Precompiled `.pte` Model](#recommended-hackathon-path-use-a-precompiled-pte-model)
   - [Advanced Path: Compile Your Own Model](#advanced-path-compile-your-own-model)
6. [Android Development Workflow](#6-android-development-workflow)
7. [Testing](#7-testing)
8. [Additional Resources](#8-additional-resources)
9. [Device Details](#9-device-details)

---

# 1. Core Architectural Concepts

## Why ExecuTorch?

ExecuTorch is PyTorch's on-device inference framework, designed to bring AI models from development to deployment across mobile, edge, and embedded devices.

It enables developers to:

- Build and train models in PyTorch.
- Export models into an optimized runtime format.
- Execute inference efficiently on-device.
- Reduce memory overhead and runtime dependencies.
- Use hardware-specific acceleration backends.

For Snapdragon-powered devices, ExecuTorch can integrate with Qualcomm's QAIRT API to take advantage of dedicated AI hardware. This can improve performance, reduce latency, and increase power efficiency compared with CPU-only execution.

This architecture supports responsive, offline, and privacy-preserving AI experiences across use cases such as:

- Computer vision
- Multimodal applications
- Generative AI assistants
- Edge AI applications

## PyTorch to ExecuTorch Workflow

ExecuTorch starts with a native PyTorch model and produces a hardware-specific `.pte` artifact. The artifact is then loaded by the ExecuTorch Runtime for on-device inference.

```text
PyTorch model
     │
     ▼
Export and lower to ExecuTorch
     │
     ▼
Hardware-specific .pte artifact
     │
     ▼
ExecuTorch Runtime
     │
     ▼
CPU, GPU, or NPU backend
```

Understanding the relationship between **export**, **lowering**, and **runtime execution** is essential for working with ExecuTorch.

- **Export** converts the PyTorch model into an ExecuTorch-compatible representation.
- **Lowering** maps supported operations to a selected hardware backend.
- **Runtime execution** loads the resulting `.pte` artifact and runs inference on the target device.

## Backend Stack Hierarchy

ExecuTorch supports multiple acceleration backends. Select the backend according to the target hardware, model compatibility, and performance requirements.

| Backend | Capability | Optimal use case |
|---|---|---|
| **XNNPACK CPU** | Portable CPU acceleration and broad operator compatibility | Maximum compatibility and CPU fallback |
| **Vulkan GPU** | Parallel GPU acceleration | GPU-based mobile inference |
| **Qualcomm QNN NPU** | Snapdragon-specific AI acceleration | Highest performance and efficiency when supported |

> **For Snapdragon devices:** Qualcomm QNN is the preferred backend when the required device access, SDK, and signing conditions are available. XNNPACK and Vulkan provide practical alternatives when QNN/NPU deployment is unavailable or restricted.

---

# 2. ExecuTorch Runtime Overview

The recommended Snapdragon architecture is:

```text
Android Application
        │
        ▼
ExecuTorch Runtime
        │
        ▼
Qualcomm AI Runtime QAIRT SDK
        │
        ▼
Snapdragon NPU
```

For XNNPACK or Vulkan deployment, the backend path changes as follows:

```text
Android Application
        │
        ▼
ExecuTorch Runtime
        ├── XNNPACK backend ──► CPU
        └── Vulkan backend  ──► Adreno GPU
```

## ExecuTorch × Qualcomm AI Runtime QAIRT

The QNN deployment workflow is:

1. Build ahead-of-time AOT components on the host computer.
2. Compile or obtain a backend-specific `.pte` model artifact.
3. Build or package the Android runtime.
4. Push the compiled model and required libraries to the target device.
5. Run inference on the device.

Documentation: [ExecuTorch Qualcomm Backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)

---

# 3. Qualcomm AI Engine Backend QNN

## Hardware Optimization

The Qualcomm QNN backend is designed to map supported model operations to Snapdragon AI hardware. A successful QNN workflow generally requires:

- A compatible Snapdragon SoC model.
- A matching QAIRT/QNN SDK.
- Correct host and Android toolchain configuration.
- Compatible model operators and data types.
- Required device permissions and signed runtime components.

If QNN/NPU access is unavailable for a target device, use an XNNPACK or Vulkan backend artifact instead of treating QNN as a required prerequisite for the hackathon project.

> **Galaxy XR:** QNN/NPU access is not available to third-party code on retail Galaxy XR units (validated 2026-10-12 on firmware `I610UEU2AZF3`). FastRPC refuses to create a CDSP process from `adb shell` and from installed apps, in both unsigned and signed process-domain modes. Use the XNNPACK or Vulkan path for the XR track.

---

# 4. Setup

## Create a Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Clone ExecuTorch

```bash
git clone https://github.com/pytorch/executorch.git
cd executorch

git submodule sync
git submodule update --init --recursive
```

## Install Dependencies

Use a release tag rather than `main`, which can pin nightly packages that have expired. Check out the newest tag (validated with `v1.5.1`) before installing:

```bash
git checkout v1.5.1
git submodule sync && git submodule update --init --recursive
./install_executorch.sh
```

> The repository has no top-level `requirements.txt`; `install_executorch.sh` installs PyTorch, the pinned dependencies and ExecuTorch itself.

## Optional Qualcomm SDK and Android NDK Setup

The build scripts may download or configure some required artifacts automatically. Alternatively, install them manually and set the required environment variables.

### QAIRT SDK

```bash
export QNN_SDK_ROOT=<path_to_qnn_sdk>/2.46.0.260424/
```

### Android NDK

```bash
export ANDROID_NDK_ROOT=<path_to_android_ndk>
```

> **Important:** Use Android NDK 26 for this workflow.

### QAIRT Runtime Libraries

Replace `<specific-toolchain>` with the appropriate toolchain directory for the host environment:

```bash
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/<specific-toolchain>:$LD_LIBRARY_PATH"
```

## Build ExecuTorch

### Path 1: Use a Coding Agent

You may use your preferred coding agent to generate the build commands.

Suggested prompt:

> Build ExecuTorch with Qualcomm backend support for Android, using the Qualcomm examples and the configured QAIRT SDK and Android NDK.

Review all generated commands before executing them.

### Path 2: Use the Qualcomm Build Script

From the ExecuTorch repository root:

```bash
./backends/qualcomm/scripts/build.sh
```

---

# 5. Models and Deployment Strategies

## Compiling Models

The Qualcomm examples provide scripts for exporting and compiling models. Refer to the examples directory:

[ExecuTorch Qualcomm OSS scripts](https://github.com/pytorch/executorch/tree/main/examples/qualcomm/oss_scripts)

### Common Arguments

| Argument | Description |
|---|---|
| `--build_folder build-android` | Uses the Android build directory created during the ExecuTorch build |
| `--soc_model SM8750` | Selects the target Snapdragon SoC model; Galaxy S25 uses SM8750 |
| `--decoder_model <model_to_compile>` | Selects the model to compile |
| `--compile_only` | Generates the `.pte` artifact without running inference |

### Example Pattern

```bash
python <qualcomm_compile_script>.py \
    --build_folder build-android \
    --soc_model SM8750 \
    --decoder_model <model_to_compile> \
    --compile_only
```

> The exact script name and model arguments depend on the example being used. Check the current ExecuTorch Qualcomm examples before running the command.

---

# 6. Android Development Workflow

## Build the Android Runtime

From the ExecuTorch repository root:

```bash
./scripts/build_android_library.sh
```

The resulting Android library can be packaged into an application as an Android Archive (`.aar`).

> **Not yet validated on device.** The app path in this section (AAR, `Module.load`, `module.forward`) has not been run on the Galaxy XR. The Vulkan backend is only included in the AAR if the library build enables it.

## Integrate the Runtime Using an ARR/AAR

Place the generated archive in the application's `libs` directory and add it to the module-level Gradle file:

```kotlin
dependencies {
    implementation(files("libs/executorch.aar"))
}
```

> The archive is commonly referred to as an **AAR**: Android Archive.

## Load the Model

```java
Module module = Module.load("MODEL.pte");
```

For a model stored in the application's assets or copied to an accessible filesystem location, first resolve the appropriate local path and then pass that path to `Module.load`.

## Run Inference

```java
module.forward(/* model inputs */);
```

The exact input objects depend on the model's exported signature. Ensure that the input tensor shapes, data types, layout, and preprocessing match the model used to generate the `.pte` artifact.

## Backend Selection

The application must use a runtime configuration compatible with the model artifact.

| Model artifact | Runtime backend |
|---|---|
| QNN `.pte` | Qualcomm QNN runtime and required QAIRT components |
| XNNPACK `.pte` | ExecuTorch XNNPACK backend |
| Vulkan `.pte` | ExecuTorch Vulkan backend and compatible Vulkan device/driver |

---

# 7. Testing

Test both correctness and performance on the target device.

## Correctness Checks

- Confirm that the model loads without an exception.
- Verify that input preprocessing matches the model's expected format.
- Compare output shapes and data types.
- Compare representative outputs against a known-good reference.
- Test multiple input samples, not only one image or prompt.

## Performance Measurements

Measure:

- Cold-start time
- Model loading time
- Inference latency
- Throughput, if applicable
- Memory usage
- Battery impact
- Thermal behavior during sustained inference

## Backend Comparison

When comparing XNNPACK, Vulkan, and QNN results, use the same:

- Model version
- Input data
- Preprocessing pipeline
- Number of warm-up iterations
- Number of timed iterations
- Output validation method

Numerical outputs may differ slightly between backends because of differences in kernels, precision, operator implementation, and hardware execution. Compare tolerances and task-level accuracy rather than requiring byte-for-byte identical output.

---

# 8. Additional Resources

- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/)
- [PyTorch Export Documentation](https://pytorch.org/docs/stable/export.html)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
- [ExecuTorch GitHub Repository](https://github.com/pytorch/executorch)
- [ExecuTorch Qualcomm Backend Documentation](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)

---

# 9. Device Details

| Property | Mobile track | XR track |
|---|---|---|
| **Device name** | Samsung Galaxy S25 Ultra | Samsung Galaxy XR |
| **Chipset** | Snapdragon® 8 Elite for Galaxy — SM8750-AC | Snapdragon XR2+ Gen 2 |
| **SoC model** | SM8750 | SXR2230P (`ro.soc.model`) |
| **Hexagon version** | v79 | v69 (not accessible on retail units) |
| **GPU** | Adreno | Adreno (Vulkan driver `vulkan.adreno.so`) |
| **Android API level** | — | 34 |
| **Supported backends** | QNN, XNNPACK, Vulkan | XNNPACK, Vulkan |

> Verify these values against the actual target device before compiling a model, for example with `adb shell getprop ro.soc.model`.


# Added Backend Extension — XNNPACK and Vulkan

The mobile guide above is retained as the baseline Qualcomm × Meta ExecuTorch Hackathon workflow. The following additions provide the flags, parameters, artifacts, and runtime commands required to support XNNPACK on the CPU and Vulkan on the GPU. These additions do not replace the mobile/QNN workflow; they add two alternative backend paths.

> **Validation status:** every command in this section was run on a retail Samsung Galaxy XR (SXR2230P, Android API 34) on 2026-10-12, using ExecuTorch `v1.5.1`, Android NDK r26c, Vulkan SDK 1.4.341.1 and a WSL Ubuntu 22.04 host. Measured results are listed under each path.

## Backend Selection

| Backend | Target | Model artifact | Required build option |
|---|---|---|---|
| XNNPACK | CPU | XNNPACK-compatible `.pte` | `-DEXECUTORCH_BUILD_XNNPACK=ON` |
| Vulkan | Adreno GPU | Vulkan-compatible FP16 `.pte` | `-DEXECUTORCH_BUILD_VULKAN=ON` plus `-DGLSLC_PATH=...` |
| QNN | Qualcomm accelerator/NPU | QNN-compatible `.pte` | Existing mobile-guide QNN workflow |

For the XR extension, build one ARM64 Android runner with both XNNPACK and Vulkan enabled. Select the backend through the precompiled model artifact rather than rebuilding the runner for each model.

## Additional Host Parameters

Set these in the Linux host environment. Adapt paths to the workshop machine:

```bash
export DEVICE=YOUR_ADB_SERIAL
# Persistent hackathon project directory
export WORK_DIR="$HOME/executorch-hackathon/work"
mkdir -p "$WORK_DIR"
export ANDROID_NDK_ROOT=/path/to/android-ndk
export GLSLC_PATH=/path/to/vulkansdk/<version>/x86_64/bin/glslc
export TORCH_HOME="$WORK_DIR/torch-cache"
export REMOTE_DIR=/data/local/tmp/executorch_validation
```

`glslc` must come from the **LunarG Vulkan SDK, version 1.4.341.1 or newer**. The `glslc` shipped with the Android NDK does not work: it lacks the `GL_EXT_integer_dot_product` extension, and CMake stops with a fatal error. Install the SDK on the Linux host:

```bash
wget https://sdk.lunarg.com/sdk/download/1.4.341.1/linux/vulkansdk-linux-x86_64-1.4.341.1.tar.xz
mkdir -p "$HOME/vulkansdk"
tar -C "$HOME/vulkansdk" -xJf vulkansdk-linux-x86_64-1.4.341.1.tar.xz
export GLSLC_PATH="$HOME/vulkansdk/1.4.341.1/x86_64/bin/glslc"
```

For wireless debugging, `DEVICE` is the `ip:port` shown by `adb devices`. The port changes when the headset reboots or wireless debugging is toggled.

Validate the target and tools:

```bash
adb -s "$DEVICE" get-state
adb -s "$DEVICE" shell getprop ro.product.cpu.abi
adb -s "$DEVICE" shell getprop ro.build.version.sdk
adb -s "$DEVICE" shell getprop ro.build.fingerprint
adb -s "$DEVICE" shell getprop ro.hardware.vulkan
git describe --tags
python -c 'import torch, torchvision, executorch; print(torch.__version__, torchvision.__version__)'
test -f "$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" && echo ndk-ok
"$GLSLC_PATH" --version
```

Expected on Galaxy XR: `device`, `arm64-v8a`, `34`, a build fingerprint, `adreno`, the ExecuTorch tag, two version numbers, `ndk-ok`, and a `shaderc` version line.

`GLSLC_PATH` must point to an executable `glslc` shader compiler. Vulkan builds require this compiler to generate the shader assets used by the Vulkan backend.

## Build the Shared ARM64 Runner

Run from the ExecuTorch repository root. These are the additional CMake flags for XNNPACK and Vulkan:

```bash
cmake -S . -B "$WORK_DIR/build" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_PRESET_FILE="$PWD/tools/cmake/preset/android.cmake" \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DGLSLC_PATH="$GLSLC_PATH"

cmake --build "$WORK_DIR/build" --target executor_runner -j"$(nproc)"
test -x "$WORK_DIR/build/executor_runner"
```

`EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON` is required: the `executor_runner` target depends on it, the Android preset does not enable it, and configuration fails without it. The build takes several minutes; compiling the Vulkan shaders is the slowest part.

The Release binary keeps debug symbols (about 200 MB). Strip it before pushing; this does not change behavior:

```bash
"$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" \
  -o "$WORK_DIR/executor_runner" "$WORK_DIR/build/executor_runner"
export RUNNER="$WORK_DIR/executor_runner"
```

Stage the shared runner:

```bash
adb -s "$DEVICE" shell "mkdir -p $REMOTE_DIR"
adb -s "$DEVICE" push "$RUNNER" "$REMOTE_DIR/executor_runner"
adb -s "$DEVICE" shell "chmod 755 $REMOTE_DIR/executor_runner"
```

## XNNPACK Path — CPU

Generate or use the precompiled XNNPACK artifact. The example below creates an FP32 MobileNet V2 artifact:

```bash
python -m examples.xnnpack.aot_compiler \
  --model_name mv2 \
  --output_dir "$WORK_DIR"

sha256sum "$WORK_DIR/mv2_xnnpack_fp32.pte"
adb -s "$DEVICE" push "$WORK_DIR/mv2_xnnpack_fp32.pte" "$REMOTE_DIR/"
```

Run it with the shared runner:

```bash
adb -s "$DEVICE" logcat -c
adb -s "$DEVICE" shell "cd $REMOTE_DIR && ./executor_runner \
  --model_path=mv2_xnnpack_fp32.pte \
  --output_file=xnnpack \
  --num_executions=5 \
  --print_output=none"

adb -s "$DEVICE" logcat -d -s ExecuTorch:I '*:S' > "$WORK_DIR/xnnpack.log"
grep -E 'Model loaded in|Model executed successfully|Writing output' "$WORK_DIR/xnnpack.log"
adb -s "$DEVICE" pull "$REMOTE_DIR/xnnpack-0.bin" "$WORK_DIR/xnnpack-0.bin"
```

Expected validation: five successful executions, a 4,000-byte output file for the prepared MobileNet V2 example, and 1,000 finite FP32 output values.

Measured on Galaxy XR: model load about 20 ms, 5 executions in about 35 ms, steady-state latency about 4.6 ms per inference.

## Vulkan Path — Adreno GPU

Generate or use the precompiled Vulkan artifact. The example below creates an FP16 MobileNet V2 artifact:

```bash
python -m examples.vulkan.export \
  --model_name mv2 \
  --force_fp16 \
  --output_dir "$WORK_DIR"

sha256sum "$WORK_DIR/mv2_vulkan.pte"
adb -s "$DEVICE" push "$WORK_DIR/mv2_vulkan.pte" "$REMOTE_DIR/"
```

Run it with the same runner:

```bash
adb -s "$DEVICE" logcat -c
adb -s "$DEVICE" shell "cd $REMOTE_DIR && ./executor_runner \
  --model_path=mv2_vulkan.pte \
  --output_file=vulkan \
  --num_executions=5 \
  --print_output=none"

adb -s "$DEVICE" logcat -d -v brief > "$WORK_DIR/vulkan.log"
grep -E 'Model loaded in|Model executed successfully|Writing output' "$WORK_DIR/vulkan.log"
grep -iE 'AdrenoVK|Driver Path|ExecuTorch Vulkan Delegate' "$WORK_DIR/vulkan.log" | head
adb -s "$DEVICE" pull "$REMOTE_DIR/vulkan-0.bin" "$WORK_DIR/vulkan-0.bin"
```

Expected validation: five successful executions, a 4,000-byte output file for the prepared MobileNet V2 example, and Vulkan/Adreno initialization in logcat (`Driver Path : /vendor/lib64/hw/vulkan.adreno.so`, `Application Name : ExecuTorch Vulkan Delegate`). The first execution may include GPU and pipeline initialization, so separate model-load time from steady-state execution time.

Measured on Galaxy XR: model load about 440–1,040 ms (GPU pipeline creation), 5 executions in about 27 ms, steady-state latency about 5.4 ms per inference.

## Using Attendee-Downloaded Precompiled Models

Attendees may skip the export commands when a track artifact has already been downloaded. They must use the artifact built for the intended backend:

```bash
export MODEL=/path/to/downloaded-model.pte
sha256sum "$MODEL"
adb -s "$DEVICE" push "$MODEL" "$REMOTE_DIR/model.pte"
```

Run the artifact with the same runner:

```bash
adb -s "$DEVICE" shell "cd $REMOTE_DIR && ./executor_runner \
  --model_path=model.pte \
  --output_file=output \
  --num_executions=5 \
  --print_output=none"
```

The runner does not convert a QNN, XNNPACK, or Vulkan artifact into another backend at runtime. Distribute the backend name, expected input format, model hash, and expected output shape with every precompiled `.pte` file.

## Compare XNNPACK and Vulkan Outputs

```bash
adb -s "$DEVICE" pull "$REMOTE_DIR/xnnpack-0.bin" "$WORK_DIR/xnnpack-0.bin"
adb -s "$DEVICE" pull "$REMOTE_DIR/vulkan-0.bin" "$WORK_DIR/vulkan-0.bin"

python - <<'PY'
import os
import numpy as np
root = os.environ["WORK_DIR"]
for backend in ("xnnpack", "vulkan"):
    output = np.fromfile(f"{root}/{backend}-0.bin", dtype=np.float32)
    assert output.size == 1000, (backend, output.size)
    assert np.isfinite(output).all(), backend
    print(backend, "values:", output.size, "top-1 index:", int(output.argmax()))
PY
```

The two backends may use different numeric paths, such as FP32 on XNNPACK and FP16 internally on Vulkan. Exact logits need not match; validate finite values, output shape, and application-level correctness. The top class is a useful sanity check but is not a performance benchmark.

> **This check runs on an all-ones input.** Without `--inputs`, `executor_runner` fills every input with 1.0. Both backends report top-1 index 885 ("velvet"), which is what MobileNet V2 predicts for a flat, featureless image. It confirms the pipeline runs, not that the model is correct.

### Validate on a real image (recommended)

Feed both backends the same preprocessed image and compare against PyTorch on the host. Replace `photo.jpg` with any test image:

```bash
python - photo.jpg <<'PY'
import os, sys, numpy as np, torch
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
root = os.environ["WORK_DIR"]
weights = MobileNet_V2_Weights.DEFAULT
x = weights.transforms()(Image.open(sys.argv[1]).convert("RGB")).unsqueeze(0).contiguous()
x.numpy().astype(np.float32).tofile(f"{root}/img.bin")            # 602,112 bytes, 1x3x224x224
with torch.no_grad():
    mobilenet_v2(weights=weights).eval()(x).numpy().astype(np.float32).tofile(f"{root}/ref.bin")
PY

adb -s "$DEVICE" push "$WORK_DIR/img.bin" "$REMOTE_DIR/"
for m in mv2_xnnpack_fp32 mv2_vulkan; do
  adb -s "$DEVICE" shell "cd $REMOTE_DIR && ./executor_runner --model_path=$m.pte \
    --inputs=img.bin --output_file=${m}_img --print_output=none"
  adb -s "$DEVICE" pull "$REMOTE_DIR/${m}_img-0.bin" "$WORK_DIR/"
done

python - <<'PY'
import os, numpy as np
from torchvision.models import MobileNet_V2_Weights
root, cats = os.environ["WORK_DIR"], MobileNet_V2_Weights.DEFAULT.meta["categories"]
ref = np.fromfile(f"{root}/ref.bin", dtype=np.float32)
for m in ("mv2_xnnpack_fp32", "mv2_vulkan"):
    out = np.fromfile(f"{root}/{m}_img-0.bin", dtype=np.float32)
    cos = float(ref @ out / (np.linalg.norm(ref) * np.linalg.norm(out)))
    top5 = len(set(np.argsort(ref)[-5:]) & set(np.argsort(out)[-5:]))
    print(f"{m:<18} top-1 {cats[out.argmax()]!r} (PyTorch {cats[ref.argmax()]!r})  top-5 overlap {top5}/5  cosine {cos:.4f}")
PY
```

Measured on Galaxy XR with three PASCAL VOC images: XNNPACK FP32 matched PyTorch exactly (top-1 3/3, top-5 5/5, cosine 1.0000); Vulkan FP16 matched top-1 on 3/3 images with cosine 0.9999.

### Quantized XNNPACK (INT8)

`python -m examples.xnnpack.aot_compiler --model_name mv2 --quantize --output_dir "$WORK_DIR"` produces `mv2_xnnpack_q8.pte` (static per-channel INT8). It runs about 30% faster than FP32 on Galaxy XR (about 3.3 ms steady state), but the script calibrates on a single random tensor, so accuracy is poor: on the same three images it matched PyTorch's top-1 on only 1/3 (cosine down to 0.84). Treat it as a smoke test of the INT8 path. For a real project, calibrate with a few hundred representative inputs using `prepare_pt2e` / `convert_pt2e` and `XNNPACKQuantizer`.

## Backend-Specific Troubleshooting

| Symptom | Check |
|---|---|
| `Use of 'EXECUTORCH_BUILD_EXECUTOR_RUNNER' requires 'EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL'` | Add `-DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON`, delete the build directory, and configure again. |
| `glslc from the Vulkan SDK must be installed` or `GL_EXT_integer_dot_product` errors | Use `glslc` from the LunarG Vulkan SDK 1.4.341.1 or newer, not the NDK's. |
| Vulkan build fails during configuration | Confirm `GLSLC_PATH` points to an executable `glslc`. |
| Vulkan model fails to load | Confirm the runner was built with `-DEXECUTORCH_BUILD_VULKAN=ON` and the artifact was exported with `--force_fp16`. |
| XNNPACK model fails to load | Confirm the runner was built with `-DEXECUTORCH_BUILD_XNNPACK=ON` and the artifact is an XNNPACK artifact. |
| Runner exits without an output file | Check the runner, model path, execute permission, and `ExecuTorch` logcat output. |
| `unauthorized` or `offline` in ADB | Reconnect debugging and accept the headset authorization prompt. |
| `adb devices` is empty after a reboot (wireless debugging) | The port changed. Run `adb connect <ip>:<new-port>` and update `DEVICE`. |
| `rg: command not found` | Use `grep -E`, or install ripgrep (`sudo apt install ripgrep`). |
| Vulkan logcat has no Adreno/Vulkan initialization | Verify the model is Vulkan-compatible and inspect the complete logcat, including vendor Vulkan messages. |

## Added Completion Criteria

- Shared `arm64-v8a` runner built with XNNPACK and Vulkan enabled.
- XNNPACK `.pte` artifact hash recorded and executed successfully.
- Vulkan FP16 `.pte` artifact hash recorded and executed successfully.
- Logcat confirms the expected backend initialization.
- Output files contain the expected number of finite values.
- On a real image, each backend's top-1 matches the PyTorch reference.
- Model metadata records backend, input format, output shape, hash, and expected runtime.