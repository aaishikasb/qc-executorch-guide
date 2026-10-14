# Mobile + XR Guide: Samsung Galaxy S25 Ultra and Galaxy XR

Build a mobile or immersive demo with installable ExecuTorch models: Galaxy S25 Ultra (SM8750, Hexagon V79) and retail Galaxy XR (SXR2230P, Hexagon V69) both run the validated QNN NPU path below, with an XNNPACK CPU fallback.

[All tracks](../README.md) · [Accompanying artifacts](artifacts/README.md)

## Attendee journey

> [!IMPORTANT]
> **ExecuTorch models will be available to install for Mobile and XR. Participants do not need to download and compile ExecuTorch from scratch, or export the supplied models.** Obtain the matching runtime from the track team; model files alone are not an executable demo.

- **Mobile / Galaxy S25 Ultra:** install the SM8750 QNN model and Android runtime bundle supplied for this track. Follow its model-specific launch, preprocessing, and output instructions, then [integrate into your app](#android-development-workflow). Show a camera/image input and a visible inference result. The S25 QNN install bundle and runnable sample are not yet included in this repository; obtain them from the track team before starting.
- **XR / Galaxy XR:** download the self-contained [xr-qnn-bundle.zip](artifacts/xr-qnn-bundle.zip), [install it](#install-the-xr-demo-files), and run the [QNN path](#qnn-path--hexagon-npu-v69) on the Hexagon NPU, then connect the result to your headset interaction. QNN is the primary, highest-performance path and is validated on retail Galaxy XR. An [XNNPACK CPU fallback](#xnnpack-fallback--cpu-not-recommended) is available if your firmware blocks NPU access; NPU source builds are not a prerequisite.

For either device, your demo should accept a real input, display the result, and repeat after an app restart. Record end-to-end latency and check sustained behavior using [Testing](#testing). Keep phone and headset measurements separate.

## Install the XR demo files

On a Linux or WSL host with Android platform tools (`adb`) and `unzip`, enable wireless debugging on the headset and authorize the host. Use the `ip:port` shown in the headset's wireless debugging settings as `DEVICE`. Download [xr-qnn-bundle.zip](artifacts/xr-qnn-bundle.zip), then run from its download directory:

```bash
export DEVICE=<headset-ip>:<port>
unzip xr-qnn-bundle.zip
cd xr-qnn-bundle
sha256sum -c SHA256SUMS
adb connect "$DEVICE"
adb -s "$DEVICE" shell getprop ro.build.version.incremental
```

Replace the serial before continuing. Expect the hashes to pass and ADB to report `device`. The bundle is self-contained: it supplies the QNN `.pte`, `qnn_executor_runner`, `libqnn_executorch_backend.so`, the matching QAIRT runtime libraries, and a `run_on_galaxy_xr.sh` script. You do **not** need to build ExecuTorch or install the QAIRT SDK for the demo; only `adb` is required.

> **Firmware check.** The NPU path is validated on firmware `I610UEU1AYIA`. Firmware `I610UEU2AZF3` blocks NPU access; on those units use the [XNNPACK CPU fallback](#xnnpack-fallback--cpu-not-recommended). The `getprop` above reports the firmware build.

Then run the demo from inside `xr-qnn-bundle`:

```bash
DEVICE="$DEVICE" ./run_on_galaxy_xr.sh
```

Expected: `Model executed successfully.` and about **7.2 ms** per inference. The script stages the model, runner, and libraries and runs on the NPU; [QNN Path — Hexagon NPU](#qnn-path--hexagon-npu-v69) documents the same steps run by hand. Python, PyTorch, torchvision, NumPy, and Pillow are needed only for the host-side real-image/reference check; use the track's prepared Python environment for that step.

## Contents

- [Mobile + XR Guide: Samsung Galaxy S25 Ultra and Galaxy XR](#mobile--xr-guide-samsung-galaxy-s25-ultra-and-galaxy-xr)
  - [Attendee journey](#attendee-journey)
  - [Install the XR demo files](#install-the-xr-demo-files)
  - [Contents](#contents)
  - [Core Architectural Concepts](#core-architectural-concepts)
    - [Why ExecuTorch?](#why-executorch)
    - [PyTorch to ExecuTorch Workflow](#pytorch-to-executorch-workflow)
    - [Backend Stack Hierarchy](#backend-stack-hierarchy)
  - [ExecuTorch Runtime Overview](#executorch-runtime-overview)
    - [ExecuTorch × Qualcomm AI Runtime QAIRT](#executorch--qualcomm-ai-runtime-qairt)
  - [Qualcomm AI Engine Backend QNN](#qualcomm-ai-engine-backend-qnn)
    - [Hardware Optimization](#hardware-optimization)
  - [Android Development Workflow](#android-development-workflow)
    - [Integrate the Runtime Using an AAR](#integrate-the-runtime-using-an-aar)
    - [Load the Model](#load-the-model)
    - [Run Inference](#run-inference)
    - [Backend Selection](#backend-selection)
  - [Testing](#testing)
    - [Correctness Checks](#correctness-checks)
    - [Performance Measurements](#performance-measurements)
    - [Backend Comparison](#backend-comparison)
  - [Validated On-Device Workflow](#validated-on-device-workflow)
    - [Backend Selection](#backend-selection-1)
    - [Additional Host Parameters](#additional-host-parameters)
    - [QNN Path — Hexagon NPU (V69)](#qnn-path--hexagon-npu-v69)
    - [XNNPACK Fallback — CPU (not recommended)](#xnnpack-fallback--cpu-not-recommended)
    - [Using Attendee-Downloaded Precompiled Models](#using-attendee-downloaded-precompiled-models)
    - [Backend-Specific Troubleshooting](#backend-specific-troubleshooting)
  - [Additional Resources](#additional-resources)
  - [Device Details](#device-details)
  - [Appendix A: Optional source builds and model export](#appendix-a-optional-source-builds-and-model-export)
    - [Setup](#setup)
      - [Create a Python Environment](#create-a-python-environment)
      - [Clone ExecuTorch](#clone-executorch)
      - [Install Dependencies](#install-dependencies)
      - [Optional Qualcomm SDK and Android NDK Setup](#optional-qualcomm-sdk-and-android-ndk-setup)
        - [QAIRT SDK](#qairt-sdk)
        - [Android NDK](#android-ndk)
        - [QAIRT Runtime Libraries](#qairt-runtime-libraries)
      - [Build ExecuTorch](#build-executorch)
        - [Path 1: Use a Coding Agent](#path-1-use-a-coding-agent)
        - [Path 2: Use the Qualcomm Build Script](#path-2-use-the-qualcomm-build-script)
    - [Models and Deployment Strategies](#models-and-deployment-strategies)
      - [Compiling Models](#compiling-models)
        - [Common Arguments](#common-arguments)
        - [Example Pattern](#example-pattern)
      - [Build the QNN Runner and Compile a Model](#build-the-qnn-runner-and-compile-a-model)
      - [Build the Android Runtime (AAR)](#build-the-android-runtime-aar)
      - [Build the XNNPACK CPU Runner](#build-the-xnnpack-cpu-runner)
      - [Export XNNPACK MobileNet V2](#export-xnnpack-mobilenet-v2)

## Core Architectural Concepts

### Why ExecuTorch?

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

### PyTorch to ExecuTorch Workflow

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

### Backend Stack Hierarchy

ExecuTorch supports multiple acceleration backends. Select the backend according to the target hardware, model compatibility, and performance requirements.

| Backend              | Capability                                                 | Optimal use case                                  |
| -------------------- | ---------------------------------------------------------- | ------------------------------------------------- |
| **Qualcomm QNN NPU** | Snapdragon-specific AI acceleration                        | Highest performance and efficiency — preferred    |
| **XNNPACK CPU**      | Portable CPU acceleration and broad operator compatibility | Fallback only, when QNN/NPU is unavailable        |

> **For Snapdragon devices:** Qualcomm QNN is the preferred backend and the focus of this guide. XNNPACK on the CPU is a portable fallback as it does not use the NPU and is not recommended when QNN is available.

## ExecuTorch Runtime Overview

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

### ExecuTorch × Qualcomm AI Runtime QAIRT

The QNN deployment workflow is:

1. Install the track's prepared model bundle.
2. Confirm its backend and target device (SM8750 / V79 or SXR2230P / V69).
3. Obtain the matching Android runner and QNN libraries from the track team.
4. Push the compiled model and required libraries to the target device.
5. Run inference on the device.

Documentation: [ExecuTorch Qualcomm Backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)

## Qualcomm AI Engine Backend QNN

### Hardware Optimization

The Qualcomm QNN backend is designed to map supported model operations to Snapdragon AI hardware. A successful QNN workflow generally requires:

- A compatible Snapdragon SoC model.
- A matching QAIRT/QNN SDK.
- Correct host and Android toolchain configuration.
- Compatible model operators and data types.

> **Galaxy XR — QNN/NPU validated.** On Galaxy XR (SXR2230P, Hexagon HTP **V69**), the QNN DSP and GPU backends both pass `qnn-platform-validator --backend all --testBackend` with on-device unit tests. FastRPC creates a CDSP process and loads `libQnnHtpV69Skel.so`. QNN is the preferred, highest-performance backend: see [QNN Path — Hexagon NPU](#qnn-path--hexagon-npu-v69). If QNN/NPU is unavailable on a target unit, fall back to the XNNPACK CPU path.

## Android Development Workflow

After the command-line demo works, integrate the model into your Android app. These Kotlin snippets illustrate integration; they are not a complete sample application. The AAR/`Module` path has not been validated on Galaxy XR. Confirm backend support in the supplied AAR before proceeding.

### Integrate the Runtime Using an AAR

Obtain a backend-compatible AAR from the track team. The model bundle and command-line runner do not include an app runtime. Place the supplied archive in the application's `libs` directory and add it to the module-level Gradle file:

```kotlin
dependencies {
    implementation(files("libs/executorch.aar"))
}
```

> The archive is commonly referred to as an **AAR**: Android Archive.

### Load the Model

```kotlin
val module = Module.load("MODEL.pte")
```

For a model stored in the application's assets or copied to an accessible filesystem location, first resolve the appropriate local path and then pass that path to `Module.load`.

### Run Inference

```kotlin
module.forward(/* model inputs */)
```

The exact input objects depend on the model's exported signature. Ensure that the input tensor shapes, data types, layout, and preprocessing match the model used to generate the `.pte` artifact.

### Backend Selection

The application must use a runtime configuration compatible with the model artifact.

| Model artifact | Runtime backend                                               |
| -------------- | ------------------------------------------------------------- |
| QNN `.pte`     | Qualcomm QNN runtime and required QAIRT components            |
| XNNPACK `.pte` | ExecuTorch XNNPACK backend                                    |

## Testing

Test both correctness and performance on the target device.

### Correctness Checks

- Confirm that the model loads without an exception.
- Verify that input preprocessing matches the model's expected format.
- Compare output shapes and data types.
- Compare representative outputs against a known-good reference.
- Test multiple input samples, not only one image or prompt.

### Performance Measurements

Measure:

- Cold-start time
- Model loading time
- Inference latency
- Throughput, if applicable
- Memory usage
- Battery impact
- Thermal behavior during sustained inference

### Backend Comparison

When comparing QNN and XNNPACK results, use the same:

- Model version
- Input data
- Preprocessing pipeline
- Number of warm-up iterations
- Number of timed iterations
- Output validation method

Numerical outputs may differ slightly between backends because of differences in kernels, precision, operator implementation, and hardware execution. Compare tolerances and task-level accuracy rather than requiring byte-for-byte identical output.

## Validated On-Device Workflow

This section provides the flags, parameters, artifacts, and runtime commands validated on a Samsung Galaxy XR (SXR2230P, Android API 34). QNN on the Hexagon NPU is the primary path; XNNPACK on the CPU is a portable fallback. It assumes the supplied model and runner bundle from [Install the XR demo files](#install-the-xr-demo-files); building them from source is optional and covered in [Appendix A](#appendix-a-optional-source-builds-and-model-export).

### Backend Selection

| Backend | Target               | Model artifact            | Supplied by                                                        |
| ------- | -------------------- | ------------------------- | ------------------------------------------------------------------ |
| QNN     | Qualcomm Hexagon NPU | QNN-compatible `.pte`     | Track bundle (`qnn_executor_runner` + QNN libs); see Appendix A    |
| XNNPACK | CPU (fallback)       | XNNPACK-compatible `.pte` | Track bundle (`executor_runner` built with `-DEXECUTORCH_BUILD_XNNPACK=ON`) |

Select the backend through the precompiled model artifact. A `.pte` is backend-specific, and the runner does not convert between backends at runtime.

### Additional Host Parameters

Set these in the Linux host environment. Adapt paths to the workshop machine:

```bash
export DEVICE=YOUR_ADB_SERIAL
# Persistent hackathon project directory
export WORK_DIR="$HOME/executorch-hackathon/work"
mkdir -p "$WORK_DIR"
export TORCH_HOME="$WORK_DIR/torch-cache"
```

For wireless debugging, `DEVICE` is the `ip:port` shown in the headset's wireless debugging settings. Connect at the start of every terminal session: a new terminal can restart the adb server, which drops wireless connections. The port changes when the headset reboots or wireless debugging is toggled.

```bash
adb connect "$DEVICE"
```

Validate the target and tools:

```bash
adb -s "$DEVICE" get-state
adb -s "$DEVICE" shell getprop ro.product.cpu.abi
adb -s "$DEVICE" shell getprop ro.build.version.sdk
adb -s "$DEVICE" shell getprop ro.soc.model
```

Expected on Galaxy XR: `device`, `arm64-v8a`, `34`, and `SXR2230P`.

### QNN Path — Hexagon NPU (V69)

This path runs the supplied QNN model on the Snapdragon NPU through the Qualcomm QNN backend. The runner is `qnn_executor_runner`. Obtain the QNN `.pte` (compiled for the device's Hexagon arch), `qnn_executor_runner`, `libqnn_executorch_backend.so`, and the matching QAIRT runtime libraries from the track bundle; building them yourself is optional (see [Appendix A](#build-the-qnn-runner-and-compile-a-model)).

Set `QNN_SDK_ROOT` to the supplied QAIRT runtime libraries. The staging commands reference the extracted bundle in `$WORK_DIR` for the model, runner, and backend, and `$QNN_SDK_ROOT/lib/...` for the QNN/HTP libraries:

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.51.0.260929   # supplied QAIRT runtime libraries
export QNN_REMOTE_DIR=/data/local/tmp/executorch_qnn
```

> **The QAIRT version used to compile the `.pte` must match the runtime libraries you push.** If you hit a `Can't create context from binary` error at runtime, the QNN API version used to compile the `.pte` does not match the runtime libraries; rebuild the model and runtime against the same SDK. On Galaxy XR, the full QNN path was validated with QAIRT **2.51.0.260929**, and `qnn-platform-validator` passes with both **2.50.0.260828** and **2.51.0.260929**.

**Stage the model, runner, backend, and the V69 HTP libraries on device.** The NPU needs the generic HTP library, the system library, the per-arch **V69** stub (CPU-side) and skel (DSP-side) libraries:

```bash
adb -s "$DEVICE" shell "mkdir -p $QNN_REMOTE_DIR"

# QNN runtime libraries (aarch64-android)
adb -s "$DEVICE" push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so"        "$QNN_REMOTE_DIR/"
adb -s "$DEVICE" push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so"     "$QNN_REMOTE_DIR/"
adb -s "$DEVICE" push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV69Stub.so" "$QNN_REMOTE_DIR/"
adb -s "$DEVICE" push "$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so" "$QNN_REMOTE_DIR/"

# V69 skel library (DSP-side, from the hexagon-v69 toolchain dir)
adb -s "$DEVICE" push "$QNN_SDK_ROOT/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so" "$QNN_REMOTE_DIR/"

# Model, runner, and ExecuTorch QNN backend (from the supplied bundle)
adb -s "$DEVICE" push "$WORK_DIR/dlv3_qnn.pte" "$QNN_REMOTE_DIR/"
adb -s "$DEVICE" push "$WORK_DIR/qnn_executor_runner" "$QNN_REMOTE_DIR/"
adb -s "$DEVICE" push "$WORK_DIR/libqnn_executorch_backend.so" "$QNN_REMOTE_DIR/"
adb -s "$DEVICE" shell "chmod 755 $QNN_REMOTE_DIR/qnn_executor_runner"
```

**Verify the backend first** (recommended on a new unit). The QAIRT SDK ships `qnn-platform-validator`, which runs a real on-device compute test on each backend. This is the cheapest check of whether the unit allows NPU access — run it before the model:

```bash
adb -s "$DEVICE" push "$QNN_SDK_ROOT/bin/aarch64-android/qnn-platform-validator" "$QNN_REMOTE_DIR/"
adb -s "$DEVICE" push "$QNN_SDK_ROOT/lib/aarch64-android/." "$QNN_REMOTE_DIR/"
adb -s "$DEVICE" push "$QNN_SDK_ROOT/lib/hexagon-v69/unsigned/." "$QNN_REMOTE_DIR/dsp/"
adb -s "$DEVICE" shell "cd $QNN_REMOTE_DIR && chmod +x qnn-platform-validator \
  && export LD_LIBRARY_PATH=$QNN_REMOTE_DIR \
  && export ADSP_LIBRARY_PATH=$QNN_REMOTE_DIR/dsp \
  && ./qnn-platform-validator --backend all --testBackend"
```

Expected on Galaxy XR: `Unit Test on the backend DSP: Passed.` and `QNN is supported for backend DSP on the device.` The validator should load `libQnnHtpV69CalculatorStub.so` and execute a compute function on Hexagon V69.

> The library push copies the SDK's entire `lib/aarch64-android` folder (about 600 MB). If the DSP unit test fails here, the unit does not allow NPU access — use the XNNPACK path.

**Run on the NPU.** Both `LD_LIBRARY_PATH` (CPU-side `.so` resolution) and `ADSP_LIBRARY_PATH` (Hexagon skel resolution) must point at the device directory. `qnn_executor_runner` prints its log to the terminal (stderr), not to logcat:

```bash
adb -s "$DEVICE" shell "cd $QNN_REMOTE_DIR \
  && export LD_LIBRARY_PATH=$QNN_REMOTE_DIR \
  && export ADSP_LIBRARY_PATH=$QNN_REMOTE_DIR \
  && ./qnn_executor_runner --model_path ./dlv3_qnn.pte --warm_up 10 --iteration 50" 2>&1 \
  | grep -E "Method loaded|Model executed|inferences took"
```

Expected: `Model executed successfully.` and a line such as `50 inferences took 361.1 ms, avg 7.22 ms`.

Measured on Galaxy XR (QAIRT 2.51): DeepLab V3 (ResNet-101, INT8) loads in about 220 ms and runs at **7.22 ms** per inference on Hexagon V69 (50 inferences after 10 warm-up runs).

To confirm the run actually used the NPU, check the FastRPC log for the loaded skel library:

```bash
adb -s "$DEVICE" logcat -d -b all | grep remote_handle64_open
```

The confirmation line is `remote_handle64_open: Successfully opened handle ... libQnnHtpV69Skel.so`. A successful run may also log `remote_handle_control_domain failed for request ID 2 ... No such file or directory` and `open_shell failed for domain 3 ... Permission denied`; both are expected and harmless.

> On Git Bash / MSYS, prefix `adb` commands that use `/data/local/tmp/...` paths with `MSYS_NO_PATHCONV=1`, or the device path is rewritten to a Windows path (`C:/Program Files/Git/data/...`) and the push fails with `secure_mkdirs failed`.

### XNNPACK Fallback — CPU (not recommended)

> XNNPACK runs entirely on the CPU and does not use the NPU. Use it only when the QNN path is unavailable.

Use the supplied `executor_runner` (CPU build) and `mv2_xnnpack_fp32.pte` from the track bundle. Set `RUNNER` to the supplied binary; building the runner and exporting the model yourself is optional (see [Appendix A](#build-the-xnnpack-cpu-runner)).

```bash
export RUNNER="$WORK_DIR/executor_runner"
export REMOTE_DIR=/data/local/tmp/executorch_validation
adb -s "$DEVICE" shell "mkdir -p $REMOTE_DIR"
adb -s "$DEVICE" push "$RUNNER" "$REMOTE_DIR/executor_runner"
adb -s "$DEVICE" shell "chmod 755 $REMOTE_DIR/executor_runner"

adb -s "$DEVICE" push "$WORK_DIR/mv2_xnnpack_fp32.pte" "$REMOTE_DIR/"

adb -s "$DEVICE" shell "cd $REMOTE_DIR && ./executor_runner \
  --model_path=mv2_xnnpack_fp32.pte \
  --output_file=xnnpack \
  --num_executions=5 \
  --print_output=none"
```

### Using Attendee-Downloaded Precompiled Models

Attendees may skip the export commands when a track artifact has already been downloaded. They must use the artifact built for the intended backend:

```bash
export MODEL=/path/to/downloaded-model.pte
sha256sum "$MODEL"
adb -s "$DEVICE" push "$MODEL" "$REMOTE_DIR/model.pte"
```

Run the artifact with the matching runner.

**QNN `.pte`** — stage the QNN libraries and `qnn_executor_runner` first (see [QNN Path](#qnn-path--hexagon-npu-v69)), and use the same QAIRT version the `.pte` was compiled with:

```bash
adb -s "$DEVICE" push "$MODEL" "$QNN_REMOTE_DIR/model.pte"
adb -s "$DEVICE" shell "cd $QNN_REMOTE_DIR \
  && export LD_LIBRARY_PATH=$QNN_REMOTE_DIR \
  && export ADSP_LIBRARY_PATH=$QNN_REMOTE_DIR \
  && ./qnn_executor_runner --model_path ./model.pte --warm_up 10 --iteration 50" 2>&1 \
  | grep -E "Model executed|inferences took"
```

**XNNPACK `.pte`:**

```bash
adb -s "$DEVICE" logcat -c
adb -s "$DEVICE" shell "cd $REMOTE_DIR && ./executor_runner \
  --model_path=model.pte \
  --output_file=output \
  --num_executions=5 \
  --print_output=none"
adb -s "$DEVICE" logcat -d -s ExecuTorch:I '*:S' | grep -E 'Model executed successfully|Writing output'
adb -s "$DEVICE" shell ls -l "$REMOTE_DIR/output-0.bin"
```

An exit code of 0 alone does not confirm the model ran; check the success line and the output file.

The runner does not convert a QNN or XNNPACK artifact into another backend at runtime. Distribute the backend name, expected input format, model hash, and expected output shape with every precompiled `.pte` file. For QNN artifacts, also state the QAIRT version and `--soc_model` used to compile; a mismatch with the device's libraries or Hexagon architecture prevents the model from loading.

### Backend-Specific Troubleshooting

| Symptom                                                                                       | Check                                                                                                               |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| QNN model: `Can't create context from binary`                                                 | QNN API version mismatch — the SDK used to compile the `.pte` differs from the runtime libraries. Rebuild both against the same `QNN_SDK_ROOT`. |
| QNN model fails to load or `Could not open libQnnHtpV69Skel.so`                                | Confirm `--soc_model SXR2230P` (V69) at compile time, that the V69 stub + skel libs were pushed, and that `ADSP_LIBRARY_PATH` points to the device dir. |
| `secure_mkdirs failed` on `adb push` to `/data/local/tmp/...` (Git Bash)                      | MSYS rewrote the device path to a Windows path. Prefix the command with `MSYS_NO_PATHCONV=1`.                       |
| `Use of 'EXECUTORCH_BUILD_EXECUTOR_RUNNER' requires 'EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL'` | Add `-DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON`, delete the build directory, and configure again.                 |
| XNNPACK model fails to load                                                                   | Confirm the runner was built with `-DEXECUTORCH_BUILD_XNNPACK=ON` and the artifact is an XNNPACK artifact.          |
| QNN runner output missing from logcat                                                         | `qnn_executor_runner` prints to the terminal (stderr), not logcat. Pipe with `2>&1`.                               |
| Runner exits without an output file                                                           | Check the runner, model path, execute permission, and `ExecuTorch` logcat output.                                   |
| `unauthorized` or `offline` in ADB                                                            | Reconnect debugging and accept the headset authorization prompt.                                                    |
| `adb devices` is empty in a new terminal or after a reboot (wireless debugging)               | Run `adb connect "$DEVICE"`. After a reboot the port changes: use the new port and update `DEVICE`.                 |
| `rg: command not found`                                                                       | Use `grep -E`, or install ripgrep (`sudo apt install ripgrep`).                                                     |

## Additional Resources

- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/)
- [PyTorch Export Documentation](https://pytorch.org/docs/stable/export.html)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
- [ExecuTorch GitHub Repository](https://github.com/pytorch/executorch)
- [ExecuTorch Qualcomm Backend Documentation](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)

## Device Details

| Property               | Mobile track                               | XR track                                  |
| ---------------------- | ------------------------------------------ | ----------------------------------------- |
| **Device name**        | Samsung Galaxy S25 Ultra                   | Samsung Galaxy XR                         |
| **Chipset**            | Snapdragon® 8 Elite for Galaxy — SM8750-AC | Snapdragon XR2+ Gen 2                     |
| **SoC model**          | SM8750                                     | SXR2230P (`ro.soc.model`)                 |
| **Hexagon version**    | v79                                        | v69                                       |
| **GPU**                | Adreno                                     | Adreno                                    |
| **Android API level**  | —                                          | 34                                        |
| **Supported backends** | QNN, XNNPACK                               | QNN, XNNPACK                              |

> Verify these values against the actual target device before compiling a model, for example with `adb shell getprop ro.soc.model`.

## Appendix A: Optional source builds and model export

Skip this appendix for the installable attendee models. Use it only to customize a runtime or export a different model. Model export creates a `.pte`; building ExecuTorch and the Qualcomm runner builds the runtime that runs it.

### Setup

#### Create a Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

#### Clone ExecuTorch

```bash
git clone https://github.com/pytorch/executorch.git
cd executorch

git submodule sync
git submodule update --init --recursive # this is a 3GB+ download, do it before or outside of limited network, otherwise skip entirely
```

#### Install Dependencies

Use a release tag rather than `main`, which can pin nightly packages that have expired. Check out the newest tag (validated with `v1.5.1`) before installing:

```bash
git checkout v1.5.1
git submodule sync && git submodule update --init --recursive
./install_executorch.sh
```

> The repository has no top-level `requirements.txt`; `install_executorch.sh` installs PyTorch, the pinned dependencies and ExecuTorch itself.

#### Optional Qualcomm SDK and Android NDK Setup

The build scripts may download or configure some required artifacts automatically. Alternatively, install them manually and set the required environment variables.

##### QAIRT SDK

```bash
export QNN_SDK_ROOT=<path_to_qnn_sdk>/2.51.0.260929/
```

> The ExecuTorch Qualcomm docs recommend QNN 2.37.0 for stability, but newer SDKs work. On Galaxy XR, the full QNN path (build, compile, run on the NPU) was validated with QAIRT **2.51.0.260929**, and `qnn-platform-validator` passes with both **2.50.0.260828** and **2.51.0.260929**. Use one SDK for everything: the `build.sh` build, the `.pte` compile, and the libraries pushed to the device. If you hit a `Can't create context from binary` error at runtime, the QNN API version used to compile the `.pte` does not match the runtime libraries you should rebuild the model and runtime against the same SDK.
>
> `source $QNN_SDK_ROOT/bin/envsetup.sh` sets `LD_LIBRARY_PATH` and the other QNN tool/library variables for you.

##### Android NDK

```bash
export ANDROID_NDK_ROOT=<path_to_android_ndk>
```

> **Important:** Use Android NDK 26 for this workflow.

##### QAIRT Runtime Libraries

Replace `<specific-toolchain>` with the appropriate toolchain directory for the host environment:

```bash
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/<specific-toolchain>:$LD_LIBRARY_PATH"
```

#### Build ExecuTorch

##### Path 1: Use a Coding Agent

You may use your preferred coding agent to generate the build commands.

Suggested prompt:

> Build ExecuTorch with Qualcomm backend support for Android, using the Qualcomm examples and the configured QAIRT SDK and Android NDK.

Review all generated commands before executing them.

##### Path 2: Use the Qualcomm Build Script

From the ExecuTorch repository root:

```bash
./backends/qualcomm/scripts/build.sh
```

### Models and Deployment Strategies

#### Compiling Models

The Qualcomm examples provide scripts for exporting and compiling models. Refer to the examples directory:

[ExecuTorch Qualcomm OSS scripts](https://github.com/pytorch/executorch/tree/main/examples/qualcomm/oss_scripts)

##### Common Arguments

| Argument                             | Description                                                          |
| ------------------------------------ | -------------------------------------------------------------------- |
| `--build_folder build-android`       | Uses the Android build directory created during the ExecuTorch build |
| `--soc_model SM8750`                 | Selects the target Snapdragon SoC model; Galaxy S25 uses SM8750      |
| `--decoder_model <model_to_compile>` | Selects the model to compile                                         |
| `--compile_only`                     | Generates the `.pte` artifact without running inference              |

##### Example Pattern

```bash
python <qualcomm_compile_script>.py \
    --build_folder build-android \
    --soc_model SM8750 \
    --decoder_model <model_to_compile> \
    --compile_only
```

> The exact script name and model arguments depend on the example being used. Check the current ExecuTorch Qualcomm examples before running the command.

#### Build the QNN Runner and Compile a Model

Follow the [ExecuTorch Qualcomm backend docs](https://docs.pytorch.org/executorch/main/backends-qualcomm.html) for the latest commands.

**Host environment:**

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.51.0.260929
export ANDROID_NDK_ROOT=/path/to/android-ndk-r26c
export EXECUTORCH_ROOT="$PWD"            # run from the executorch repo root
source "$QNN_SDK_ROOT/bin/envsetup.sh"   # sets LD_LIBRARY_PATH for QNN tools
export PYTHONPATH="$EXECUTORCH_ROOT/..:$PYTHONPATH"
export QNN_REMOTE_DIR=/data/local/tmp/executorch_qnn
```

**Build the QNN AOT components and the Android runtime:**

```bash
cd "$EXECUTORCH_ROOT"
./backends/qualcomm/scripts/build.sh --release
```

This produces `build-android/examples/qualcomm/executor_runner/qnn_executor_runner` and `build-android/backends/qualcomm/libqnn_executorch_backend.so`.

> `build.sh` deletes `build-android` and `build-x86_64` and rebuilds from scratch on every run. To rebuild incrementally after small changes, add `--no_clean`. After switching `QNN_SDK_ROOT` to a different SDK version, run a clean build.

**Compile a model to a QNN `.pte`.** Use `--soc_model SXR2230P` for Galaxy XR (confirmed in the ExecuTorch `QcomChipset` enum as `SXR2230P = 53`, Hexagon **V69**). The example below compiles DeepLab V3; the Qualcomm examples live under [`examples/qualcomm`](https://github.com/pytorch/executorch/tree/main/examples/qualcomm).

The script calibrates INT8 quantization on PASCAL VOC 2012 images. With `--download`, torchvision fetches the 2 GB dataset from a server that often runs at under 1 MB/s (hours). Pre-fetch it from a mirror first; torchvision verifies the checksum and skips its own download:

```bash
mkdir -p deeplab_v3/voc_image
wget -P deeplab_v3/voc_image http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
md5sum deeplab_v3/voc_image/VOCtrainval_11-May-2012.tar   # must be 6cd6e144f989b92b3379bac3b3de84fd
```

```bash
python -m examples.qualcomm.scripts.deeplab_v3 \
  --build_folder build-android \
  --soc_model SXR2230P \
  --compile_only \
  --download

# Output: ./deeplab_v3/dlv3_qnn.pte
```

> **`--soc_model` must match the device.** Galaxy XR = `SXR2230P` (V69). Galaxy S25 Ultra = `SM8750` (V79). A model compiled for the wrong Hexagon arch fails to load on device. The full chipset list is the `QcomChipset` enum in `backends/qualcomm/serialization/qc_schema.py`.

Stage `dlv3_qnn.pte`, `qnn_executor_runner`, `libqnn_executorch_backend.so`, and the V69 HTP libraries as shown in [QNN Path — Hexagon NPU](#qnn-path--hexagon-npu-v69).

#### Build the Android Runtime (AAR)

From the ExecuTorch repository root:

```bash
./scripts/build_android_library.sh
```

The resulting Android library can be packaged into an application as an Android Archive (`.aar`).

> **Not yet validated on device.** The app path (AAR, `Module.load`, `module.forward`) has not been run on the Galaxy XR.

#### Build the XNNPACK CPU Runner

Build the CPU runner from the ExecuTorch repository root:

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
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON

cmake --build "$WORK_DIR/build" --target executor_runner -j"$(nproc)"
```

`EXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON` is required: the `executor_runner` target depends on it, and the Android preset does not enable it.

The Release binary keeps debug symbols (about 200 MB). Strip it before pushing; this does not change behavior:

```bash
"$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" \
  -o "$WORK_DIR/executor_runner" "$WORK_DIR/build/executor_runner"
export RUNNER="$WORK_DIR/executor_runner"
```

#### Export XNNPACK MobileNet V2

Generate the FP32 MobileNet V2 artifact used by the [XNNPACK fallback](#xnnpack-fallback--cpu-not-recommended):

```bash
python -m examples.xnnpack.aot_compiler --model_name mv2 --output_dir "$WORK_DIR"
sha256sum "$WORK_DIR/mv2_xnnpack_fp32.pte"
adb -s "$DEVICE" push "$WORK_DIR/mv2_xnnpack_fp32.pte" "$REMOTE_DIR/"
```
