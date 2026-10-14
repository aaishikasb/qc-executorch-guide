# IoT Guide: Arduino VENTUNO Q

Build an edge AI application on Arduino VENTUNO Q: connect Linux and the microcontroller through Arduino Bridge, then work through an add model, DINOv2 vision, and Qwen 0.6B text generation (example) with ExecuTorch and Qualcomm QNN.

[All tracks](../README.md) · [Accompanying artifacts](artifacts/README.md)

## Attendee journey

> [!IMPORTANT]
> **ExecuTorch models will be available to install for IoT. Participants do not need to download and compile ExecuTorch from scratch, or export the supplied models.** Install the VENTUNO Q track bundle and compatible runtime using the track team's instructions. The [artifact inventory](artifacts/README.md) lists the required files; the model bundle and installation link are not yet included in this repository.

1. **Prepare the board.** Follow [Setup and compatibility](#setup-and-compatibility) using the supplied Linux ARM64 runners and QCS8275 models.
2. **Prove the hardware connection.** Run [the LED example](#toggle-an-led-from-linux); confirm Linux can control the MCU before connecting inference.
3. **Run a model.** Use the installed [add model](#part-2-run-the-add-model) for a smoke test, then choose [DINOv2 vision](#part-3-run-and-evaluate-dinov2) or [Qwen text generation](#part-4-run-qwen-06b). You do not need both for your project.
4. **Close the loop.** Follow [Build your application](#part-5-build-your-application): show an input, its inference result, and a bounded MCU response such as the LED changing state.
5. **Rehearse on the board.** Check expected outputs and sensor-to-action latency, then repeat after restarting the app. Retain a known input and expected response for the demo.

Your milestone is a repeatable Linux inference-to-MCU interaction. Runtime builds and custom model export are optional appendices.

## Contents

- [IoT Guide: Arduino VENTUNO Q](#iot-guide-arduino-ventuno-q)
  - [Attendee journey](#attendee-journey)
  - [Contents](#contents)
  - [Device and runtime architecture](#device-and-runtime-architecture)
  - [Setup and compatibility](#setup-and-compatibility)
  - [Part 1: Linux and MCU communication](#part-1-linux-and-mcu-communication)
    - [Arduino Bridge and the Router](#arduino-bridge-and-the-router)
    - [Toggle an LED from Linux](#toggle-an-led-from-linux)
    - [Use the MessagePack-RPC protocol directly](#use-the-messagepack-rpc-protocol-directly)
    - [Bidirectional RPC peers](#bidirectional-rpc-peers)
    - [How this relates to QNN and FastRPC](#how-this-relates-to-qnn-and-fastrpc)
  - [Part 2: Run the add model](#part-2-run-the-add-model)
  - [Part 3: Run and evaluate DINOv2](#part-3-run-and-evaluate-dinov2)
  - [Part 4: Run Qwen 0.6B](#part-4-run-qwen-06b)
  - [Part 5: Build your application](#part-5-build-your-application)
  - [Verify accuracy and NPU performance](#verify-accuracy-and-npu-performance)
  - [Troubleshooting](#troubleshooting)
  - [Appendix A: Build the runtime from source](#appendix-a-build-the-runtime-from-source)
    - [Prepare the checkout and SDK](#prepare-the-checkout-and-sdk)
    - [Build the core](#build-the-core)
    - [Build the Qualcomm examples](#build-the-qualcomm-examples)
  - [Appendix B: Command quick reference](#appendix-b-command-quick-reference)
  - [Additional resources](#additional-resources)
  - [Appendix C: Optional custom model export](#appendix-c-optional-custom-model-export)
    - [Export the add model](#export-the-add-model)
    - [Export DINOv2](#export-dinov2)

## Device and runtime architecture

[Arduino VENTUNO Q](https://docs.arduino.cc/hardware/ventuno-q) combines a **Qualcomm Dragonwing IQ8 / QCS8275** processor running Ubuntu Linux with a **STM32H5F5** microcontroller running the Arduino Zephyr core. Linux runs application logic and the ExecuTorch runtime; the MCU handles sensors, GPIO, and time-critical control. QNN delegates supported model operations to the Hexagon NPU within the QCS8275.

```text
Sensors / actuators ↔ STM32H5F5 ↔ Arduino Bridge / Router ↔ Linux application
                                                            │
                                                      ExecuTorch runtime
                                                            │
                                                         QNN HTP
                                                            │ FastRPC
                                                        Hexagon NPU
```

ExecuTorch captures a PyTorch model, lowers supported operations to a hardware backend, and saves a `.pte` program. The application loads that program with a compatible runtime. QNN's HTP backend supplies Hexagon acceleration; QAIRT supplies the QNN development tools and runtime libraries. CPU preprocessing, tokenization, and any nondelegated operators can still run on Linux.

## Setup and compatibility

1. Complete the board's first-time setup using [Arduino's VENTUNO Q setup guide](https://docs.arduino.cc/tutorials/ventuno-q/sbc-mode-setup/). Use Arduino App Lab for the Python/sketch example and a terminal on the board for native runner commands.
2. Install the track's prepared model/runtime bundle on the supported board image. Record the image version, `uname -m`, and the bundle's ExecuTorch and QAIRT/QNN versions.
3. Confirm that `qnn_executor_runner` and, for the LLM example, `qnn_llama_runner` are Linux ARM64 builds compatible with the board's QNN libraries. If missing, obtain the compatible runtime from the track team.
4. Install `add.pte` and your chosen DINOv2 or Qwen bundle, including sample inputs/reference outputs or the matching tokenizer. See the [artifact inventory](artifacts/README.md).

**Prepared-image paths.** The examples below use the directory layout in the supplied document, with `$HOME` in place of its hard-coded `/home/debian`. Set these variables to the actual locations on your board; this does not install the checkout or runners.

```bash
export EXECUTORCH_ROOT="$HOME/executorch"
export MODEL_OUTPUT="$HOME/model_output"
export QNN_RUNNER="$EXECUTORCH_ROOT/cmake-out/examples/qualcomm/executor_runner/qnn_executor_runner"
export QNN_LLAMA_RUNNER="$EXECUTORCH_ROOT/cmake-out/examples/qualcomm/executor_runner/qnn_llama_runner"
mkdir -p "$MODEL_OUTPUT"

uname -m
test -x "$QNN_RUNNER"
ldd "$QNN_RUNNER"
"$QNN_RUNNER" --help
```

Stop and resolve missing paths or libraries before continuing. The Qwen runner may instead be under `cmake-out/examples/qualcomm/oss_scripts/llama/`; set `QNN_LLAMA_RUNNER` to the executable your build actually produced.

The source document sets `/usr/lib` as the DSP library directory:

```bash
export ADSP_LIBRARY_PATH=/usr/lib
```

Use the image's documented DSP directory if it differs. `ADSP_LIBRARY_PATH` locates Hexagon libraries; Linux host libraries must also resolve through the image's loader configuration or `LD_LIBRARY_PATH`. Keep the HTP stub, DSP skeleton, FastRPC support, and QNN runtime compatible with the board image and model export.

## Part 1: Linux and MCU communication

### Arduino Bridge and the Router

Remote Procedure Call (RPC) lets one processor request a function on another processor. It needs a transport, a message format, and a way to match replies to requests.

Arduino Bridge provides the application API. The Linux `arduino-router` service routes MessagePack-RPC calls between connected peers, including Linux processes and the MCU. Peers register method names; the Router forwards calls to the peer that provides each method. See [Arduino's Bridge guide](https://docs.arduino.cc/tutorials/ventuno-q/bridge/) and the [Router implementation](https://github.com/arduino/arduino-router).

Inspect the service from the board terminal:

```bash
systemctl status arduino-router
systemctl cat arduino-router
journalctl -u arduino-router -f
```

The supplied setup reserves `/dev/ttyHS1` on Linux and `Serial1` on the MCU for the Bridge transport. Do not open the Bridge UART from application code. Check the installed service and board core for the actual port mapping: newer cores can select a dedicated Bridge UART rather than `Serial1`.

| API                                   | Behavior                                                                   |
| ------------------------------------- | -------------------------------------------------------------------------- |
| MCU `Bridge.begin()`                  | Initializes the Bridge transport.                                          |
| Python `Bridge.call(method, args...)` | Calls a remote method and waits for its response.                          |
| MCU `Bridge.call(method, args...)`    | Returns an `RpcCall`; `.result(value)` waits for and retrieves the result. |
| `Bridge.notify(method, args...)`      | Sends a notification without expecting a response.                         |
| MCU `Bridge.provide(name, fn)`        | Registers a callback on the RPC thread; keep it short and thread-safe.     |
| MCU `Bridge.provide_safe(name, fn)`   | Schedules the callback in the main loop context for hardware APIs.         |

The MCU call semantics are documented in [Arduino_RouterBridge](https://github.com/arduino-libraries/Arduino_RouterBridge). Avoid nested `Bridge.call()` or RPC-backed logging such as `Monitor.print()` inside RPC handlers; waiting for another exchange from a handler can deadlock the link. Keep `loop()` responsive so safe callbacks can run.

### Toggle an LED from Linux

Create an app in Arduino App Lab and put this in its sketch. It registers `set_led_state` with `provide_safe` because the callback touches GPIO. The example follows the source document's active-low built-in LED convention; confirm the LED pin and polarity for your board revision.

```cpp
#include <Arduino_RouterBridge.h>

void set_led_state(bool state) {
  digitalWrite(LED_BUILTIN, state ? LOW : HIGH);
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  if (!Bridge.begin()) {
    while (true) {}
  }
  Bridge.provide_safe("set_led_state", set_led_state);
}

void loop() {}
```

Use this as the app's Linux `main.py`, where the App Lab helpers are available:

```python
import time
from arduino.app_utils import App, Bridge

led_state = False

def loop():
    global led_state
    time.sleep(1)
    led_state = not led_state
    Bridge.call("set_led_state", led_state)

App.run(user_loop=loop)
```

Run the app and check that the LED changes once per second. This tests Linux-to-MCU communication independently of model inference.

### Use the MessagePack-RPC protocol directly

The Router's Unix socket in the supplied setup is `/var/run/arduino-router.sock`. A Linux process can call the same MCU handler without using App Lab's Python helpers:

```text
Request:      [0, message_id, method, arguments]
Response:     [1, message_id, error, result]
Notification: [2, method, arguments]

LED request:  [0, 1, "set_led_state", [true]]
LED response: [1, 1, null, null]
```

The message ID ties a response to its request. A Unix stream can split one message across reads or combine messages in a read, so use a streaming MessagePack decoder rather than assuming a single `recv()` returns one response.

The accompanying [msgpack_test.py](artifacts/msgpack_test.py) is a single-call client with a timeout, response validation, and streaming decoding. Copy it to the board, leave the MCU sketch running, and stop the App Lab Python LED loop so the two clients do not compete. From the directory containing the script, run:

```bash
sudo apt install python3-msgpack
/usr/bin/python3 msgpack_test.py 1
/usr/bin/python3 msgpack_test.py 0
```

These commands use the system Python where the distribution package is installed. Use `python -m pip install msgpack` instead if running the script inside a virtual environment. The calling user needs access to the Router socket.

### Bidirectional RPC peers

The supplied document describes a fuller `rpc_base.py` peer with an `ArduinoBridge` class, but includes only excerpts and omits helpers such as `_send()`, `_next_msgid()`, and `close()`. A complete copy is not included in this repository. Use the official Bridge library for application code.

The design is useful for understanding the protocol: a background receiver feeds a persistent `msgpack.Unpacker`; response messages complete pending calls by ID; incoming requests look up registered handlers and send either a result or an error. Notifications dispatch without a reply. A peer advertises its handler names through the Router's `$/register` method.

For example, Linux can provide a `linux_mul` handler while calling an MCU's `mcu_add`. Both handlers must actually be registered by their respective peers. A custom peer also needs synchronized writes, bounded waits, disconnect handling, and cleanup of pending calls; the document's excerpts alone are not a complete RPC library.

### How this relates to QNN and FastRPC

Arduino Bridge and Qualcomm FastRPC both cross processor boundaries, but use different protocols and transports. Bridge routes application-level calls between Linux and the MCU. QNN uses the board's FastRPC infrastructure to execute delegated work on Hexagon. The MessagePack socket and `$/register` method are not the NPU interface.

For inference, export produces the QNN-delegated `.pte`; the ExecuTorch runtime loads it and invokes the delegate. The QNN runtime handles communication with Hexagon. You do not need to write a FastRPC client, and the LED demo is not required to run an NPU model.

## Part 2: Run the add model

Use the installed `add.pte` under `MODEL_OUTPUT`. This small graph checks the inference pipeline before larger models. Run on the board:

```bash
test -s "$MODEL_OUTPUT/add.pte"
mkdir -p "$MODEL_OUTPUT/add-run/outputs"
cd "$MODEL_OUTPUT/add-run"
"$QNN_RUNNER" --model_path "$MODEL_OUTPUT/add.pte"
```

The [upstream runner](https://github.com/pytorch/executorch/blob/v1.5.1/examples/qualcomm/executor_runner/qnn_executor_runner.cpp) uses default inputs when it cannot open `input_list.txt`. In this fresh run directory, that is a load/execution smoke test. A successful run does not establish numerical correctness, NPU delegation, or speedup; verify those separately below.

## Part 3: Run and evaluate DINOv2

Use the installed DINOv2 bundle with its preprocessed sample images, nonempty `input_list.txt`, reference outputs, and label mapping. The runner resolves input paths from the working directory; each line lists one sample's tensor files separated by single spaces. Use filenames without spaces. For new images, follow the bundle's exact shapes, dtypes, layout, and preprocessing.

For a prepared bundle staged under `$MODEL_OUTPUT/dino_v2`:

```bash
cd "$MODEL_OUTPUT/dino_v2"
test -s dino_v2.pte
test -s input_list.txt
mkdir -p outputs
"$QNN_RUNNER" \
  --model_path ./dino_v2.pte \
  --input_list_path ./input_list.txt \
  --output_folder_path ./outputs
```

Compare outputs with PyTorch on the same images, then calculate top-1/top-5 accuracy with the matching label mapping. A `.pte` alone does not provide preprocessing, labels, or accuracy evaluation. Record preprocessing and dataset details with the results.

## Part 4: Run Qwen 0.6B

Use the pre-exported Qwen3 0.6B bundle paired with your board's runtime. Obtain the matching model, tokenizer, quantization details, context limits, and runtime versions from the bundle provider before this step.

```bash
test -x "$QNN_LLAMA_RUNNER"
"$QNN_LLAMA_RUNNER" --help
test -s "$MODEL_OUTPUT/qwen_0p6B/qwen_0p6B_monza.pte"
test -s "$MODEL_OUTPUT/qwen_0p6B/tokenizer.bin"
cd "$MODEL_OUTPUT/qwen_0p6B"
"$QNN_LLAMA_RUNNER" \
  --model_path ./qwen_0p6B_monza.pte \
  --tokenizer_path ./tokenizer.bin \
  --prompt "Why is the sky blue?" \
  --decoder_model_version qwen3
```

The [QNN LLM runner](https://github.com/pytorch/executorch/blob/v1.5.1/examples/qualcomm/oss_scripts/llama/qnn_llama_runner.cpp) drives prompt processing and autoregressive decoding. The tokenizer must match the exported model; changing the decoder flag does not convert a different model architecture into Qwen3. LLM export requires model-specific quantization and cache handling beyond the add exporter.

Model execution can involve multiple delegated partitions and CPU work. Do not assume exactly one FastRPC call per generated token or that the entire application runs on the NPU. Measure time to first token and steady-state decode speed separately.

## Part 5: Build your application

- **Connect sensing to inference.** Read sensors on the MCU, request their values from Linux through Bridge, preprocess on Linux, run the model, and send the result back to a short MCU handler. Keep timing-sensitive control and actuator limits on the MCU.
- **Adapt a model.** Start from the closest vision or LLM exporter. Check operator support, input shapes, representative calibration data, and output accuracy before integrating it into the app.
- **Compare CPU and HTP.** Export separate CPU and QNN versions of the same model, then compare identical inputs and preprocessing. A QNN `.pte` is not a switchable CPU baseline.
- **Make a repeatable demo.** Keep the model loaded where possible and reuse buffers. Use existing runner iteration/profiling flags before modifying its CLI. Keep LLM prompt/decode handling in the LLM runner.

## Verify accuracy and NPU performance

For supplied models, use the bundle’s delegation report and reference outputs. If exporting your own model, inspect delegation in the exporter, after lowering and before calling `to_executorch()`. Here `edge_manager` is the returned `EdgeProgramManager`:

```python
from executorch.devtools.backend_debug import get_delegation_info

info = get_delegation_info(edge_manager.exported_program().graph_module)
print(info.get_summary())
print(info.get_operator_delegation_dataframe())
```

Then verify outputs against PyTorch and profile on the board. With real inputs in the DINOv2 run directory, and after confirming these flags in your runner's `--help`:

```bash
"$QNN_RUNNER" \
  --model_path ./dino_v2.pte \
  --input_list_path ./input_list.txt \
  --output_folder_path ./outputs \
  --warm_up 10 --iteration 100 --profile_level 0
```

Use per-input average latency from the runner logs; check how your runner aggregates iterations before interpreting its final summary. Run a separate diagnostic pass with `--warm_up 0 --iteration 1 --profile_level 1`, preserving the ETRecord from export and ETDump from execution. See [ExecuTorch runtime profiling](https://docs.pytorch.org/executorch/1.5/runtime-profiling.html) for correlating runtime events and delegated work with Inspector.

Record delegated and CPU operations, warmed model latency, end-to-end sensor-to-action latency, peak memory, and sustained thermal behavior. Keep power mode, input set, and other workloads consistent. Hardware profiling and output checks are required before claiming NPU acceleration or an accuracy result.

## Troubleshooting

| Symptom                                       | Check                                                                                                                                                                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Bridge calls hang or the LED does not respond | Check `systemctl status arduino-router` and `journalctl -u arduino-router -f`. Confirm the sketch registered the exact method, avoid nested RPC calls, and keep the MCU loop responsive.                                             |
| Router socket is missing or inaccessible      | Inspect `systemctl cat arduino-router` for the socket path and check the calling user's permissions. Do not bypass the Router by opening its UART.                                                                                   |
| Communication remains stuck                   | After stopping affected apps, run `sudo systemctl restart arduino-router`, then restart the apps so peers reconnect and register their methods.                                                                                      |
| `QCS8275` is rejected                         | Use the intended board checkout and its supported QNN SDK. A different chipset target is not a compatibility fix.                                                                                                                    |
| Python export imports fail                    | Activate the checkout's environment and install its requirements. The document also calls out `numpy`, `ruamel.yaml`, `tabulate`, and `pandas`; a missing `PyQnnManagerAdaptor` requires the QNN AOT build, not just these packages. |
| FlatBuffers tooling is missing                | Install the version required by the checkout; the supplied image uses `sudo apt install flatbuffers-compiler`.                                                                                                                       |
| QNN/HTP cannot initialize                     | Check `ldd` for host-library errors, the image's DSP library directory, FastRPC availability, and matching SDK/runtime versions.                                                                                                     |
| DINOv2 exports but gives no accuracy result   | `--compile_only` skips execution. Check the meaning of the document's `-c` shorthand in your checkout, stage real inputs, and use the board's Linux evaluation workflow.                                                             |
| Runner succeeds with unexpected outputs       | Confirm it opened a nonempty input list rather than falling back to default inputs; check tensor dtype, shape, preprocessing, and labels.                                                                                            |
| Qwen fails to load or produces invalid text   | Check that the `.pte`, tokenizer, decoder type, and runner are from the same compatible bundle.                                                                                                                                      |

For file-access diagnostics, prefix the failing export command with `strace -f -e trace=openat,open,access,stat,statx -s 200 -o /tmp/export-trace.log`. If Router debug logs are needed, inspect its installed service first and use a systemd override to add `--verbose` while preserving its existing executable and arguments; reload systemd and restart the service afterward.

## Appendix A: Build the runtime from source

Participants can skip this appendix and use the installable track models and runtime. These commands adapt the supplied document's native Linux build layout; they are conditional on a **board-supported checkout and SDK**, not a verified recipe for an unmodified upstream release.

### Prepare the checkout and SDK

Obtain the exact source revision used by the board image, including its submodules. In that checkout, with the supported Python version and a working C/C++ toolchain:

```bash
cd "$EXECUTORCH_ROOT"
git rev-parse HEAD
git submodule sync
git submodule update --init --recursive
python3 -m venv env_executorch
source env_executorch/bin/activate
python -m pip install --upgrade pip
python install_requirements.py
```

Install CMake, Ninja, and GCC/G++ if the image does not provide them. Use the checkout's dependency versions. Set `QNN_SDK_ROOT` to the complete matching SDK; headers alone are insufficient:

```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export QNN_SDK_ROOT=/path/to/matching/qairt-sdk
export PYTHONPATH="$EXECUTORCH_ROOT/..${PYTHONPATH:+:$PYTHONPATH}"
test -d "$QNN_SDK_ROOT/include/QNN"
test -d "$QNN_SDK_ROOT/share/QNN/converter/jni"
```

The source document uses a system-layout SDK at `/usr/local`. That layout works only when the matching headers, converter files, and Linux libraries are all installed there. Set `LD_LIBRARY_PATH` to the actual host-library directory if needed, and retain the board image's documented `ADSP_LIBRARY_PATH` for DSP libraries.

**AOT build limitation:** `EXECUTORCH_BUILD_PYBIND=ON` does not by itself guarantee native ARM64 QNN export support. The inspected [upstream QNN CMake configuration](https://github.com/pytorch/executorch/blob/v1.5.1/backends/qualcomm/CMakeLists.txt) gates its QNN Python bindings on x86-64.

### Build the core

From `EXECUTORCH_ROOT`, for a board checkout supporting the supplied build options:

```bash
cmake -S . -B cmake-out -G Ninja \
  -DCMAKE_INSTALL_PREFIX="$EXECUTORCH_ROOT/cmake-out/et-install" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_QNN=ON \
  -DEXECUTORCH_BUILD_DEVTOOLS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_PYBIND=ON \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT" \
  -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build cmake-out --parallel 4
cmake --install cmake-out
```

Stop on configuration or build errors. If your board checkout requires separate AOT and runtime builds, follow that checkout's instructions and use their actual paths in Parts 2–4.

### Build the Qualcomm examples

```bash
cmake -S examples/qualcomm -B cmake-out/examples/qualcomm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$EXECUTORCH_ROOT/cmake-out/et-install;$EXECUTORCH_ROOT/cmake-out;$EXECUTORCH_ROOT/cmake-out/third-party/gflags" \
  -DSUPPORT_REGEX_LOOKAHEAD=ON \
  -DBUILD_TESTING=OFF \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build cmake-out/examples/qualcomm --parallel 4
```

Check where this checkout emits both runners and update `QNN_RUNNER` and `QNN_LLAMA_RUNNER`. Before returning to Part 2, repeat the compatibility checks, verify that QNN Python bindings import in the export environment, and check that the board runner's dynamic libraries resolve. Building C++ binaries alone does not provide QCS8275 target support or missing model bundles.

## Appendix B: Command quick reference

These commands assume the environment variables and prerequisites above are already configured.

| Task                   | Command or section                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| Inspect Router         | `systemctl status arduino-router`                                                        |
| Follow Router logs     | `journalctl -u arduino-router -f`                                                        |
| LED on / off           | `/usr/bin/python3 msgpack_test.py 1` / `/usr/bin/python3 msgpack_test.py 0`              |
| Inspect export options | `python -m examples.qualcomm.scripts.export_example --help`                              |
| Run add smoke test     | `"$QNN_RUNNER" --model_path "$MODEL_OUTPUT/add.pte"` from a fresh run directory          |
| Export DINOv2          | [Appendix C](#appendix-c-optional-custom-model-export); `--compile_only` skips execution |
| Generate text          | [Part 4](#part-4-run-qwen-06b); requires the matching Qwen bundle                        |
| Build core and runners | [Appendix A](#appendix-a-build-the-runtime-from-source)                                  |

## Additional resources

- [Arduino VENTUNO Q hardware and user manual](https://docs.arduino.cc/hardware/ventuno-q)
- [VENTUNO Q Bridge and RPC](https://docs.arduino.cc/tutorials/ventuno-q/bridge/)
- [Arduino Router](https://github.com/arduino/arduino-router) and [Arduino_RouterBridge API](https://github.com/arduino-libraries/Arduino_RouterBridge)
- [ExecuTorch Qualcomm backend](https://docs.pytorch.org/executorch/1.5/backends-qualcomm.html)
- [Qualcomm model examples](https://github.com/pytorch/executorch/tree/v1.5.1/examples/qualcomm): check the board checkout's differences before using upstream commands
- [Qualcomm AI Hub](https://aihub.qualcomm.com/): hosted-device results do not replace measurements on your VENTUNO Q

## Appendix C: Optional custom model export

Only use this section when replacing the supplied models. Obtain the board-supported checkout and export environment first; the attendee runtime installation does not require export tools.

Activate the Python environment supplied with the checkout before exporting. In that environment, check the target mapping:

```bash
python - <<'PY'
from executorch.backends.qualcomm.utils.utils import get_soc_to_chipset_map

targets = get_soc_to_chipset_map()
assert "QCS8275" in targets, "This checkout lacks the required QCS8275 target"
print("QCS8275 target:", targets["QCS8275"])
PY
```

This check establishes the mapping, not full hardware compatibility. Export also requires functioning QNN bindings and an SDK supported by the board build. If export runs on a separate development host, use the same supported checkout/SDK combination there and transfer the generated artifacts to the board.

### Export the add model

Use the prepared QCS8275 export environment from [Setup and compatibility](#setup-and-compatibility). The add graph is a small pipeline check before trying larger models.

```bash
cd "$EXECUTORCH_ROOT"
python -m examples.qualcomm.scripts.export_example --help
python -m examples.qualcomm.scripts.export_example \
  --model_name add \
  --generate_etrecord \
  --output_folder "$MODEL_OUTPUT/" \
  --soc QCS8275
```

Expected output: `add.pte` and profiling metadata under `MODEL_OUTPUT`. The trailing slash also accommodates exporters that construct the ETRecord filename by concatenation. The [upstream exporter](https://github.com/pytorch/executorch/blob/v1.5.1/examples/qualcomm/scripts/export_example.py) selects HTP and uses FP16 when no quantization option is supplied; follow the board checkout's precision requirements. Example-input calibration is sufficient only for this toy graph, not for evaluating a real model's accuracy.

On the board, after transferring the output if export ran elsewhere:

Return to [Part 2](#part-2-run-the-add-model) to run the generated model.

### Export DINOv2

DINOv2 introduces a vision transformer and image preprocessing. Prepare a labeled ImageNet-format validation subset, with the class directories and label mapping expected by the supplied script. Keep held-out evaluation data separate from calibration data. Dataset files and model weights are not bundled here.

The document places `dino_v2.py` under `examples/qualcomm/scripts/`. Use that path only if it exists in your board checkout, and inspect its help first:

```bash
export IMAGENET_VAL="$HOME/datasets/imagenet-mini/val"
cd "$EXECUTORCH_ROOT"
python -m examples.qualcomm.scripts.dino_v2 --help
python -m examples.qualcomm.scripts.dino_v2 \
  --soc_model QCS8275 \
  --build_folder "$EXECUTORCH_ROOT/cmake-out" \
  --artifact "$MODEL_OUTPUT/dino_v2" \
  --dataset "$IMAGENET_VAL" \
  --compile_only
```

`--build_folder` must identify the build directory expected by your script; the example above uses the `cmake-out` tree from Appendix A. Use `build-aot` only if that is the separate tree your supplied environment actually uses.

**`--compile_only` exports the model without on-board evaluation or measured accuracy.** The [shared export utilities](https://github.com/pytorch/executorch/blob/v1.5.1/backends/qualcomm/export_utils.py) exit after compilation in this mode. The inspected [upstream DINOv2 example](https://github.com/pytorch/executorch/blob/v1.5.1/examples/qualcomm/oss_scripts/dino_v2.py) lives in `examples/qualcomm/oss_scripts/`, uses the shared deployment helpers, and is not evidence that the document's board-specific path works on a plain checkout. Confirm the Linux deployment path before attempting its execution mode.

To evaluate on the VENTUNO Q, use the board-specific script's documented Linux evaluation mode, or prepare the `.pte`, preprocessed raw tensors, and a nonempty `input_list.txt` for the board runner. Each line names one sample's input files, separated by single spaces, with the exported shapes, dtypes, and layout. Resolve paths from the runner's working directory and use filenames without spaces.

Return to [Part 3](#part-3-run-and-evaluate-dinov2) to evaluate the prepared bundle.
