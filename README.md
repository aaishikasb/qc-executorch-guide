# ExecuTorch on Qualcomm: Developer Journey Guide

Build on-device AI applications with PyTorch ExecuTorch on Qualcomm-powered PCs, phones, XR headsets, and IoT devices.

> [!NOTE]
> If you're participating in a Qualcomm event, start with your device's track, then follow its prepared-model setup, deployment, and demo checks.

## Start with your track

| Track                                                                                    | Device details                                                                                                                                                                                                                                                                                                 | Guide                                    | Artifacts                                              |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------ |
| **Compute**: local AI applications for PCs                                               | Dell Latitude with **Snapdragon X Elite**; Windows 11 ARM64; Hexagon NPU                                                                                                                                                                                                                                       | [Compute guide](Compute/README.md)       | [Compute artifacts](Compute/artifacts/README.md)       |
| **Mobile + XR**: mobile intelligence, immersive interaction, and multimodal applications | [Galaxy S25 Ultra](https://www.qualcomm.com/smartphones/device-finder/samsung-galaxy-s25-ultra): **Snapdragon 8 Elite for Galaxy**, Android.<br>[Galaxy XR headset](https://news.samsung.com/us/introducing-galaxy-xr-opening-new-worlds/): **Snapdragon XR2+ Gen 2**, Android XR.<br>Both have a Hexagon NPU. | [Mobile + XR guide](Mobile-XR/README.md) | [Mobile + XR artifacts](Mobile-XR/artifacts/README.md) |
| **IoT**: edge AI with sensing and real-time control                                      | [Arduino VENTUNO Q](https://docs.arduino.cc/hardware/ventuno-q): **Dragonwing IQ8 / QCS8275**, Ubuntu Linux, Hexagon NPU; separate STM32H5 microcontroller for control tasks                                                                                                                                   | [IoT guide](IoT/README.md)               | [IoT artifacts](IoT/artifacts/README.md)               |

> [!IMPORTANT]
> **Read the documentation for the track you have been shortlisted for before starting your project. The other tracks are not prerequisites and do not need to be completed.** For independent projects, choose the track that matches your target device.
>
> Each track is independent, with its own setup, deployment, and verification instructions. Follow only your selected track's `README.md`; `artifacts/` holds its accompanying files as they become available.

## Start with the supplied models

**ExecuTorch models will be preinstalled on Compute devices and available to install for Mobile, XR, and IoT. Participants do not need to download and compile ExecuTorch from scratch.** Run the supplied model with its matching runtime first; custom model export and runtime source builds are optional appendix material.

| Track | First working demo | Start here |
| --- | --- | --- |
| Compute | Local image segmentation on the X Elite PC | [Compute attendee journey](Compute/README.md#attendee-journey) |
| Mobile | Phone input to a visible inference result | [Mobile attendee journey](Mobile-XR/README.md#attendee-journey) |
| XR | Headset inference with XNNPACK CPU or Vulkan GPU | [Install the XR demo](Mobile-XR/README.md#install-the-xr-demo-files) |
| IoT | Linux inference connected to an MCU response | [IoT attendee journey](IoT/README.md#attendee-journey) |

The artifact pages identify available files and remaining distribution requirements. The XR model ZIP is included; its runner, the Mobile and IoT install bundles, and the Compute installation location still need to be supplied by the track team. Do not substitute a model or runtime from another track to fill a missing file.

## How ExecuTorch and QNN fit together

- **ExecuTorch** is PyTorch's framework for on-device inference. It exports and optimizes a PyTorch model into a `.pte` program that a lightweight runtime loads inside your application. See [how ExecuTorch works](https://docs.pytorch.org/executorch/1.5/intro-how-it-works.html).
- **QNN**, also called **Qualcomm AI Engine Direct**, provides APIs for Qualcomm accelerators. ExecuTorch's Qualcomm backend delegates supported model operations to QNN. For the Hexagon **neural processing unit (NPU)**, select QNN's **HTP (Hexagon Tensor Processor)** backend. The **Qualcomm AI Runtime (QAIRT) SDK** supplies the QNN tools and runtime libraries. See the [Qualcomm backend guide](https://docs.pytorch.org/executorch/1.5/backends-qualcomm.html).
- **Qualcomm AI Hub** helps you find models and compile, profile, and validate them on hosted devices. It is optional for the local ExecuTorch build workflow. See [AI Hub getting started](https://workbench.aihub.qualcomm.com/docs/hub/getting_started.html).

```text
PyTorch model → export + quantize → lower supported operations to QNN → .pte
Application → ExecuTorch runtime → QNN HTP → Qualcomm Hexagon NPU
```

An NPU-equipped device does not automatically accelerate a CPU-exported model. NPU execution requires a model lowered for that target, the Qualcomm backend, and compatible device runtime libraries. Operations outside the delegated partitions need another supported execution path, often CPU kernels.

## Optional: From your own model to verified NPU inference

1. **Check the target and versions.** Follow your track's host tools, target OS, ExecuTorch release, QAIRT/QNN version, and driver requirements. Keep export and runtime versions aligned; use libraries built for the device's OS and architecture.
2. **Start with a small model and real inputs.** Confirm the PyTorch outputs and input shapes first. For quantized models, calibrate with representative data and check accuracy on held-out inputs.
3. **Export for the device.** Select the correct chipset and QNN HTP backend, inspect which operations were delegated, and package the `.pte` with the required runtime libraries.
4. **Measure on the hardware.** Compare outputs with PyTorch, then measure warmed inference and end-to-end latency, memory, and sustained behavior. Inspect delegate/CPU profiling events and compare with a CPU baseline; successful loading alone does not prove NPU execution.

## Check device support with Qualcomm AI Hub

After [AI Hub setup](https://workbench.aihub.qualcomm.com/docs/hub/getting_started.html), run `qai-hub list-devices` to inspect the current device catalog. Match the chipset, OS, runtime, and compute unit in profile results. A family, proxy, or reference device's timings are not measurements of your physical device; see [AI Hub device selection](https://workbench.aihub.qualcomm.com/docs/hub/devices.html).

- **Compute:** AI Hub lists [Snapdragon X Elite CRD / SC8380XP](https://aihub.qualcomm.com/jobs/jpx738olg). The ExecuTorch target is `SC8380XP`, with HTP v73; benchmark again on the Dell Latitude.
- **Mobile:** use the Galaxy S25 Ultra or S25 family entry. Snapdragon 8 Elite uses ExecuTorch target `SM8750`, with HTP v79.
- **XR:** follow the guide’s validated XNNPACK CPU or Vulkan GPU path on retail Galaxy XR. Its recorded firmware validation does not permit third-party QNN/NPU execution. A reference-device AI Hub result does not change that restriction.
- **IoT:** AI Hub publishes [VENTUNO Q / QCS8275 NPU results](https://aihub.qualcomm.com/jobs/jgdd7k7rg). Follow the standalone [VENTUNO Q setup and compatibility checks](IoT/README.md#setup-and-compatibility) for the board's ExecuTorch/QNN build. Run inference on Linux and connect to the STM32 through Arduino Bridge for sensing and real-time control.

Check the [ExecuTorch chipset mapping](https://github.com/pytorch/executorch/blob/v1.5.1/backends/qualcomm/serialization/qc_schema.py) for your release. AI Hub results from another runtime do not establish ExecuTorch compatibility.
