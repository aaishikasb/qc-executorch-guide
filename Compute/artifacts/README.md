# Compute Artifacts

[Compute guide](../README.md) · [All tracks](../../README.md)

ExecuTorch models will be preinstalled on Compute devices. Participants do not need to download and compile ExecuTorch from scratch.

The track team must provide the installed bundle location and its model-specific instructions. The guide uses the following DeepLab example layout; these files are not bundled in this repository:

- `dlv3_qnn.pte`, exported for SC8380XP / HTP v73.
- Windows ARM64 `qnn_executor_runner.exe`, `qnn_executorch_backend.dll`, matching QNN DLLs, and v73 DSP skeleton/catalog files.
- `input_list.txt`, sample input tensors, preprocessing instructions, reference outputs, and mask decoding/rendering instructions.
- Model provenance/hash, ExecuTorch and QAIRT versions, input/output specification, and delegation information. Supply a profiling-enabled runner for the guide's diagnostic checks.

Start at [Windows ARM64 Deployment](../README.md#windows-arm64-deployment). If the installation is incomplete, ask the track team to repair it before proceeding. Source build and packaging instructions are in the optional appendix.
