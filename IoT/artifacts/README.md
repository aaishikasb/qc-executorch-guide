# IoT Artifacts

[IoT guide](../README.md) · [All tracks](../../README.md)

## Installable attendee bundle

ExecuTorch models will be available to install for IoT. Participants do not need to download and compile ExecuTorch from scratch. The installation link and model/runtime bundle are not yet included here; obtain the VENTUNO Q bundle from the track team.

The attendee bundle needs Linux ARM64 runners and compatible QNN libraries, `add.pte`, and the chosen DINOv2 or Qwen model. DINOv2 needs sample tensors, `input_list.txt`, labels, preprocessing instructions, and reference outputs; Qwen needs the matching tokenizer. Follow [Setup and compatibility](../README.md#setup-and-compatibility) to set paths on the board. Source checkout, SDK, and export datasets below are only needed for custom export/build work.

## Included

| File                                         | Purpose                                                                                                                                        |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| [msgpack_test.py](msgpack_test.py)           | Single-call Linux client for the guide's MCU `set_led_state` handler. Requires `msgpack`, access to the Router socket, and the running sketch. |
| [test_msgpack_test.py](test_msgpack_test.py) | Local protocol checks for fragmented replies, RPC errors, malformed replies, and disconnects; no board required.                               |

Run the checks with `python -m unittest discover -s IoT/artifacts -p 'test_*.py'` from the repository root in an environment with `msgpack` installed.

## Required external files

| Item                                                                                            | Needed for                                                                                                                          |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Board image, QCS8275-capable ExecuTorch checkout/commit, and matching QAIRT/QNN SDK and runtime | Exporting and running models; the stock upstream version inspected for this guide does not list QCS8275.                            |
| Linux ARM64 `qnn_executor_runner` and `qnn_llama_runner`                                        | Running exported models on the board; supplied by the image or built with the supported checkout.                                   |
| Board-specific `dino_v2.py` and a labeled ImageNet-format subset                                | DINOv2 export/evaluation. The document's script path differs from the inspected upstream example.                                   |
| `qwen_0p6B_monza.pte` and its matching `tokenizer.bin`                                          | Qwen3 0.6B text generation; stage under `$MODEL_OUTPUT/qwen_0p6B/` as shown in the guide.                                           |
| Complete `rpc_base.py`, if supplied separately                                                  | Optional custom bidirectional peer described by the document. The official Bridge library covers application use without this file. |

The attendee installation supplies models and sample inputs; optional custom export generates replacement models and ETRecords. Profiling outputs are generated during verification. When sharing a model bundle, include its model/tokenizer provenance, ExecuTorch commit, SDK/runtime versions, QCS8275 target, input specification, and calibration details so another participant can reproduce it.
