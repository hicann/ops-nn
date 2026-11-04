# Introduction

English | [简体中文](./README.md)

> Note:
>
> - In the current operator library, most operators run on the AI Core, and a few operators run on the AI CPU. By default, operators mentioned in this project refer to AI Core operators.
> - For the introduction to AI Core and AI CPU, see the "Concepts, Principles, and Terms > Hardware Architecture and Data Processing Principles" section in [Ascend C Operator Development](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC).

This project provides development and invocation samples of AI Core operators and AI CPU operators. Developers can refer to the corresponding implementations based on actual conditions.

## Directory Description

```text
├── examples
│   ├── add_example                # AI Core operator name
│   │   ├── CMakeLists.txt         # Operator compilation configuration file; keep the original file
│   │   ├── examples               # Operator usage example
│   │   ├── op_graph               # Directory related to operator graph construction
│   │   ├── op_host                # Implementations related to the operator information library, Tiling, and InferShape
│   │   └── op_kernel              # Operator kernel directory
│   ├── add_example_aicpu          # AI CPU operator name
│   │   ├── CMakeLists.txt         # Operator compilation configuration file; keep the original file
│   │   ├── examples               # Operator usage example
│   │   ├── op_graph               # Directory related to operator graph construction
│   │   ├── op_host                # Implementations related to the operator information library and InferShape
│   │   └── op_kernel_aicpu        # Operator kernel directory
│   ├── add_example_pypto          # PyPTO operator name; the kernel is written in the PyPTO DSL
│   │   ├── CMakeLists.txt         # Operator compilation configuration file; keep the original file
│   │   ├── examples               # Operator usage example
│   │   ├── op_host                # Implementations related to the operator information library, Tiling, and InferShape
│   │   └── op_kernel              # Operator kernel directory (.py)
│   ├── CMakeLists.txt             # Operator compilation configuration file; keep the original file
│   └── README.md                  # Operator description document

```

## Operator Development Samples

|Sample Directory|  Sample Introduction            |Operator Development|Operator Invocation |
|---|------------------|---|---|
| add_example |  An operator that implements the addition of two tensors.  | For the end-to-end operator development process, see [AI Core Operator Development Guide](../docs/zh/develop/aicore_develop_guide.md) |For invocation, see [README](add_example/README.md)|
|add_example_aicpu|  An operator that implements the addition of two tensors.  |For the end-to-end operator development process, see [AI CPU Operator Development Guide](../docs/zh/develop/aicpu_develop_guide.md)| For invocation, see [README](add_example_aicpu/README.md) |
|add_example_pypto|  An operator that implements the addition of two tensors; the kernel is written in the PyPTO DSL.  |For the end-to-end operator development process, see [README](add_example_pypto/README.md)| For invocation, see [README](add_example_pypto/README.md) |
|fast_kernel_launch_example|  A sample that implements fast end-to-end operator development in the PyTorch scenario.  |For the end-to-end operator development process, see [PyTorch Operator Quick Development Guide](./fast_kernel_launch_example/README_en.md)| For invocation, see [README](fast_kernel_launch_example/README.md) |
