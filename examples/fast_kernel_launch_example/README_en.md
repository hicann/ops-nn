# Fast Kernel Launch

English | [简体中文](./README.md)

## Introduction

This document demonstrates how to use Ascend C and [PyTorch Extension](https://docs.pytorch.org/tutorials/extension.html) capabilities to develop custom NPU operators.

**Core advantages:**

- **Single deliverable:** One file completes operator development and PyTorch framework adaptation.
- **Efficient invocation:** Uses the `<<<>>>` syntax to launch the kernel. The process is simple and efficient.

## Prerequisites

- Refer to [Environment Deployment](../../docs/zh/install/quick_install.md) first to complete the basic environment setup
- gcc 9.4.0+
- python 3.8+
- torch>=2.6.0
- [TorchNPU](https://gitcode.com/Ascend/pytorch/releases) of the corresponding version

## Installation Steps

1. Go to the `examples/fast_kernel_launch_example` directory.

2. Install dependencies:

    ```sh
    python3 -m pip install -r requirements.txt
    ```

3. Build the wheel:

    ```sh
    # NPU_SOC_VERSION sets the compilation model. Atlas A2 products use "ascend910b" (default), Atlas A3 products use "ascend910_93", and Ascend 950PR&950DT products use "ascend950"
    export NPU_SOC_VERSION=ascend910b
    # -n: non-isolated build (uses existing environment)
    python3 -m build --wheel -n
    ```

    After the build completes, the artifact is in the `dist` folder of the current directory, with the name `ascend_ops-1.0.0-${python_version}-abi3-${arch}.whl`,
    `${python_version}` denotes the python version in the current environment (python3.8.3 is cp38), and `${arch}` denotes the CPU architecture.

4. Install the wheel package:

    ```sh
    python3 -m pip install dist/*.whl --force-reinstall --no-deps
    ```

5. (Optional) Before rebuilding, it is recommended to run the following command first to clean the compilation cache

   ```sh
    python setup.py clean
    ```

## Quick Start

After installation, you can use NPU operators just like ordinary PyTorch operations. The add operator invocation is used as an example.

```python
import torch
import torch_npu
import ascend_ops  # The built Python package

# Initialize data on NPU
x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()

# Call the custom NPU operator
npu_result = torch.ops.ascend_ops.add(x, y)  # PyTorch Custom Operator Dispatch mechanism: torch.ops.<library_name>.<operator_name>

# Verify against CPU ATen implementation
cpu_x = x.cpu()
cpu_y = y.cpu()
cpu_result = cpu_x + cpu_y

assert torch.allclose(cpu_result, npu_result.cpu(), rtol=1e-6)
print("Verification successful!")
```

## Developer Guide: Adding a New Operator

To implement a new operator (such as `add`), you only need to provide a C++ implementation.

1. First, create a folder named after the operator `add` under the csrc directory. Inside this folder, create a subfolder named after the soc you want to develop for. Atlas A2 products use `ascend910b`, Atlas A3 products use `ascend910_93`, and Ascend 950PR&950DT products use `ascend950`.

2. Create a new `CMakeLists.txt` in the soc directory

    ```bash
    add_sources("--npu-arch=dav-2201")
    ```

    Here, `dav-2201` is the compilation parameter corresponding to the ascend910b chip. For how to obtain it, refer to [NpuArch description and usage guide](https://gitcode.com/cann/ops-math/wiki/NpuArch%E8%AF%B4%E6%98%8E%E5%92%8C%E4%BD%BF%E7%94%A8%E6%8C%87%E5%AF%BC.md).

3. Create a new `add.cpp` in the soc directory (the operator name is recommended as the file name). This file contains all modules required to develop an AI Core operator.
    - Operator schema registration
    - Operator Meta function implementation and registration
    - Operator kernel implementation (Ascend C)
    - Operator NPU invocation implementation and registration

    ```cpp
    #include <ATen/Operators.h>
    #include <torch/all.h>
    #include <torch/library.h>
    #include "torch_npu/csrc/core/npu/NPUStream.h"
    #include "torch_npu/csrc/framework/OpCommand.h"
    #include "kernel_operator.h"
    #include "platform/platform_ascendc.h"
    #include <type_traits>

    namespace ascend_ops {  // The current project is one namespace
    namespace Add {         // Each operator is recommended to have its own namespace to prevent global variable pollution

    /**
     * Registers the operator schema with the PyTorch framework
     * The framework knows that such an operator exists
     */
    // Register the operator's schema
    TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
    {
        m.def("add(Tensor x, Tensor y) -> Tensor");
    }

    /**
     * Implements the Meta function of the operator, that is, InferShape+InferDtype
     * Infers from the inputs what the output of the operator looks like and how much space it needs. The operator is not actually computed.
     */
    // Meta function implementation of Add
    torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
    {
        TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
        auto z = torch::empty_like(x);
        return z;
    }

    /**
     * Registers the Meta function of the operator with the framework
     * The framework can call this Meta function to know how much space is needed before the operator is actually executed
     * This later enables graph acceleration such as torch.compile/AutoGrad/AclGraph
     */
    // Register the Meta implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
    {
        m.impl("add", add_meta);
    }

    /**
     * Implements the NPU operator kernel using AscendC APIs, written for the current soc
     */
    template <typename T>
    __global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        // kernel implementation
    }

    /**
     * Implements the operator invocation interface
     * In this interface, the NPU kernel invocation must be completed
     * 1. Compute the number/Shape/Dtype of the output tensors (implemented by calling the Meta function or directly)
     * 2. Compute the Tiling: determine how to split the computation based on the Shape
     * 3. Call the NPU kernel
     *
     */
    torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
    {
        // OptionalDeviceGuard ensures that subsequent operations run in the correct device context
        // It records the current device state and automatically restores it after the scoped code finishes
        const c10::OptionalDeviceGuard guard(x.device());
        auto z = add_meta(x, y);
        auto stream = c10_npu::getCurrentNPUStream().stream(false);
        int64_t totalLength, numBlocks, blockLength, tileSize;
        totalLength = x.numel();
        std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(totalLength);
        auto x_ptr = (GM_ADDR)x.data_ptr();
        auto y_ptr = (GM_ADDR)y.data_ptr();
        auto z_ptr = (GM_ADDR)z.data_ptr();
        auto acl_call = [=]() -> int {
            AT_DISPATCH_SWITCH(
                x.scalar_type(), "add_npu",
                // Calls different NPU kernels for different data types
                AT_DISPATCH_CASE(torch::kFloat32, [&] {
                    using scalar_t = float;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr,     totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kFloat16, [&] {
                    using scalar_t = half;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr,     totalLength, blockLength, tileSize);
                })
                AT_DISPATCH_CASE(torch::kInt32, [&] {
                    using scalar_t = int32_t;
                    add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr,     totalLength, blockLength, tileSize);
                })
            );
            return 0;
        };
        // You need to use the RunOpApi/RunOpApiV2 interface for the call, ensuring the timing is consistent with TorchNPU invoking the aclnn interface.
        at_npu::native::OpCommand::RunOpApi("Add", acl_call);
        return z;
    }

    /**
     * Registers the invocation function of the operator with the framework, and Device is PrivateUse1
     * The framework knows that when all inputs are on the NPU Device, dispatch goes to this operator implementation
     */
    // Register the NPU implementation
    TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
    {
        m.impl("add", add_npu);
    }

    }  // namespace Add
    }  // namespace ascend_ops

    ```

4. Rebuild and install the wheel package by referring to the [Installation Steps](#installation-steps) section.
5. To test the operator API based on pytest, refer to the implementation of [test_add.py](tests/add/test_add.py).
