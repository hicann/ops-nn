# cann_ops_nn

English | [简体中文](./README.md)

A high-performance operator extension library for the Ascend NPU. It bridges PyTorch interfaces with the ACLNN operator library through JIT compilation.

## Build and Installation

### Prerequisites

- Operating system: Linux
- Python: 3.8+
- Compiler: GCC 9.4.0+
- Framework: PyTorch>=2.6.0 and torch_npu (the torch_npu version must match the PyTorch version)
- Toolkit: Ascend CANN Toolkit

### Building the Wheel Package

Two build methods are supported: building through `build.sh` and building directly with `python3 -m build`.

#### Method 1: Building through build.sh (recommended)

```sh
# Build the full package (contains all operators)
bash build.sh --torch_extension

# Build a single-operator package (contains only the specified operator)
bash build.sh --torch_extension --ops=swiglu_group --vendor_name=custom

# Build a multi-operator package
bash build.sh --torch_extension --ops=swiglu_group,swiglu_group_quant --vendor_name=custom

# Build an experimental operator package (contains only operators under the experimental directory)
bash build.sh --torch_extension --experimental
```

After the build completes, the wheel package is automatically copied to the `build_out/` directory.

**Parameter description:**

| Parameter | Required | Description |
| --- | --- | --- |
| `--torch_extension` | Yes | Builds only the torch_extension wheel package and does not run cmake compilation |
| `--ops=op1,op2,...` | No | Specifies the names of operators to compile (separated by commas). If not specified, all operators are compiled |
| `--vendor_name=name` | No | Specifies the sub-package name suffix for sub-package naming and isolation. This parameter is invalid when `--ops` is not specified. The default value is `custom` |
| `--experimental` | No | Compiles only operators under the experimental directory. If not specified, the experimental directory is skipped |

**Package naming rules:**

| Scenario | Condition | Package name | Installation directory |
| --- | --- | --- | --- |
| Full package | `--ops` not specified | `cann_ops_nn` | `cann_ops_nn/` |
| Single-operator/multi-operator package | `--ops` specified, `--vendor_name` optional | `cann_ops_nn_<vendor>` | `cann_ops_nn_<vendor>/` |

> **Naming logic:** If `--ops` is not specified, the full package is built and the package name is fixed to `cann_ops_nn`. If `--ops` is specified, a sub-package is built and the package name is `cann_ops_nn_` concatenated with the value of `--vendor_name` (`custom` by default if not specified). The full package and sub-packages have physically isolated installation directories and can coexist.

#### Method 2: Building directly with python3 -m build

```sh
# Install dependencies
python3 -m pip install -r requirements.txt

# Build the full package
python3 -m build --wheel -n

# Build a single-operator package
TORCH_EXTENSION_OPS=swiglu_group TORCH_EXTENSION_VENDOR=custom python3 -m build --wheel -n
```

### Installation

```sh
# Install the full package
python3 -m pip install dist/cann_ops_nn-*.whl

# Install a single-operator package
python3 -m pip install dist/cann_ops_nn_custom-*.whl
```

### Coexistence Mechanism of the Full Package and Sub-packages

The full package and single-operator packages can be installed at the same time without conflicts:

- The **full package** is installed to the `cann_ops_nn/` directory and contains all operators.
- A **single-operator package** is installed to the `cann_ops_nn_<vendor>/` directory, physically isolated from the full package.
- Single-operator packages register operators through the **entry point** mechanism and have a higher priority than the full package. When a user calls `cann_ops_nn.swiglu_group(x)`, if the sub-package is installed, the operator implementation of the sub-package is used.
- After a single-operator package is uninstalled, the operator with the same name in the full package takes over automatically.

```sh
# Install the full package
pip install cann_ops_nn-*.whl

# Install a single-operator package (overriding the operator with the same name in the full package)
pip install cann_ops_nn_custom-*.whl

# Uninstall the single-operator package (the operator of the full package is restored automatically)
pip uninstall cann-ops-nn-custom
```

## Directory Structure

```
├── torch_extension
│   ├── build.sh                              # Build script (supports the --torch_extension parameter)
│   ├── setup.py                              # Wheel packaging configuration (automatically collects distributed operators)
│   ├── requirements.txt
│   └── cann_ops_nn                           # Full package installation directory (common framework part)
│       ├── __init__.py                       # Package entry, from .ops import *
│       ├── op_builder/
│       │   └── builder.py                    # OpBuilder base class, JIT compilation management
│       ├── common/
│       │   └── aclnn_common.h                # Common capabilities such as the ACLNN_CMD macro
│       ├── csrc/
│       │   └── extension.cpp                 # Placeholder module (satisfies PyTorch extension requirements)
│       ├── ops/
│       │   └── __init__.py                   # Automatic operator discovery + entry point loading (populated at build time)
│       └── docs/
│           └── torch_extension_guidelines.md # Development guidelines
├── <category>/<op>/torch_extension/          # Operator development directory (under the repository root)
│   ├── <op>.py                               # Python frontend (OpBuilder + schema + meta)
│   ├── __init__.py
│   ├── graph_convert_<op>.py                 # (Optional) graph fallback conversion
│   └── csrc/<op>.cpp                         # C++ kernel wrapper
└── build_out/                                # Output directory of build.sh
```

> **Collection at build time:** During the packaging stage, `setup.py` automatically traverses all `<category>/<op>/torch_extension/` directories in the repository and collects `<op>.py` and `csrc/<op>.cpp` into the wheel package directories `cann_ops_nn/ops/<category>/<op>/` and `cann_ops_nn/csrc/`. Developers do not need to manually create operator files under `cann_ops_nn/`.

## Quick Start

```python
import torch
import torch_npu
import cann_ops_nn

# Initialize an NPU tensor
x = torch.randn(16, 32, dtype=torch.float16).npu()

# Call the operator (the two methods are equivalent)
result = cann_ops_nn.swiglu_group(x)
result = cann_ops_nn.ops.swiglu_group(x)

print(result.shape)  # torch.Size([16, 16])
```



## Developer Guide: Adding a New Operator

This section uses adding a new operator `new_operator` (category `my_cat`) as an example. Create a `torch_extension/` subdirectory under the operator directory `<my_cat>/new_operator/`, containing the C++ binding and the Python builder.

### 1. C++ Backend (`<my_cat>/new_operator/torch_extension/csrc/new_operator.cpp`)

This file bridges PyTorch tensors to the ACLNN C-API.

```cpp
#include <torch/extension.h>
#include "aclnnop/aclnn_new_operator.h"
#include "aclnn_common.h"  // Resolved through -I cann_ops_nn/common

namespace cann_ops_nn {
namespace my_cat {

at::Tensor new_operator(
    const at::Tensor& input1,
    const at::Tensor& input2,
    int64_t param1,
    const std::string& param2)
{
    // Device check
    TORCH_CHECK(input1.device().type() == at::kPrivateUse1, "input1 must be on NPU device");
    TORCH_CHECK(input2.device().type() == at::kPrivateUse1, "input2 must be on NPU device");

    // Output creation
    at::Tensor out = at::empty(output_shape,
        at::TensorOptions().dtype(output_dtype).device(at::kPrivateUse1));

    // ACLNN invocation (the macro automatically handles type conversion and workspace internally)
    ACLNN_CMD(aclnnNewOperator, input1, input2, out);

    return out;
}

}  // namespace my_cat
}  // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("new_operator", &cann_ops_nn::my_cat::new_operator,
          "NewOperator on NPU");
}
```

### 2. Python Frontend (`<my_cat>/new_operator/torch_extension/new_operator.py`)

This file manages the JIT compilation logic and registers the operator to the PyTorch Dispatcher.

```python
import torch
import torch_npu
from torch.library import impl
from cann_ops_nn.op_builder import OpBuilder, get_as_library

class NewOperatorOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("new_operator")

    def sources(self):
        """C++ source file paths."""
        return [self.resolve_source("new_operator.cpp")]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "new_operator(Tensor input1, Tensor input2, int param1=0, str param2=\"\") -> Tensor"

    def register_meta(self):
        """
        Registers the Meta implementation (shape/type inference).
        This is critical for Autograd and FakeTensor support.
        """
        @impl(get_as_library(), self.name, "Meta")
        def new_operator_meta(input1, input2, param1=0, param2=""):
            return torch.empty_like(input1)

# Instantiate the builder
builder = NewOperatorOpBuilder()
builder._ensure_initialized()

@impl(get_as_library(), builder.name, "PrivateUse1")
def new_operator(input1, input2, param1=0, param2=""):
    """
    NPU implementation of the Dispatcher.
    'PrivateUse1' is the dispatch key of the custom NPU backend.
    """
    op_module = builder.load()  # Compile/load the .so file
    return op_module.new_operator(input1, input2, param1, param2)
```

### Technical Description

| Component | Responsibility |
| --- | --- |
| **OpBuilder** | Uses `ninja` to handle the JIT compilation of C++ source code |
| **Meta dispatch** | Allows PyTorch to infer output shapes/types without running NPU code |
| **PrivateUse1** | Backend dispatch key used by PyTorch to route NPU-specific operations |
| **ACLNN_CMD macro** | Automatically handles `at::Tensor` → `aclTensor*` type conversion, workspace allocation/release, and stream scheduling |

### Deterministic Computation

After deterministic mode is enabled through `torch.use_deterministic_algorithms(True)`, all operators invoked through `ACLNN_CMD` automatically pass the deterministic flag to the CANN runtime, ensuring that the same input produces the same output:

```python
import torch
import cann_ops_nn

torch.use_deterministic_algorithms(True)
# Subsequent operator invocations use the deterministic implementation
```

Implementation principle: before each operator invocation, the `ACLNN_CMD` macro reads `at::globalContext().deterministicAlgorithms()`, writes the flag into the current device context through `aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, ...)`, and the operator selects the deterministic implementation path based on this flag.

### Operator Library Search Mechanism

At runtime, the JIT-compiled `.so` searches for operator symbols in the following order (`dlopen`/`dlsym`):

1. **`libcust_opapi.so`** (vendor single-operator package, optional): traverses the paths in `ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH` and loads `libcust_opapi.so`
2. **`libopapi_nn.so`** (operator library built and installed by the repository)

> The vendor package is optional. If it is not installed, `libopapi_nn.so` is used directly.

```sh
# Set the vendor package path (optional, generated automatically after the vendor package is installed)
source <vendor_path>/set_env.bash
# Or set it manually
export LD_LIBRARY_PATH=<vendor_path>/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=<vendor_path>:$ASCEND_CUSTOM_OPP_PATH
```
