# torch_extension Development Guidelines

This document defines the directory organization, naming, implementation of each layer, and documentation conventions for adding or modifying operator APIs in `cann_ops_nn` (torch_extension). Before you add an operator, read this guideline in full and use existing operators (such as `swiglu_group`) as templates.

`cann_ops_nn` uses PyTorch JIT (`torch.utils.cpp_extension.load`) to compile the C++ kernel wrapper on the first call. It bridges PyTorch function interfaces to the aclnn APIs of CANN, and supports the torchair graph mode through the GE Converter. A complete operator API usually consists of four parts: the Python frontend, the C++ backend, the torchair graph-mode Converter, and the documentation.

## 1. Operator API Directory Organization

Operators in the nn repository use a **distributed directory structure**: the torch_extension files of each operator are placed in the corresponding operator directory under the repository root. `setup.py` automatically collects and stages them into the whl package at build time. Take the operator `${op_api}` (in the `${category}` category) as an example:

```
├── ${category}/
│   └── ${op_api}/
│       └── torch_extension/                    # torch_extension directory of the operator
│           ├── __init__.py                     # exports the operator API and the Converter
│           ├── ${op_api}.py                    # Python frontend implementation (OpBuilder, schema, meta, public functions)
│           ├── graph_convert_${op_api}.py      # torchair graph-mode (GE Converter) implementation
│           └── csrc/
│               └── ${op_api}.cpp               # C++ kernel wrapper, calls the aclnn API
├── torch_extension/
│   ├── cann_ops_nn/                            # whl package installation directory (staged automatically at build time)
│   │   ├── __init__.py                         # package root entry
│   │   ├── op_builder/
│   │   │   └── builder.py                      # OpBuilder base class, JIT compilation, schema/meta registration (usually no modification needed)
│   │   ├── common/
│   │   │   └── aclnn_common.h                  # common capabilities such as the ACLNN_CMD macro and type conversion (usually no modification needed)
│   │   ├── csrc/                               # C++ source staging directory (collected automatically at build time)
│   │   │   └── ${op_api}.cpp
│   │   └── ops/
│   │       ├── __init__.py                     # automatic operator discovery and loading
│   │       └── ${category}/
│   │           ├── __init__.py                 # category entry (copied from the source code)
│   │           └── ${op_api}/
│   │               ├── __init__.py             # operator entry (imports the API and the Converter)
│   │               ├── ${op_api}.py
│   │               └── graph_convert_${op_api}.py
│   ├── setup.py                                # wheel packaging configuration (automatically collects distributed operators)
│   └── build.sh                                # build script
└── build_out/                                  # whl artifact output directory
```

Standard steps to add a new operator API (using `swiglu_group` as an example):

1. Implement the C++ kernel wrapper in `${category}/${op_api}/torch_extension/csrc/${op_api}.cpp` and call `ACLNN_CMD` to launch the aclnn API;
2. Write an `OpBuilder` subclass in `${category}/${op_api}/torch_extension/${op_api}.py` (define `sources`/`schema`/`register_meta`), register the dispatcher implementation, and provide the public Python function;
3. Write the graph-mode Converter in `${category}/${op_api}/torch_extension/graph_convert_${op_api}.py` (if graph-mode support is needed);
4. Export the public API and the Converter in `${category}/${op_api}/torch_extension/__init__.py`;
5. Export the new operator in `${category}/__init__.py` (so that the source `__init__.py` is copied into the whl at build time);
6. Add the operator documentation in `docs/zh/${op_api}.md` (if any).

> Place new files in `torch_extension/` under the operator directory. `setup.py` collects them automatically at build time. The import path always uses `cann_ops_nn` as the root.

## 2. Naming Conventions

### 2.1 API Naming

Naming an operator from schema registration to public export involves multiple levels. The names must stay consistent, and each level serves its own purpose. **Public API names and operator names never carry the `npu_` prefix**. Use the lowercase snake_case name that reflects the operator semantics (such as `swiglu_group`):

| Level | Naming convention | Example |
| --- | --- | --- |
| Library name (DEF domain) | Fixed to `cann_ops_nn` | `Library("cann_ops_nn", "DEF")` |
| schema operator name / aten registration name | Lowercase snake_case name of the operator semantics, without the `npu_` prefix | `swiglu_group` |
| C++ wrapper function name | Same as the schema operator name, placed in `namespace cann_ops_nn::category` | `cann_ops_nn::activation::swiglu_group` |
| `PYBIND11_MODULE` export name | Same as the schema operator name | `m.def("swiglu_group", &swiglu_group, "SwigluGroup on NPU")` |
| Meta implementation function name | schema operator name + the `_meta` suffix | `swiglu_group_meta` |
| PrivateUse1 dispatcher function name | Underscore prefix + the schema operator name | `_swiglu_group` |
| OpBuilder subclass name | PascalCase of the operator name + the `OpBuilder` suffix | `SwigluGroupOpBuilder` |
| Public Python API name | Function name that users call directly, reflecting usage semantics, without the `npu_` prefix | `swiglu_group` |
| Graph-mode GE op function name | PascalCase that matches the GE operator `op_type` | `SwigluGroup` |
| Graph-mode Converter function name | `convert_` + the schema operator name | `convert_swiglu_group` |

Key naming points:

- **No `npu_` prefix**: the public operator name and API names uniformly use the operator semantic name (lowercase snake_case), without backend prefixes such as `npu_`. The schema name, the C++ function name, and the pybind export name must match this name exactly; otherwise the JIT compilation artifact cannot be called correctly.
- **API names reflect semantics**: the public function name should stay close to the business semantics. Operators that purely pass through the aclnn API (such as `swiglu_group`) and APIs that wrap extra logic such as struct construction and parameter arrangement all use semantic naming.
- **Independent aclnn API names**: the underlying aclnn APIs keep the CANN naming (PascalCase, such as `aclnnSwigluGroup`) and stay decoupled from the public operator name. Call them inside the C++ wrapper through `ACLNN_CMD(aclnnSwigluGroup, ...)`.
- **Version suffixes**: different iteration versions of the same operator use suffixes such as `_v2` and `_v3`. The schema name, file names, and Converter names must carry the version suffix consistently (such as `rms_norm_v2` and `graph_convert_rms_norm_v2.py`).
- **Auxiliary/utility APIs**: utility functions that accompany a main operator use verb-object semantic naming, such as `get_swiglu_group_workspace_size`.

### 2.2 File Naming

- Use **lowercase snake_case** uniformly. Connect words with `_`; uppercase letters, camelCase, and hyphens are forbidden.
- The files of the same operator across layers **keep the same main name**. Only directories and prefixes distinguish their roles:
  - Python frontend: `${op_api}.py`, such as `swiglu_group.py`;
  - C++ backend: `csrc/${op_api}.cpp`, with the main name matching the Python frontend, such as `swiglu_group.cpp`;
  - Graph mode: `graph_convert_${op_api}.py`, always with the `graph_convert_` prefix, such as `graph_convert_swiglu_group.py`;
  - Operator entry: `__init__.py`, which exports the operator API and the Converter.
- The file main name should match the operator semantics that the file mainly exports. Versioned operator file names must carry the version suffix (such as `rms_norm_v2.cpp`).
- Place common header files under `common/` and name them by capability domain (such as `aclnn_common.h`).

### 2.3 Identifier Naming

#### Python identifiers

- **Functions/variables/parameters**: lowercase snake_case, such as `clamp_limit`, `group_index`, and `dst_type`.
- **Class names**: PascalCase, such as `OpBuilder` and `SwigluGroupOpBuilder`.
- **Module-level constants**: UPPER_SNAKE_CASE, such as `BLOCK_FP8_QUANT_MODE` and `MX_BLOCK_SIZE`.
- **Module-private symbols**: mark them with a single underscore `_` prefix, such as `_swiglu_group` and `_TORCHAIR_AVAILABLE`.
- **Type annotations**: public APIs and key internal functions should carry type annotations (`from typing import Optional, Tuple, List`). Use `Optional[...]` uniformly for optional parameters. For example:
  ```python
  def swiglu_group(
      x: torch.Tensor,
      *,
      weight: Optional[torch.Tensor] = None,
      group_index: Optional[torch.Tensor] = None,
      clamp_limit: float = -1.0,
  ) -> torch.Tensor:
  ```
- **Parameter naming consistency**: equivalent parameters of the same operator across schema, meta, dispatcher, public functions, and Converter should use the same name (such as `clamp_limit` and `group_index`), to avoid inconsistent spellings across layers.

#### C++ identifiers

- **Functions/local variables/parameters**: lowercase snake_case, such as `clamp_limit`, `group_index`, and `k_split_factor`.
- **Namespaces**: place operator implementations uniformly in `namespace cann_ops_nn::${category}` (such as `namespace cann_ops_nn::activation`).
- **Constants**: `const`/`constexpr` constants use PascalCase with a `k` prefix or UPPER_SNAKE_CASE, such as `constexpr int64_t kSplitFactor = 2;`.
- **Type aliases/structs**: PascalCase, such as `TensorWrapper` and `TensorListWrapper`.
- **Input parameter type conventions:**
  - Required tensors use `const at::Tensor&`;
  - Optional tensors use `const c10::optional<at::Tensor>&`;
  - Integer attributes use `int64_t`; optional integer attributes use `c10::optional<int64_t>`; floating-point attributes use `double`; string attributes use `std::string`.

#### Schema identifiers (operator signature)

- Parameter names use lowercase snake_case, consistent with the Python/C++ layers.
- Use `*` to separate positional parameters from keyword parameters: parameters before `*` are required positional parameters, and parameters after it are optional keyword parameters (with default values).
- Mark optional parameters with `?` and provide default values, such as `Tensor? weight=None` and `float clamp_limit=-1.0`.
- Express multiple outputs as a tuple, such as `-> (Tensor, Tensor)`.

  The schema of `swiglu_group` as an example:

  ```
  swiglu_group(Tensor x, *, Tensor? weight=None, Tensor? group_index=None, float clamp_limit=-1.0) -> Tensor
  ```

## 3. Implementation Guidelines for Each Layer

### 3.1 C++ Backend (`csrc/${op_api}.cpp`)

The C++ backend bridges PyTorch tensors to the aclnn C API. Key points:

1. Include `#include <torch/extension.h>` and `#include "aclnn_common.h"` at the top of the file. Place the implementation in `namespace cann_ops_nn` and `namespace ${category}`.
2. The function signature strictly matches the schema: choose required/optional parameter types according to [2.3 C++ input parameter type conventions](#c-identifiers).
3. **Input validation**: use `TORCH_CHECK(cond, msg...)` to validate shape, dtype, dimensions, value ranges, and so on. Error messages must be readable and contain the actual values. For example:
   ```cpp
   TORCH_CHECK(torch_npu::utils::is_npu(tensor), "tensor must be on NPU device");
   TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
   TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
               "x dtype must be float16 or bfloat16, but got ", x.scalar_type());
   ```
4. **Manual output tensor allocation**: allocate outputs with `at::empty(...)` based on the shape/dtype derived by the meta. Specify the dtype through `x.options().dtype(...)`.
5. **Launching the kernel**: use the `ACLNN_CMD(aclnn API name, input..., output...)` macro to call the aclnn API (such as `ACLNN_CMD(aclnnSwigluGroup, ...)`). The input order must match the aclnn API definition. The macro automatically completes type conversion, workspace allocation, stream dispatch, and resource release.
6. **Export binding**: bind the C++ function through `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` as a Python API with the same name as the schema.
7. Express magic numbers (such as the dimension count `2` and default dtype enums) as named `constexpr` constants; avoid bare literals.

   Typical skeleton of the C++ wrapper (see `swiglu_group.cpp`):

   ```cpp
   #include <torch/extension.h>
   #include "aclnn_common.h"

   namespace cann_ops_nn {
   namespace activation {
   namespace {
   constexpr int64_t kSplitFactor = 2;

   void CheckNpuTensor(const at::Tensor& tensor, const char* name)
   {
       TORCH_CHECK(tensor.defined(), name, " must be defined");
       TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
   }
   } // namespace

   at::Tensor swiglu_group(const at::Tensor& x, const c10::optional<at::Tensor>& weight,
                           const c10::optional<at::Tensor>& group_index, double clamp_limit)
   {
       CheckNpuTensor(x, "x");
       TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

       c10::SmallVector<int64_t, N> y_shape;
       // ... derive the output shape ...

       at::Tensor y = at::empty(y_shape, x.options());

       ACLNN_CMD(aclnnSwigluGroup, x, weight, group_index, clamp_limit, y);
       return y;
   }

   } // namespace activation
   } // namespace cann_ops_nn

   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
   {
       m.def("swiglu_group", &cann_ops_nn::activation::swiglu_group, "SwigluGroup on NPU");
   }
   ```

### 3.2 Python Frontend (`${op_api}.py`)

The Python frontend manages JIT compilation, schema/meta registration, and public API encapsulation:

1. **OpBuilder subclass**: inherit `OpBuilder`, pass the operator name through `super().__init__("<schema operator name>")` in `__init__`, and implement three abstract methods:
   - `sources()`: returns the list of C++ source file paths relative to the `cann_ops_nn` package root. Obtain them uniformly through `self.resolve_source("${op_api}.cpp")` (such as `[self.resolve_source("swiglu_group.cpp")]`);
   - `schema()`: returns the operator schema string (see [2.3 Schema identifiers](#schema-identifiers-operator-signature));
   - `register_meta()`: registers the Meta implementation with `@impl(get_as_library(), self.name, "Meta")`. It only derives shape/dtype and does not touch real NPU computation (required for FakeTensor/graph mode).
2. **Instantiation and initialization**: instantiate the builder at module load and call `_ensure_initialized()` to register the schema and meta:
   ```python
   builder = SwigluGroupOpBuilder()
   builder._ensure_initialized()
   ```
3. **PrivateUse1 dispatcher**: register the NPU backend implementation with `@impl(get_as_library(), builder.name, "PrivateUse1")`. The function body passes through to the compilation artifact `op_module.<operator name>(...)`. `PrivateUse1` is the dispatch key that PyTorch reserves for custom NPU backends.
4. **Public API**: provide user-facing functions that handle parameter arrangement, default value processing, and so on, and finally call `torch.ops.cann_ops_nn.<operator name>(...)` for dispatcher scheduling.
5. **Public APIs must carry docstrings**: every exported public API needs a docstring that covers at least the function description, the meaning/shape/dtype/value range of each parameter, and the return value description.
6. The parameter order and default values of the Meta implementation, the dispatcher, and the public function must match the schema.

   Typical skeleton of the Python frontend (see `swiglu_group.py`):

   ```python
   import torch
   from torch.library import impl
   from cann_ops_nn.op_builder import OpBuilder, get_as_library


   class SwigluGroupOpBuilder(OpBuilder):
       def __init__(self):
           super().__init__("swiglu_group")

        def sources(self):
            return [self.resolve_source("swiglu_group.cpp")]

       def schema(self):
           return "swiglu_group(Tensor x, *, Tensor? weight=None, Tensor? group_index=None, float clamp_limit=-1.0) -> Tensor"

       def register_meta(self):
           @impl(get_as_library(), self.name, "Meta")
           def swiglu_group_meta(x, *, weight=None, group_index=None, clamp_limit=-1.0):
               last_dim = x.size(x.dim() - 1)
               shape = list(x.shape)
               shape[-1] = last_dim // 2
               return x.new_empty(shape)


   builder = SwigluGroupOpBuilder()
   builder._ensure_initialized()


   @impl(get_as_library(), builder.name, "PrivateUse1")
   def swiglu_group(x, *, weight=None, group_index=None, clamp_limit=-1.0):
       op_module = builder.load()
       return op_module.swiglu_group(x, weight, group_index, clamp_limit)
   ```

### 3.3 Graph-mode Converter (`graph_convert_${op_api}.py`)

The Converter converts aten operators into GE nodes in torchair graph mode (GE):

1. **Optional dependency protection**: wrap all torchair-related imports uniformly in `try/except ImportError`, controlled by the `_TORCHAIR_AVAILABLE` flag, to avoid import failures in environments without torchair.
2. **GE op functions**: define a PascalCase function with the same name as `op_type` (such as `SwigluGroup`). Describe the IR definition of `REG_OP` (INPUT/OPTIONAL_INPUT/OUTPUT/ATTR, and so on) in the docstring, organize `inputs`/`attrs`/`outputs`, and then call `ge_op(...)`. Declare the IR through chained `IrDef(...)` calls.
3. **Converter registration**: decorate the `convert_<operator name>` function with `@register_fx_node_ge_converter(torch.ops.cann_ops_nn.<operator name>.default)`. Its parameter order exactly matches the schema, and the function body calls the GE op function above.
4. Export the Converter (such as `convert_swiglu_group`) in the `__init__.py` of the operator, so that the registration logic executes.

### 3.4 Public Export (`__init__.py`)

Public export has two levels:

1. **Operator `__init__.py`**: explicitly import and export the public API and Converter of every new operator here. The import triggers schema/meta/converter registration:
   ```python
   __all__ = ["swiglu_group", "convert_swiglu_group"]

   from .swiglu_group import swiglu_group
   from .graph_convert_swiglu_group import convert_swiglu_group
   ```
2. **Category `__init__.py`** (such as `activation/__init__.py`): export operators in the category directory; the build copies them into the whl:
   ```python
   __all__ = ["swiglu_group", "swiglu_group_quant"]

   from .swiglu_group import swiglu_group
   from .swiglu_group_quant import swiglu_group_quant
   ```

   `cann_ops_nn/__init__.py` (the package root layer) triggers automatic discovery and loading of all operators through `from . import ops`. Users can access them directly through `cann_ops_nn.<API name>`.

## 4. Documentation Guidelines (`docs/zh/${op_api}.md`)

Each public operator API needs a matching Chinese document (if any). The suggested section order aligns with existing operator documents:

1. **Title**: the operator name (escape special characters such as `_` as `\_`).
2. **Applicable Products**: a table lists the supported product forms and whether they are supported.
3. **Function**: an API function overview plus the calculation formula (use LaTeX for math expressions), and explain the mapping between each symbol and its parameter.
4. **Function Prototype**: a code block gives the complete function signature, including default values and `*` separation.
5. **Parameters**: describe each parameter one by one: required/optional, semantics, shape, dtype, and value range/constraints.
6. **Output Description**: describe each output one by one: shape, dtype, and so on.
7. **Constraints**: list parameter consistency constraints, shape/value range constraints, and so on by category.
8. **Invocation Example**: provide a complete runnable example in single-operator mode.

## 5. General Coding Constraints

- **License header**: all new source files (`.py`/`.cpp`/`.h`) must contain the Huawei copyright header and the CANN Open Software License Agreement Version 2.0 license header, with the current year. Use `#` comments for Python/scripts, and `//` or `/* */` for C++.
- **API comments**: public API interfaces must carry docstrings (function, parameters, return values). Key C++ wrapper logic (validation, aclnn calls) should also carry brief comments.
- **Validate parameters up front**: use `TORCH_CHECK(cond, msg...)` on the C++ side. Error messages need to contain the actual variable values to make debugging easier.
- **Avoid magic numbers**: express dimension counts, dtype enum values, and so on as named `constexpr` constants, and explain the enum meanings in documents/comments.
- **Reuse common capabilities**: prefer reusing the common headers under `common/` for type conversion, `ACLNN_CMD`, and so on; do not reimplement them in each operator.
- **Consistency self-check**: before submission, confirm that the operator names, parameter names, parameter order, and default values stay consistent across the seven places: schema, C++ wrapper, Meta, dispatcher, public function, Converter, and documentation.
