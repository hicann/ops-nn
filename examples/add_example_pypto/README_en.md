# AddExamplePypto

English | [简体中文](./README.md)

The PyPTO version of `examples/add_example`: the function is likewise element-wise addition, and the difference is that the kernel is written with the PyPTO DSL
(`op_kernel/add_example_pypto.py`) instead of AscendC C++. This operator demonstrates how to integrate PyPTO operators in ops-nn.

## Compilation

```bash
source <ascend-toolkit>/set_env.sh
bash build.sh --pkg --ops=add_example_pypto --soc=ascend950
```

## Integration method

In the `CMakeLists.txt` of the operator, add two lines **before** `add_all_modules_sources` (or `add_modules_sources`):

```cmake
require_pypto_pro(add_example_pypto)
enable_pypto_kernel(add_example_pypto)

add_all_modules_sources(OPTYPE add_example_pypto ACLNNTYPE aclnn)
```

> The host directory of the sample operator is inferred automatically by `add_all_modules_sources` (`op_host/` is used when `op_api/` does not exist),
> and the UT sources under `tests/ut` are collected automatically, so separate files such as `op_host/CMakeLists.txt` and `tests/.../CMakeLists.txt` are not needed.

- `require_pypto_pro(<op_name>)`: when `pypto_pro` does not exist in the environment, this macro issues a warning and skips the operator, without failing the whole configure.
- `enable_pypto_kernel(<op_file>)`: `<op_file>` must be consistent with `op_kernel/<op_file>.py` and with
  `ExtendCfgInfo("opFile.value", ...)` in the operator definition.

## Build flow

`enable_pypto_kernel` calls `scripts/util/pypto_codegen.py` **at the configure stage** to perform codegen for the `.py` kernel:

| Artifact | Purpose |
| --- | --- |
| `<Tiling>_tiling.h` | Tiling struct, used directly by host-side tiling |
| `<TilingKey>_tilingkey.h` | `ASCENDC_TPL` tiling key declaration |
| `<op_file>_pypto_infer.cpp` | Self-contained source code used by the `.i` infer (produced at the build stage, not generated at the configure stage) |

The codegen directory of tiling is injected into the include path of the tiling target through `target_include_directories` (compatible with CMake 3.16),
and `*_tiling*.cpp` must **explicitly include** the generated headers (order: tilingkey first, then tiling):

```cpp
// Explicitly include the headers generated at the configure stage
#include "AddExamplePyptoTilingKey_tilingkey.h"
#include "AddExamplePyptoTiling_tiling.h"
// AddExamplePyptoTiling comes from op_kernel/add_example_pypto.py
AddExamplePyptoTiling* tilingData = context->GetTilingData<AddExamplePyptoTiling>();
```

The host and the kernel therefore share the same tiling layout and tiling key bit allocation, and no handwritten copies are needed.

**At the build stage**, the wrapper generated for this operator (`<build>/tbe/dynamic/<op_file>.py`) calls `pypto_compile_op`
instead of `compile_op`, which performs codegen by tiling key and reuses the AscendC backend to build the kernel binary.

## Chip and arch

The codegen arch of PyPTO is derived automatically by the platform (`a2`/`a3`/`a5`). This operator registers only `ascend950`(a5), consistent with the arch
automatically derived by the current toolkit. When the kernel binary is compiled, cmake exports `PYPTO_JIT_ARCH` according to the soc being built
(`ascend910b`/`ascend910_93` → `a3`, `ascend950` → `a5`), preventing `pypto_compile_op` from falling back to the default `a5`
and generating instructions not supported by the target chip.

## Invocation Description

The aclnn interface of AddExamplePypto is generated automatically from the operator definition, and **no handwritten source code is needed in the `op_host/op_api` directories**; the generated header is
`aclnn_add_example_pypto.h`, and the two-stage interfaces are `aclnnAddExamplePyptoGetWorkspaceSize` and `aclnnAddExamplePypto`.

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn invocation</td>
    <td><a href="./examples/test_aclnn_add_example_pypto.cpp">test_aclnn_add_example_pypto</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">operator invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>

Compile, install, and run the sample:

```bash
# Compile (only ascend950 is supported)
bash build.sh --pkg --soc=ascend950 --ops=add_example_pypto -j16

# Install
./build_out/cann-ops-nn-*linux*.run

# Run the sample (the run method is the same as that for standard operators)
bash build.sh --run_example add_example_pypto eager cust --vendor_name=custom --soc=ascend950
```

Expected output: when `first input` is 1 and `second input` is 2, `result` is 3, indicating that the operator has been successfully deployed and executed correctly.
