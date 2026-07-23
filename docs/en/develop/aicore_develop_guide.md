# AI Core Operator Development Guide

## Overview

> **Note:**
>
> 1. For basic concepts involved in operator development, such as Tiling, Kernel, and hardware architecture, refer to [Ascend C Operator Development](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html). For the interfaces involved, refer to [Ascend C Operator Development Interface](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0003.html) and [Basic Data Structures and Interfaces](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/basicdataapi/atlasopapi_07_00001.html).
> 2. AI Core operators are developed using the Ascend C language and run on the AI Core hardware unit. AI CPU operators are developed using the C++ language and run on the AI CPU hardware unit. If you want to contribute AI CPU operators, refer to the [AI CPU Operator Development Guide](./aicpu_develop_guide.md).
> 3. For operators contributed based on the [Ascend/samples](https://gitee.com/ascend/samples/tree/master) repository, refer to [Appendix > Operator Project Migration](#operator-project-migration) to complete the migration of existing operators to this project.
> 4. build.sh: The commands involved in operator development can be viewed through `bash build.sh --help`. For function parameter descriptions, refer to [build Parameter Description](../context/build.md).

This development guide uses the `AddExample` operator as an example to introduce the new operator development process and the deliverables involved. For complete sample code, visit the project `examples` directory.

1. [Project Creation](#project-creation): Before developing an operator, complete the environment deployment and create the operator directory for subsequent operator compilation and deployment.

2. [Operator Definition](#operator-definition): Determine the operator functionality and prototype definition.

3. [Tiling Implementation](#tiling-implementation): Implement the Host-side operator Tiling function.

4. [Kernel Implementation](#kernel-implementation): Implement the Device-side operator kernel function.

5. [aclnn Adaptation](#aclnn-adaptation): Custom operators are recommended to use the aclnn interface for invocation, which requires completing binary publishing in advance. **If you use graph mode to invoke the operator**, refer to the [Graph Mode Adaptation Guide](./graph_develop_guide.md).

6. [Compilation and Deployment](#compilation-and-deployment): Complete the compilation and installation of the custom operator through the project compilation script.

7. [Operator Verification](#operator-verification): Verify the custom operator functionality through common operator invocation methods.

## Project Creation

**1. Environment Deployment**

Before developing an operator, complete the basic environment setup by following [Environment Deployment](../context/quick_install.md).

**2. Directory Creation**

Directory creation is an important step in operator development, providing a unified directory structure and file organization for subsequent code writing, compilation, and debugging.

You can quickly create the operator directory through `build.sh`. Enter the project root directory and execute the following command:

```bash
# Create the specified operator directory, for example: bash build.sh --genop=activation/op_example
# ${op_class} represents the operator type, such as the activation class.
# ${op_name} represents the lowercase underscore form of the operator name. For example, the `AddExample` operator corresponds to add_example. New operators must not have the same name as existing operators.
bash build.sh --genop=${op_class}/${op_name}
```

After the command executes successfully, the following message appears:

```bash
Create the initial directory for ${op_name} under ${op_class} success
```

After creation, the directory structure is as follows:

```text
${op_name}                              # Replace with the lowercase underscore form of the actual operator name
├── examples                            # Operator invocation samples
│   └── test_aclnn_${op_name}.cpp       # Operator aclnn invocation sample
├── op_host                             # Host-side implementation
│   ├── ${op_name}_def.cpp              # Operator information library, defining basic operator information such as name, input/output, and data types
│   ├── ${op_name}_infershape.cpp       # InferShape implementation, implementing operator shape inference, inferring the output shape at runtime
│   └── ${op_name}_tiling.cpp           # Tiling implementation, dividing tensors into multiple small blocks and distinguishing data types for parallel computation
├── op_kernel                           # Device-side Kernel implementation
│   ├── ${op_name}_tiling_key.h         # Tilingkey file, defining the Key of the Tiling strategy to identify different partitioning methods
│   ├── ${op_name}_tiling_data.h        # Tilingdata file, storing configuration data related to the Tiling strategy, such as block size and parallelism
│   ├── ${op_name}.cpp                  # Kernel entry file, containing the main function and scheduling logic
│   └── ${op_name}.h                    # Kernel implementation file, defining the Kernel header file, including function declarations, structure definitions, and logic implementation
├── tests                               # UT implementation
│   └── ut                              # tiling/kernel/aclnn UT implementation
└── CMakeLists.txt                      # Operator cmakelist entry
```

If `${op_class}` is a new operator category, you need to additionally add `${op_class}` to the `OP_CATEGORY_LIST` in `cmake/variables.cmake`; otherwise, normal compilation is not possible.

## Operator Definition

Operator definition requires two deliverables: `README.md` and `${op_name}_def.cpp`

**Deliverable 1: README.md**

Before developing an operator, determine the functionality and computation logic of the target operator.

For an example of the custom `AddExample` operator, refer to [AddExample Operator Description](../../../examples/add_example/README.md).

**Deliverable 2: ${op_name}_def.cpp**

Operator information library.

For an example of the custom `AddExample` operator, refer to [AddExample Operator Information Library](../../../examples/add_example/op_host/add_example_def.cpp).

## Tiling Implementation

### Tiling Introduction

Because the internal storage space of the AI Core in the NPU is limited and cannot load the entire tensor data into the computation unit at once, the input tensor needs to be divided into multiple small blocks (Tiles) for computation one by one. This process is called Tiling.

The algorithm used to guide data partitioning is called the Tiling strategy or Tiling algorithm. It determines how to divide the input data into multiple computation blocks and guides the Kernel on how to allocate memory and schedule computation tasks. Tiling and the Kernel communicate information through the `TilingData` structure.

### Code Implementation

Tiling requires three deliverables: `${op_name}_tiling.cpp`, `${op_name}_tiling_key.h`, and `${op_name}_tiling_data.h`

> Note:
>
> 1. `${op_name}_tiling.cpp` is placed in the `${op_name}/op_host` directory;
> 2. `${op_name}_tiling_key.h` and `${op_name}_tiling_data.h` are placed in the `${op_name}/op_kernel` directory;
> 3. If `${op_name}_tiling.cpp` needs to reference `${op_name}_tiling_data.h`, use a relative path, for example: `#include "../op_kernel/${op_name}_tiling_data.h"`.

**Deliverable 1: ${op_name}_tiling.cpp**

Main Tiling partitioning logic.

For detailed implementation, refer to [add_example_tiling.cpp](../../../examples/add_example/op_host/add_example_tiling.cpp).

> **Explanation of empty function implementations in the sample:**
>
> 1. **TilingParse**: A standard deliverable for graph mode. The function definition is retained to meet the framework invocation specification. When there is no actual logic, it can be left empty.
> 2. **CompileInfo**: A standard deliverable for graph mode. The function definition is retained to meet the framework invocation specification. When there is no actual logic, it can be left empty.

```CPP
// ${op_name}_tiling.cpp
// 1. Tiling needs to obtain runtime environment information, including the available core count and UB (Unified Buffer) size, and pass the obtained information to CompileInfo. The auto-generated aclnn does not call this function; simply return ge::GRAPH_SUCCESS.
static ge::graphStatus TilingParse(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
    // If you are writing the aclnn interface manually, you can complete the parse function according to the following steps
    // // 1.1 Obtain environment information
    // auto compileInfo = context->GetCompiledInfo<CompileInfo>();
    // OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // auto platformInfo = context->GetPlatformInfo();
    // auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    // // 1.2 Obtain the available core count
    // compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    // // 1.3 Obtain the UB size
    // uint64_t ubSizePlatForm;
    // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    // compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    // ...
    // return ge::GRAPH_SUCCESS;
}

// 2. Tiling computation main entry
static ge::graphStatus TilingFunc(gert::TilingContext* context){
    // 2.1 Obtain platform information
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2.2 Obtain input information
    // Obtain the input tensor shape information
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);

    // If the input shape is a scalar, convert it to {1}; otherwise, keep the original shape unchanged
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    // Obtain the input tensor description information
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    // Obtain the data type
    dataType = inputDesc->GetDataType();

    // 2.3 Calculate Tiling parameters (design according to the operator functionality)
    ...

    // 2.4 Set TilingData information
    ${op_name}TilingData* tiling = context->GetTilingData<${op_name}TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(${op_name}TilingData), 0, sizeof(${op_name}TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalLength = totalIdx;
    tiling->tileNum = TILE_NUM;

    // 2.5 Set WorkspaceSize (optional)
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
}

// 3. Tiling registration entry
IMPL_OP_OPTILING(${op_name}).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
```

**Deliverable 2: ${op_name}_tiling_key.h**

TilingKey is a method within an operator that distinguishes different implementations by partitioning the kernel code. The kernel side can select different algorithm logic through TilingKey.

For detailed implementation, refer to [add_example_tiling_key.h](../../../examples/add_example/op_kernel/add_example_tiling_key.h).

> **Note:**
> For implementing complex parameter combinations to complete branch selection (involving multiple TilingKey scenarios), refer to [Ascend C Operator Development Interface](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/programug/Ascendcopdevg/atlas_ascendc_map_10_0002.html) in "Utils API > Tiling Template Programming > Template Parameter Meaning".

```CPP
// ${op_name}_tiling_key.h
ASCENDC_TPL_ARGS_DECL(
    ${op_name},
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1)));
```

**Deliverable 3: ${op_name}_tiling_data.h**

Parameters related to the partitioning algorithm, such as the total data volume size and the number of data chunks per core, are stored through a structure.

For detailed implementation, refer to [add_example_tiling_data.h](../../../examples/add_example/op_kernel/add_example_tiling_data.h).

```CPP
// ${op_name}_tiling_data.h
struct ${op_name}TilingData {
    int64_t totalLength;
    int64_t tileNum;
};
```

## Kernel Implementation

### Kernel Introduction

Kernel is the core part of an operator executed on the NPU, responsible for loading, computing, and storing tensor data. It is the final carrier of operator functionality implementation. The Kernel implementation needs to work closely with the Tiling strategy, performing memory allocation and computation scheduling based on the `TilingData` and `TilingKey` information provided by Tiling.

The Kernel implementation includes the following steps. The entire process is connected through the `Process` function, implementing the complete operator flow.

```mermaid
graph LR
    H([Kernel Function Definition]) -->A([Define Kernel Class])
    A -->B([Initialization Function<br>Init])
    B -->C([Main Processing Function<br>Process])
    subgraph C [Main Processing Function Process]
        D([Data Copy-In<br>CopyIn]) -->E([Computation<br>Compute]) -->F([Data Copy-Out<br>CopyOut])
    end
    F -->G([Kernel Execution Complete])
```

### Code Implementation

Kernel requires two deliverables: `${op_name}.cpp` and `${op_name}.h`

> Note:
>
> 1. `${op_name}.cpp` is the kernel entry function and must be placed in the `${op_name}/op_kernel` directory;
> 2. `${op_name}.h` can be placed in the corresponding directory according to different SoC or template configurations, for example: `${op_name}/op_kernel/arch32`, `${op_name}/op_kernel/arch35`, or `${op_name}/op_kernel/impl` directories;

**Deliverable 1: ${op_name}.cpp**

Kernel entry file, containing the main function and scheduling logic.

For detailed implementation, refer to [add_example.cpp](../../../examples/add_example/op_kernel/add_example.cpp).

```CPP
// 1. Kernel function definition
// schMode is a template parameter used to support computation paths for different data types (such as float and int32)
// __global__ __aicore__ indicates that this function is a global function that can be executed on the AI Core
template <uint32_t schMode>
__global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling){
    ....
    // Tiling registration entry
    REGISTER_TILING_DEFAULT(AddExampleTilingData);

    // Obtain TilingData through macro
    GET_TILING_DATA_WITH_STRUCT(AddExampleTilingData, tilingData, tiling);

    // Instantiate the Kernel object based on TilingKey and complete the computation
    if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_EXAMPLE_FLOAT)) { // The float data type follows this branch
        NsAddExample::AddExample<float> op;     // Obtain the operator Kernel instance
        op.Init(x, y, z, &tilingData);          // Initialize the operator Kernel instance
        op.Process();                           // Execute the operator Kernel instance
    }
    ....
}
```

**Deliverable 2: ${op_name}.h**

Define the Kernel header file, including function declarations, structure definitions, and logic implementation.

For detailed implementation, refer to [add_example.h](../../../examples/add_example/op_kernel/add_example.h).

```C++
// 2. Define the Kernel class
template <typename T>
class AddExample
{
public:
    // Default constructor; __aicore__ indicates that this function runs on the AI Core
    __aicore__ inline AddExample(){};
    // Initialization function, used to set input/output addresses and Tiling partitioning information computation
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddExampleTilingData* tilingData);
    // Main processing function, executing data copying and computation
    __aicore__ inline void Process();

private:
    // Function for copying data from GM to LM
    __aicore__ inline void CopyIn(int32_t progress);
    // Function for copying data from LM to GM
    __aicore__ inline void CopyOut(int32_t progress);
    // Function for executing computation; datalength represents the current data length being processed
    __aicore__ inline void Compute(const int32_t dataLength);

private:
    // Pipeline object, used for managing data flow (copying and computation pipeline)
    TPipe pipe_;
    // Input queue X, copied from GM to LM; BUFFER_NUM represents the buffer count. Enable double buffer for pipeline parallelism, set to 2
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX_;
    // Input queue Y, copied from GM to LM; BUFFER_NUM represents the buffer count. Enable double buffer for pipeline parallelism, set to 2
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY_;
    // Output queue Z, copied from LM to GM; BUFFER_NUM represents the buffer count. Enable double buffer for pipeline parallelism, set to 2
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ_;

    // GM address of input X
    GlobalTensor<T> inputGMX_;
    // GM address of input Y
    GlobalTensor<T> inputGMY_;
    // GM address of input Z
    GlobalTensor<T> outputGMZ_;

    // Total data length
    int64_t blockLength_ = 0;
    // Number of blocks each block is divided into
    int64_t tileNum_ = 0;
    // Data length processed by each tile
    int64_t tileLength_ = 0;
    ...
};

// 3. Initialization function Init
template <typename T>
__aicore__ inline void AddExample<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddExampleTilingData* tilingData)
{
    // 3.1 Initialize member variables
    blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
    ...
    // 3.2 Initialize GM addresses
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockLength_ * AscendC::GetBlockIdx(), blockLength_);
    ...
    // 3.3 Initialize queue lengths
    pipe.InitBuffer(inputQueueX_, BUFFER_NUM, tileLength_ * sizeof(T));
    ...
}

// 4. Main processing function Process
template <typename T>
__aicore__ inline void AddExample<T>::Process()
{
    // Calculate the number of data processing loops for the current core
    int32_t loopCount = tileNum_ * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);              // Data copy-in
        Compute(i);             // Computation
        CopyOut(i);             // Data copy-out
    }
}
...
```

## aclnn Adaptation

After operator development and compilation are completed, the aclnn interface (a set of C-based APIs) is automatically generated. You can directly invoke the aclnn interface in your application to call the operator.

To enable this invocation method, you need to generate the operator's corresponding binary package in advance and configure the binary compilation JSON file.

Using the `AddExample` operator as an example, register the operator's NPU model and implementation mode in the [ascendc_config.json](../../../scripts/kernel/binary_config/ascendc_config.json) file in the `scripts/kernel/binary_config` directory. The example is as follows; enter the actual name and compute_units.

```json
{"name":"AddExample", "compute_units": ["${soc_version}"], "auto_sync":true, "impl_mode" : "high_performance"}
```

## Compilation and Deployment

After operator development is completed, compile the operator project to generate a custom operator installation package *.run. The specific operations are as follows:

1. **Preparation.**

    Complete the basic environment setup by following [Project Creation](#project-creation), and check whether the operator development deliverables are complete and in the corresponding operator category directory.

2. **Compile the custom operator package.**

    Using the `AddExample` operator as an example, assuming the development deliverables are in the `examples` directory, the complete code is in the [add_example](../../../examples/add_example) directory.

    > Note: The compilation process depends on third-party open-source software. In online scenarios, it is automatically downloaded. In offline compilation scenarios, you need to install it manually. For details, refer to [Offline Compilation](../invocation/quick_op_invocation.md#offline-compilation).

    ```bash
    # Compile the specified operator, for example: bash build.sh --pkg --ops=add_example
    bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list} [--experimental]
    ```

    - --soc: ${soc_version} represents the NPU model. For Atlas A2 series products, use "ascend910b" (default). For Atlas A3 series products, use "ascend910_93". For Ascend 950PR/Ascend 950DT products, use "ascend950".
    - --vendor_name (optional): ${vendor_name} represents the name of the custom operator package to build. The default name is custom.
    - --ops (optional): ${op_list} represents the operators to compile. If not specified, all operators are compiled by default. The format is "--ops=add_example".
    - --experimental (optional): If the operator being compiled is a contributed operator, configure --experimental.

    If the following message appears, the compilation is successful:

    ```bash
    Self-extractable archive "cann-ops-nn-${vendor_name}-linux.${arch}.run" successfully created.
    ```

3. **Install the custom operator package.**

    ```bash
    # Install the run package
    ./build_out/cann-ops-nn-${vendor_name}-linux.${arch}.run
    ```

    The custom operator package is installed in the `${ASCEND_HOME_PATH}/opp/vendors` path. `${ASCEND_HOME_PATH}` represents the CANN software installation directory, which can be configured in the environment variable in advance.

4. **(Optional) Uninstall the custom operator package.**

    After the custom operator package is installed, an `uninstall.sh` script is generated in the `${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_nn/scripts` directory. You can uninstall the custom operator package through this script. The command is as follows:

    ```bash
    bash ${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_nn/scripts/uninstall.sh
    ```

## Operator Verification

Before verifying the operator, ensure that the environment variables are configured. The command is as follows:

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_nn/op_api/lib:${LD_LIBRARY_PATH}
```

- **UT Verification**

  During operator development, you can quickly verify through UT verification (such as Tiling). For detailed implementation, refer to [Tiling UT](../../../examples/add_example/tests/ut/op_host/test_add_example_tiling.cpp).
  For commands to execute UT verification, refer to [Operator Invocation](../invocation/quick_op_invocation.md).

- **aclnn Invocation Verification**

  After the developed operator is compiled and deployed, you can verify the functionality through the aclnn method. For the method, refer to [Operator Invocation Methods](../invocation/op_invocation.md).

## Appendix

### Operator Project Migration

Because the [Ascend/samples](https://gitee.com/ascend/samples/tree/master) project differs from this project (refer to [Project Creation](#project-creation)), the operator implementation deliverables and quantities are different. You can refer to the following table to migrate the operator samples in the `operator` directory.

<table border="1">
  <tr>
    <th>Ascend/samples</th>
    <th>This Project</th>
    <th>Migration Method</th>
    <th>Code Example</th>
  </tr>
  <tr>
    <td rowspan="4">op_host/{op_name}.cpp</td>
    <td>op_host/{op_name}_def.cpp</td>
    <td>Separate the operator prototype description part from the original op_host/{op_name}.cpp</td>
    <td><a href="#op_host/{op_name}_def.cpp">op_host/{op_name}_def.cpp</a></td>
  </tr>
  <tr>
    <td>op_host/{op_name}_infershape.cpp</td>
    <td>(Optional) Separate the type inference part from the original op_host/{op_name}.cpp</td>
    <td><a href="#op_host/{op_name}_infershape.cpp">op_host/{op_name}_infershape.cpp</a></td>
  </tr>
  <tr>
    <td>op_host/{op_name}_tiling.cpp</td>
    <td>Retain only the TilingFunc from the original op_host/{op_name}.cpp</td>
    <td><a href="#op_host/{op_name}_tiling.cpp">op_host/{op_name}_tiling.cpp</a></td>
  </tr>
  <tr>
    <td>op_graph/{op_name}_graph_infer.cpp</td>
    <td>(Optional) Separate the type inference part from the original op_host/{op_name}.cpp</td>
    <td><a href="#op_graph/{op_name}_graph_infer.cpp">op_graph/{op_name}_graph_infer.cpp</a></td>
  </tr>
  <tr>
    <td>op_host/{op_name}_tiling.h</td>
    <td>op_kernel/{op_name}_tiling_data.h</td>
    <td>Change the macro-defined TilingData structure definition in the original op_host directory to a C++ standard definition</td>
    <td><a href="#op_kernel/{op_name}_tiling_data.h">op_kernel/{op_name}_tiling_data.h</a></td>
  </tr>
  <tr>
    <td rowspan="2">op_kernel/{op_name}.cpp</td>
    <td>op_kernel/{op_name}.h</td>
    <td>Retain the operator class definition part of the Kernel implementation from the original op_host/{op_name}.cpp</td>
    <td><a href="#op_kernel/{op_name}.h">op_kernel/{op_name}.h</a></td>
  </tr>
  <tr>
    <td>op_kernel/{op_name}.cpp</td>
    <td>Migrate the Kernel implementation's core function from the original op_host/{op_name}.cpp to the cpp file, and:
      <ul>
        <li>Add REGISTER_TILING_DEFAULT call to register the TilingData structure, and use GET_TILING_DATA_WITH_STRUCT to obtain TilingData</li>
        <li>Add Tiling template to support template parameter input, and select different Kernel-side implementations based on template parameter branches</li>
      </ul>
    </td>
    <td><a href="#op_kernel/{op_name}.cpp">op_kernel/{op_name}.cpp</a></td>
  </tr>
  <tr>
    <td>op_kernel/tiling_key_{op_name}.h</td>
    <td>op_kernel/{op_name}_tiling_key.h</td>
    <td>Retain the operator's template parameter definitions from the original op_kernel/tiling_key_{op_name}.h. If op_kernel/tiling_key_{op_name}.h does not exist, add template parameter and template parameter combination definitions</td>
    <td><a href="#op_kernel/{op_name}_tiling_key.h">op_kernel/{op_name}_tiling_key.h</a></td>
  </tr>
</table>

<div id="op_host/{op_name}_def.cpp">
<p style="font-size:18px;"><b>op_host/{op_name}_def.cpp</b></p>
</div>

Migrate the operator information library content from the original ${op_name}.cpp to this file independently. You need to remove the SetInferShape and SetTiling content.

```CPP
// Operator information library content from the original ${op_name}.cpp
namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
        ....
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);   // SetInferShape needs to be removed
        this->AICore()
            .SetTiling(optiling::TilingFunc)                                       // SetTiling needs to be removed
            .AddConfig("ascend910")
            .AddConfig("ascend310p")
            .AddConfig("ascend310b")
            .AddConfig("ascend910b");
    }
};
OP_ADD(AddCustom);
} // namespace ops

// After migrating to op_host/{op_name}_def.cpp, the code does not contain SetInferShape and SetTiling content
namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
        ....
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore()
            .AddConfig("ascend910")
            .AddConfig("ascend310p")
            .AddConfig("ascend310b")
            .AddConfig("ascend910b");
    }
};
OP_ADD(AddCustom);
} // namespace ops
```

<div id="op_host/{op_name}_infershape.cpp">
<p style="font-size:18px;"><b>op_host/{op_name}_infershape.cpp</b></p>
</div>

For graph mode scenarios, you need to adapt this file. Migrate the shape inference part from the original ${op_name}.cpp to this file independently, and call the IMPL_OP_INFERSHAPE interface to complete InferShape registration.

```CPP
// InferShape from the original ${op_name}.cpp
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
} // namespace ge

// After migrating to op_host/{op_name}_infershape.cpp, call the IMPL_OP_INFERSHAPE interface to complete InferShape registration
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AddCustom).InferShape(InferShape);   // Complete InferShape registration in this file
} // namespace ge
```

<div id="op_host/{op_name}_tiling.cpp">
<p style="font-size:18px;"><b>op_host/{op_name}_tiling.cpp</b></p>
</div>

After migrating TilingFunc from the original ${op_name}.cpp to this file, call the IMPL_OP_OPTILING interface to complete TilingFunc registration.
After changing the macro-defined TilingData structure to a standard C++ structure, TilingFunc no longer uses the tiling.set_xxx method for assignment, but directly assigns values to member variables.
If you are adding new template parameter and template parameter combination definitions, you also need to configure the template parameter tilingKey in TilingFunc.
Refer to [add_example_tiling.cpp](../../../examples/add_example/op_host/add_example_tiling.cpp).

```CPP
// TilingFunc from the original ${op_name}.cpp
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t DEFAULT_TILE_NUM = 8;
constexpr int MIN_LENGTH_FOR_SPLIT = 2048;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
    ge::DataType dtype_z = context->GetOutputDesc(0)->GetDataType();
    ....
    tiling.set_totalLength(totalLength);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT); // Template parameter tilingKey configuration
    context->SetTilingKey(tilingKey);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

// After migrating to op_host/{op_name}_tiling.cpp, call the IMPL_OP_OPTILING interface to complete TilingFunc registration, and directly assign values to structure member variables
namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t DEFAULT_TILE_NUM = 8;
constexpr int MIN_LENGTH_FOR_SPLIT = 2048;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // TilingData tiling;
    TilingData* tiling = context->GetTilingData<TilingData>();
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_y = context->GetInputDesc(1)->GetDataType();
    ge::DataType dtype_z = context->GetOutputDesc(0)->GetDataType();
    ....
    tiling->totalLength = totalLength;   // Directly assign values to structure member variables
    // tiling.set_totalLength(totalLength);   // No longer use the tiling.set_xxx method for assignment
    // tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    // context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X, D_T_Y, D_T_Z, TILE_NUM, IS_SPLIT); // Template parameter tilingKey configuration
    context->SetTilingKey(tilingKey);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(AddCustom).Tiling(TilingFunc);   // Complete TilingFunc registration in this file
} // namespace optiling
```

<div id="op_graph/{op_name}_graph_infer.cpp">
<p style="font-size:18px;"><b>op_graph/{op_name}_graph_infer.cpp</b></p>
</div>

For graph mode scenarios, you need to adapt this file. After migrating the type inference from the original ${op_name}.cpp to this file independently, call the IMPL_OP interface to complete InferDataType registration.

```CPP
// InferDataType from the original ${op_name}.cpp
namespace ge {
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

// After migrating to op_graph/{op_name}_graph_infer.cpp, call the IMPL_OP interface to complete InferDataType registration
namespace ge {
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP(AddCustom).InferDataType(InferDataType);   // Complete InferDataType function registration in this file
} // namespace ge
```

<div id="op_kernel/{op_name}_tiling_data.h">
<p style="font-size:18px;"><b>op_kernel/{op_name}_tiling_data.h</b></p>
</div>

```CPP
// Macro-defined TilingData structure in the original op_host/{op_name}_tiling.h
namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(XXX, TilingData)
} // namespace optiling

// After migrating to op_kernel/{op_name}_tiling_data.h, change to a C++ standard structure
struct TilingData {
    uint32_t  totalLength;
};
```

<div id="op_kernel/{op_name}.h">
<p style="font-size:18px;"><b>op_kernel/{op_name}.h</b></p>
</div>

Retain the operator class definition part of the Kernel implementation from the original op_host/{op_name}.cpp.

<div id="op_kernel/{op_name}.cpp">
<p style="font-size:18px;"><b>op_kernel/{op_name}.cpp</b></p>
</div>

```CPP
// Core function implementation in the original op_kernel/{op_name}.cpp
template<int D_T_X, int D_T_Y, int D_T_Z, int TILE_NUM, int IS_SPLIT>
 __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if(D_T_X == ADD_TPL_FP32 && D_T_Y == ADD_TPL_FP32 && D_T_Z == ADD_TPL_FP32){
        KernelAdd<float, float, float> op;
        op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
        op.Process1();
    }else if(D_T_X == ADD_TPL_FP16 && D_T_Y == ADD_TPL_FP16 && D_T_Z == ADD_TPL_FP16){
        KernelAdd<half, half, half> op;
        if(IS_SPLIT == 0){
            op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
            op.Process1();
        }else if(IS_SPLIT == 1){
            op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
            op.Process2();
        }
    }
}

// After migrating to op_kernel/{op_name}.cpp, add REGISTER_TILING_DEFAULT call to register the TilingData structure, and use GET_TILING_DATA_WITH_STRUCT to obtain TilingData
template<int D_T_X, int D_T_Y, int D_T_Z, int TILE_NUM, int IS_SPLIT>
 __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // GET_TILING_DATA(tiling_data, tiling);
    REGISTER_TILING_DEFAULT(TilingData);   // Add REGISTER_TILING_DEFAULT call to register the TilingData structure
    GET_TILING_DATA_WITH_STRUCT(TilingData, tiling_data, tiling);   // Use GET_TILING_DATA_WITH_STRUCT macro to obtain TilingData
    if(D_T_X == ADD_TPL_FP32 && D_T_Y == ADD_TPL_FP32 && D_T_Z == ADD_TPL_FP32){
        KernelAdd<float, float, float> op;
        op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
        op.Process1();
    }else if(D_T_X == ADD_TPL_FP16 && D_T_Y == ADD_TPL_FP16 && D_T_Z == ADD_TPL_FP16){
        KernelAdd<half, half, half> op;
        if(IS_SPLIT == 0){
            op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
            op.Process1();
        }else if(IS_SPLIT == 1){
            op.Init(x, y, z, tiling_data.totalLength, TILE_NUM);
            op.Process2();
        }
    }
}
```

<div id="op_kernel/{op_name}_tiling_key.h">
<p style="font-size:18px;"><b>op_kernel/{op_name}_tiling_key.h</b></p>
</div>

Retain the operator's template parameter definitions from the original op_kernel/tiling_key_{op_name}.h. If op_kernel/tiling_key_{op_name}.h does not exist, refer to [add_example_tiling_key.h](../../../examples/add_example/op_kernel/add_example_tiling_key.h) to add template parameter and template parameter combination definitions.
