# aclnnGroupedDynamicMxQuant

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/quant/grouped_dynamic_mx_quant)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- API description: Dynamically quantizes the input data to float8 based on the start value of the input group index.

- Formula:
  - The input x is first grouped in dimension 0 based on groupIndex. Each group is further divided into k groups, where k is equal to blocksize. The quantization scale mxscale_pre and $\{mxscale\_pre, \{P_i\}_{i=1}^{k}\}$ corresponding to the group of numbers $\{\{x_i\}_{i=1}^{k}\}$ are calculated using the following formulas (1) and (2).
  $$
  shared\_exp = floor(log_2(max_i(|V_i|))) - emax  \tag{1}
  $$
  $$
  mxscale\_pre = 2^{shared\_exp}  \tag{2}
  $$
  - Each number in the group is divided by mxscale and converted to the corresponding dst_type based on round_mode to obtain the quantization result y. The calculation formula is as follows (3):
  $$
  P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize \tag{3}
  $$
  
  ​ The quantized $P_i$ forms the output y according to the position of the corresponding $x_i$. mxscale_pre is grouped based on the corresponding groupIndex. The first dimension in each group is padded with an even number, and the output mxscale is formed.
  
  - emax: exponent of the maximum positive regular number of the corresponding data type.
  
    |   DataType    | emax |
    | :-----------: | :--: |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. You must call aclnnGroupedDynamicMxQuantGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnGroupedDynamicMxQuant to perform the computation.

```cpp
aclnnStatus aclnnGroupedDynamicMxQuantGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *groupIndex, 
  const char      *roundMode, 
  int64_t          dstType, 
  int64_t          blocksize, 
  const aclTensor *y, 
  const aclTensor *mxscale, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedDynamicMxQuant(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnGroupedDynamicMxQuantGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor, Input x in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndex (aclTensor*)</td>
      <td>Input</td>
      <td>Start index of a quantization group.</td>
      <td><ul><li>Empty tensors are not supported. </li><li> The index must be greater than or equal to 0 and does not decrease. The last number must be the same as the first dimension of x.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundMode (char*) </td>
      <td>Input</td>
      <td>round_mode in the formula, which indicates the data conversion mode.</td>
      <td>Only the rint mode is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>dst_type in the formula, which specifies the type of the output y after data conversion.</td>
      <td>The value can be {35, 36}, corresponding to the output y data type {35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blocksize (int64_t)</td>
      <td>Input</td>
      <td>blocksize in the formula, which specifies the number of elements to be quantized each time.</td>
      <td>Currently, only 32 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Quantized output tensor, It corresponds to y in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape dimension is the same as that of x.</li><li>The data type can be FLOAT8_E4M3FN or FLOAT8_E5M2, which must correspond to dstType.</li></ul></td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale (aclTensor*)</td>
      <td>Output</td>
      <td>mxscale output formed by mxscale_pre in the formula. It indicates the quantization scale of each group.</td>
      <td><ul><li>Empty tensors are supported. </li><li>If the shape of x is [m, n] and the shape of groupedIndex is [g], then the shape of mxscale is [(m/(blocksize*2)+g), n, 2].</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
  </colgroup>
  <thead>
  <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>The required input, output, or attribute is passed as a null pointer.</td>
  </tr>
  <tr>
    <td>The input roundMode is a null pointer.</td>
  </tr>
  <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data types of x, groupIndex, y, and mxscale are not supported.</td>
  </tr>
  <tr>
    <td>The shapes of x, y, and mxscale do not meet the verification conditions.</td>
  </tr>
  <tr>
    <td>approximate, quantMode, roundMode, and dstType are not supported.</td>
  </tr>
  <tr>
    <td>The dimensions of x, groupIndex, y, and mxscale are not supported.</td>
  </tr>
  <tr>
    <td>roundMode, dstType, and blocksize are not supported.</td>
  </tr>
  <tr>
    <td>mxscale does not support non-contiguous tensors.</td>
  </tr>
  </tbody>
  <tr>
    <td>ACLNN_ERR_RUNTIME_ERROR</td>
    <td>361001</td>
    <td>The current platform is not supported.</td>
  </tr>
  </table>

## aclnnGroupedDynamicMxQuant

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnGroupedDynamicMxQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The aclnnGroupedDynamicMxQuant is implemented in deterministic mode by default.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
  
```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_dynamic_mx_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // (Fixed writing) Initialize resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnGroupedDynamicMxQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> xShape = {8, 1};
    std::vector<int64_t> groupedIndexShape = {2};
    std::vector<int64_t> yOutShape = {8, 1};
    std::vector<int64_t> mxscaleOutShape = {2, 1, 2};
    void* xDeviceAddr = nullptr;
    void* groupedIndexDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* mxscaleOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* groupedIndex = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* mxscaleOut = nullptr;
    // Values for BF16: (0, 8, 64, 512)
    std::vector<uint16_t> xHostData = {{0}, {16640}, {17024}, {17408}, {0}, {16640}, {17024}, {17408}};
    
    std::vector<uint32_t> groupedIndexHostData = {4,8};
    // Values for float8_e4m3: (0, 4, 32, 256)
    std::vector<uint8_t> yOutHostData = {{0}, {72}, {96}, {120}, {0}, {72}, {96}, {120}};
    // Value of float8_e8m0 (2)
    std::vector<std::vector<uint8_t>> mxscaleOutHostData = {{{128, 0}}, {{128, 0}}};
    const char* roundModeOptional = "rint";
    int64_t dstType = 36;
    int64_t blocksize = 32;
    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the groudedIndex aclTensor.
    ret = CreateAclTensor(groupedIndexHostData, groupedIndexShape, &groupedIndexDeviceAddr, aclDataType::ACL_INT32, &groupedIndex);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupedIndexTensorPtr(groupedIndex, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupedIndexDeviceAddrPtr(groupedIndexDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the yOut aclTensor.
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an mxscaleOut ACL tensor.
    ret = CreateAclTensor(mxscaleOutHostData, mxscaleOutShape, &mxscaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscaleOutTensorPtr(mxscaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscaleOutDeviceAddrPtr(mxscaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    // Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first part of the aclnnGroupedDynamicMxQuant API.
    ret = aclnnGroupedDynamicMxQuantGetWorkspaceSize(x, groupedIndex, roundModeOptional, dstType, blocksize, yOut, mxscaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second API of aclnnGroupedDynamicMxQuant.
    ret = aclnnGroupedDynamicMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuant failed. ERROR: %d\n", ret); return ret);

    // (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(yOutShape);
    std::vector<uint8_t> yOutData(
        size, 0); // In C language, the fp4 data cannot be directly printed. You need to read the data using uint8 and convert it to fp4 in binary mode.
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                    size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yOutData[i]);
    }
    size = GetShapeSize(mxscaleOutShape);
    std::vector<uint8_t> mxscaleOutData(
        size, 0); // In C language, the fp8 data cannot be directly printed. You need to read the data using uint8 and convert it to fp8 in binary mode.
    ret = aclrtMemcpy(mxscaleOutData.data(), mxscaleOutData.size() * sizeof(mxscaleOutData[0]), mxscaleOutDeviceAddr,
                    size * sizeof(mxscaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscaleOut from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mxscaleOut[%ld] is: %d\n", i, mxscaleOutData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGroupedDynamicMxQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicMxQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
