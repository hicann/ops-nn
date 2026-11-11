# aclnnGroupedQuantMax

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:35:15.308Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Quantizes input x into groups based on the input scale and computes amax, the maximum absolute value of input x for each group.

- Formula:

  $group\_list$ splits $x$ along dim-0 into num_groups groups, and $group\_list[g]$ is the cumulative size of the first $g+1$ groups along dim-0 (monotonically increasing, $group\_list[-1]=x.shape[0]$). For each group (g = 0, 1, ..., num_groups - 1):

    $$
    x\_g = x[group\_list[g-1] : group\_list[g], ...]
    $$

    $$
    y = \text{Cast}(x\_g \times scale[g], \text{dst\_type}, \text{round\_mode})
    $$

    $$
    amax[g] = \max(|x\_g|)
    $$

    where $x$ indicates the input tensor, $x\_g$ indicates the g-th group of data in the input tensor, and the lower bound is 0 when g=0; $scale$ indicates the quantization scale factor, $amax[g]$ is the maximum absolute value of the corresponding group of the input tensor, $\text{dst\_type}$ specifies the output type, and $\text{round\_mode}$ specifies the rounding mode.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnGroupedQuantMaxGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call "aclnnGroupedQuantMax" to perform the computation.

```cpp
aclnnStatus aclnnGroupedQuantMaxGetWorkspaceSize(
  const aclTensor  *x,
  const aclTensor  *scale,
  const aclTensor  *groupList,
  const char       *roundMode,
  int64_t           dstType,
  aclTensor        *y,
  aclTensor        *amax,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnGroupedQuantMax(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnGroupedQuantMaxGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor to be quantized, corresponding to x in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Input</td>
      <td>Quantization scale factor, corresponding to scale in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is fixed to [num_groups].</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupList (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the offset of each group of input x on axis 0 (cumsum mode), corresponding to group_list in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Indicates the start index of quantization grouping. The values must be greater than 0 and monotonically increasing, and the last value must be equal to the size of axis 0 of x.</li><li>The shape is [num_groups], and num_groups ranges from 1-384.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundMode (char*)</td>
      <td>Input</td>
      <td>Specifies the rounding mode.</td>
      <td><ul><li>If dstType is 34, roundMode supports "round" and "hybrid".</li></ul><ul><li>If dstType is 35 or 36, roundMode supports "rint".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Specifies the output quantization data type. 34=HIFLOAT8, 35=FLOAT8_E5M2, 36=FLOAT8_E4M3.</td>
      <td><ul><li>Value range: {34, 35, 36}.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Quantized output tensor, corresponding to y in the formula.</td>
      <td><ul><li>Data type is determined by dstType.</li><li>Shape is the same as x.</li></ul></td>
      <td>HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>amax (aclTensor*)</td>
      <td>Output</td>
      <td>Maximum absolute value of x, corresponding to amax in the formula.</td>
      <td><ul><li>The data type must be the same as that of x.</li><li>The shape is fixed to [num_groups].</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x, scale, groupList, y, or amax is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of x, scale, or groupList is not supported.</td>
    </tr>
    <tr>
      <td>The value of dstType is not within the range of {34, 35, 36}.</td>
    </tr>
    <tr>
      <td>roundMode is invalid.</td>
    </tr>
    <tr>
      <td>The number of dimensions of x is not in the range of 2-8, or the shapes of scale and amax are not [num_groups].</td>
    </tr>
  </tbody></table>

## aclnnGroupedQuantMax

- **Parameters**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>Input</td><td>Workspace memory address allocated on the Device.</td></tr>
    <tr><td>workspaceSize</td><td>Input</td><td>Workspace size allocated on the Device, obtained by the first-phase API aclnnGroupedQuantMaxGetWorkspaceSize.</td></tr>
    <tr><td>executor</td><td>Input</td><td>Op executor that contains the operator computation process.</td></tr>
    <tr><td>stream</td><td>Input</td><td>Stream that specifies the task to be executed.</td></tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnGroupedQuantMax defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_quant_max.h"

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
    // Boilerplate: resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnGroupedQuantMaxTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("Init acl success.\n");

    std::vector<int64_t> xShape = {4, 4};
    std::vector<int64_t> scaleShape = {1};
    std::vector<int64_t> groupListShape = {1};
    std::vector<int64_t> amaxShape = {1};

    // The value of x can cover multiple cases.
    std::vector<float> xHostData = {1.0f, -2.0f, 3.0f, -4.0f,  5.0f,    -6.0f,  7.0f,    -8.0f,
                                    0.5f, -0.5f, 0.0f, 448.0f, -448.0f, 0.125f, -0.125f, 2.5f};

    // scale = 1.0
    std::vector<float> scaleHostData = {1.0f};

    // Initialize output y to 0.
    std::vector<uint8_t> yHostData(GetShapeSize(xShape), 0);

    // Initialize amax to 0.
    std::vector<int64_t> groupListHostData = {4};
    std::vector<float> amaxHostData = {0.0f};

    void* xDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* amaxDeviceAddr = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* scaleTensor = nullptr;
    aclTensor* groupListTensor = nullptr;
    aclTensor* yTensor = nullptr;
    aclTensor* amaxTensor = nullptr;

    // Create input x tensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &xTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(xTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create scale tensor (float, shape [1])
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scaleTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scaleTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create groupList tensor (int64, shape [1])
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupListTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupListTensorPtr(groupListTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create output y tensor (FLOAT8_E4M3FN, same shape as x)
    ret = CreateAclTensor(yHostData, xShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(yTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create output amax tensor (float, shape [1])
    ret = CreateAclTensor(amaxHostData, amaxShape, &amaxDeviceAddr, aclDataType::ACL_FLOAT, &amaxTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> amaxTensorPtr(amaxTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> amaxDeviceAddrPtr(amaxDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call grouped_quant_max API
    int64_t dstType = 36; // FLOAT8_E4M3FN
    char* roundMode = const_cast<char*>("rint");
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // First stage: get workspace size
    ret = aclnnGroupedQuantMaxGetWorkspaceSize(
        xTensor, scaleTensor, groupListTensor, roundMode, dstType, yTensor, amaxTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedQuantMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnGroupedQuantMaxGetWorkspaceSize success, workspaceSize: %lu\n", workspaceSize);

    // Allocate workspace
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // Second stage: execute
    ret = aclnnGroupedQuantMax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedQuantMax failed. ERROR: %d\n", ret); return ret);

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnGroupedQuantMax execution success.\n");

    // 5. Get results
    // Copy y (FLOAT8_E4M3FN) back to host
    auto ySize = GetShapeSize(xShape);
    std::vector<uint8_t> yOutData(ySize, 0);
    ret = aclrtMemcpy(
        yOutData.data(), ySize * sizeof(uint8_t), yDeviceAddr, ySize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

    // Copy amax back to host
    std::vector<float> amaxOutData(1, 0);
    ret = aclrtMemcpy(amaxOutData.data(), sizeof(float), amaxDeviceAddr, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy amax from device to host failed. ERROR: %d\n", ret); return ret);

    // Print results
    auto size = GetShapeSize(xShape);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
    }
    LOG_PRINT("amaxOut is: %f\n", amaxOutData[0]);

    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. Refer to the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGroupedQuantMaxTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedQuantMaxTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    LOG_PRINT("All test cases passed!\n");
    return 0;
}
```
