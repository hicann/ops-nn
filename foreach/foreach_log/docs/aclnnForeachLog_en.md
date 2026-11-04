# aclnnForeachLog

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T23:19:42.401Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

- Description: Performs an element-wise natural logarithm operation (ln(x)) on a tensor list.
- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = {\rm log}_e(x_i) (i=0,1,...n-1)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnForeachLogGetWorkspaceSize" API to obtain the workspace size required for computation and an executor that contains the operator computation process, and then call the "aclnnForeachLog" API to perform the computation.

```Cpp
aclnnStatus aclnnForeachLogGetWorkspaceSize(
  const aclTensorList *x,
  const aclTensorList *out,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnForeachLog(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachLogGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
    <tr>
      <td>x (aclTensorList*)</td>
      <td>Input</td>
      <td>Input tensor list for element-wise natural logarithm computation, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>All tensors in this parameter have the same data type.</li><li>The number of tensors must be the same as in `out` and must not exceed 256; when the number is 0, the call returns success directly and `workspaceSize` is 0.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND, NCHW, NHWC, HWCN, NDHWC, NCDHW</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensorList*)</td>
      <td>Output</td>
      <td>Output tensor list for element-wise natural logarithm computation, corresponding to `y` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>All tensors in this parameter must be of the same data type.</li><li>Has the same data type and data format as input `x`, and each tensor has the same shape (same rank and size of each dimension) as the corresponding tensor in input `x`.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND, NCHW, NHWC, HWCN, NDHWC, NCDHW</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of x or out is not supported.</td>
    </tr>
    <tr>
      <td>The data types of x and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of x and out do not meet the constraints.</td>
    </tr>
    <tr>
      <td>Tensors in x or out have inconsistent data types.</td>
    </tr>
    <tr>
      <td>The dimensions of tensors in x or out exceed 8.</td>
    </tr>
    <tr>
      <td>The data format of x or out is not in the supported list.</td>
    </tr>
  </tbody></table>

## aclnnForeachLog

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device, obtained by calling the first-phase API aclnnForeachLogGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnForeachLog defaults to a deterministic implementation.
- Data formats ND, NCHW, NHWC, HWCN, NDHWC, and NCDHW are supported; other formats return 161002.
- A single TensorList must not contain more than 256 tensors.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_log.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// Registers and releases all allocated device/ACL resources. Every failure branch runs the destructor,
// so aclDestroyTensorList/aclrtFree/aclrtDestroyStream/aclrtResetDevice/aclFinalize are never skipped.
struct Resources {
    aclTensorList* inputList = nullptr;
    aclTensorList* outputList = nullptr;
    void* input1 = nullptr;
    void* input2 = nullptr;
    void* out1 = nullptr;
    void* out2 = nullptr;
    void* workspace = nullptr;
    aclrtStream stream = nullptr;
    int32_t deviceId = -1;
    bool inited = false;

    ~Resources()
    {
        if (inputList != nullptr) {
            aclDestroyTensorList(inputList);
        }
        if (outputList != nullptr) {
            aclDestroyTensorList(outputList);
        }
        if (input1 != nullptr) {
            aclrtFree(input1);
        }
        if (input2 != nullptr) {
            aclrtFree(input2);
        }
        if (out1 != nullptr) {
            aclrtFree(out1);
        }
        if (out2 != nullptr) {
            aclrtFree(out2);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceId >= 0) {
            aclrtResetDevice(deviceId);
        }
        if (inited) {
            aclFinalize();
        }
    }
};

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
    // Boilerplate: resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        // Release the allocated device memory immediately on failure so it does not leak after returning.
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed.\n");
        // Also release the device memory when aclCreateTensor fails.
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_FAILURE;
    }
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Resources releases everything when main exits, including every early return in failure branches.
    Resources res;
    res.deviceId = deviceId;
    res.inited = true;
    res.stream = stream;

    // 2. Construct the input and output. Custom construction based on the API is required.
    std::vector<int64_t> selfShape1 = {2, 3};
    std::vector<int64_t> selfShape2 = {1, 3};
    std::vector<int64_t> outShape1 = {2, 3};
    std::vector<int64_t> outShape2 = {1, 3};
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensor* out1 = nullptr;
    aclTensor* out2 = nullptr;
    std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> input2HostData = {7, 8, 9};
    std::vector<float> out1HostData(6, 0);
    std::vector<float> out2HostData(3, 0);

    // Create the input1 aclTensor.
    ret = CreateAclTensor(input1HostData, selfShape1, &res.input1, aclDataType::ACL_FLOAT, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the input2 aclTensor.
    ret = CreateAclTensor(input2HostData, selfShape2, &res.input2, aclDataType::ACL_FLOAT, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the out1 aclTensor.
    ret = CreateAclTensor(out1HostData, outShape1, &res.out1, aclDataType::ACL_FLOAT, &out1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the out2 aclTensor.
    ret = CreateAclTensor(out2HostData, outShape2, &res.out2, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<aclTensor*> tempInput{input1, input2};
    res.inputList = aclCreateTensorList(tempInput.data(), tempInput.size());
    CHECK_RET(res.inputList != nullptr, LOG_PRINT("aclCreateTensorList(input) failed.\n"); return ACL_ERROR_FAILURE);
    std::vector<aclTensor*> tempOutput{out1, out2};
    res.outputList = aclCreateTensorList(tempOutput.data(), tempOutput.size());
    CHECK_RET(res.outputList != nullptr, LOG_PRINT("aclCreateTensorList(output) failed.\n"); return ACL_ERROR_FAILURE);

    // 3. Call the CANN operator library API. Replace with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnForeachLog.
    ret = aclnnForeachLogGetWorkspaceSize(res.inputList, res.outputList, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachLogGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&res.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnForeachLog.
    ret = aclnnForeachLog(res.workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachLog failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the results from device memory to the host.
    auto size = GetShapeSize(outShape1);
    std::vector<float> out1Data(size, 0);
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), res.out1,
                      size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out1 result[%ld] is: %f\n", i, out1Data[i]);
    }

    size = GetShapeSize(outShape2);
    std::vector<float> out2Data(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), res.out2,
                      size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out2 result[%ld] is: %f\n", i, out2Data[i]);
    }

    // 6/7. aclTensor and device resources are released by the Resources destructor.
    return 0;
}
```
