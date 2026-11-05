# aclnnForeachZeroInplace

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T03:40:54.957Z -->

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

- Description: Updates the input tensor list in-place, setting each tensor in the input tensor list to 0.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  $$

  $$
  x_i = 0 (i=0,1,...n-1)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnForeachZeroInplaceGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnForeachZeroInplace" API to execute the computation.

```Cpp
aclnnStatus aclnnForeachZeroInplaceGetWorkspaceSize(
  const aclTensorList *x,
  uint64_t *workspaceSize,
  aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnForeachZeroInplace(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachZeroInplaceGetWorkspaceSize

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
      <td>Input/Output</td>
      <td>Input and output tensor list for the zeroing operation, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The number of tensors in the TensorList ranges from 0 to 256. If the number is 0 (empty list), the API returns success without performing computation.</li><li>All tensors in this parameter must have the same data type.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT16, INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
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
      <td>The input x is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>The data type of x is not supported.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="3">561002</td>
      <td>The data types of tensors in x are inconsistent.</td>
    </tr>
    <tr>
      <td>The number of dimensions of a tensor in x exceeds 8.</td>
    </tr>
    <tr>
      <td>The number of tensors in x exceeds 256.</td>
    </tr>
  </tbody></table>

## aclnnForeachZeroInplace

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
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnForeachZeroInplaceGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- All tensors in the TensorList must use the same data type.
- If x is an empty TensorList or all tensors in x are empty, the API returns success without performing computation.
- Deterministic computation:
  - aclnnForeachZeroInplace defaults to a deterministic implementation.

## Example

The following code demonstrates initialization, input construction, the two-phase call, synchronization, result copy-back, and resource cleanup.

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_zero_inplace.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void LogCleanupFailure(const char* operation, aclError status)
{
    if (status != ACL_SUCCESS) {
        LOG_PRINT("%s failed. ERROR: %d\n", operation, status);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Initialize ACL resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
              LogCleanupFailure("aclFinalize", aclFinalize()); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
              LogCleanupFailure("aclrtResetDevice", aclrtResetDevice(deviceId));
              LogCleanupFailure("aclFinalize", aclFinalize()); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    *deviceAddr = nullptr;
    *tensor = nullptr;
    auto size = GetShapeSize(shape) * sizeof(T);
    // Allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Copy input data from the host to the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate strides for a contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);
    return 0;
}

int main()
{
    // 1. Initialize the device and stream.
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input TensorList.
    std::vector<int64_t> selfShape1 = {2, 3};
    std::vector<int64_t> selfShape2 = {1, 3};
    void* input1DeviceAddr = nullptr;
    void* input2DeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensorList* tensorListInput = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool streamHasPendingWork = false;

    auto cleanup = [&](int status) {
        bool deviceWasReset = false;
        auto recordCleanupStatus = [&](const char* operation, aclError cleanupStatus) {
            if (cleanupStatus != ACL_SUCCESS) {
                LOG_PRINT("%s failed. ERROR: %d\n", operation, cleanupStatus);
                if (status == ACL_SUCCESS) {
                    status = cleanupStatus;
                }
            }
        };
        if (stream != nullptr) {
            const aclError streamStatus = streamHasPendingWork ? aclrtDestroyStreamForce(stream) :
                                                                 aclrtDestroyStream(stream);
            recordCleanupStatus(streamHasPendingWork ? "aclrtDestroyStreamForce" : "aclrtDestroyStream", streamStatus);
            if (streamHasPendingWork && streamStatus != ACL_SUCCESS) {
                const aclError resetStatus = aclrtResetDeviceForce(deviceId);
                recordCleanupStatus("aclrtResetDeviceForce", resetStatus);
                if (resetStatus != ACL_SUCCESS) {
                    LOG_PRINT("Device reset failed; retain in-flight resources until process exit.\n");
                    return status;
                }
                deviceWasReset = true;
                workspaceAddr = nullptr;
                input1DeviceAddr = nullptr;
                input2DeviceAddr = nullptr;
            }
            stream = nullptr;
            streamHasPendingWork = false;
        }
        if (tensorListInput != nullptr) {
            recordCleanupStatus("aclDestroyTensorList", aclDestroyTensorList(tensorListInput));
            tensorListInput = nullptr;
        } else {
            if (input1 != nullptr) {
                recordCleanupStatus("aclDestroyTensor(input1)", aclDestroyTensor(input1));
                input1 = nullptr;
            }
            if (input2 != nullptr) {
                recordCleanupStatus("aclDestroyTensor(input2)", aclDestroyTensor(input2));
                input2 = nullptr;
            }
        }
        if (!deviceWasReset && workspaceAddr != nullptr) {
            recordCleanupStatus("aclrtFree(workspace)", aclrtFree(workspaceAddr));
            workspaceAddr = nullptr;
        }
        if (!deviceWasReset && input1DeviceAddr != nullptr) {
            recordCleanupStatus("aclrtFree(input1)", aclrtFree(input1DeviceAddr));
            input1DeviceAddr = nullptr;
        }
        if (!deviceWasReset && input2DeviceAddr != nullptr) {
            recordCleanupStatus("aclrtFree(input2)", aclrtFree(input2DeviceAddr));
            input2DeviceAddr = nullptr;
        }
        if (!deviceWasReset) {
            recordCleanupStatus("aclrtResetDevice", aclrtResetDevice(deviceId));
        }
        recordCleanupStatus("aclFinalize", aclFinalize());
        return status;
    };

    std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> input2HostData = {7, 8, 9};

    // Create input1.
    ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));

    // Create input2.
    ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));

    std::vector<aclTensor*> tempInput{input1, input2};
    tensorListInput = aclCreateTensorList(tempInput.data(), tempInput.size());
    CHECK_RET(tensorListInput != nullptr, LOG_PRINT("aclCreateTensorList failed\n"); return cleanup(ACL_ERROR_FAILURE));

    // 3. Call the two-phase API.
    // Call the first-phase API.
    ret = aclnnForeachZeroInplaceGetWorkspaceSize(tensorListInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachZeroInplaceGetWorkspaceSize failed. ERROR: %d\n", ret);
              return cleanup(ret));
    // Allocate workspace based on workspaceSize.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return cleanup(ret));
    }
    // Call the second-phase API.
    ret = aclnnForeachZeroInplace(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachZeroInplace failed. ERROR: %d\n", ret); return cleanup(ret));
    streamHasPendingWork = true;

    // 4. Wait for execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return cleanup(ret));
    streamHasPendingWork = false;

    // 5. Copy the in-place results back to the host.
    auto size = GetShapeSize(selfShape1);
    std::vector<float> self1Data(size, 0);
    ret = aclrtMemcpy(self1Data.data(), self1Data.size() * sizeof(self1Data[0]), input1DeviceAddr,
                      size * sizeof(self1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return cleanup(ret));
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out1 result[%ld] is: %f\n", i, self1Data[i]);
    }

    size = GetShapeSize(selfShape2);
    std::vector<float> self2Data(size, 0);
    ret = aclrtMemcpy(self2Data.data(), self2Data.size() * sizeof(self2Data[0]), input2DeviceAddr,
                      size * sizeof(self2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return cleanup(ret));
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out2 result[%ld] is: %f\n", i, self2Data[i]);
    }

    // 6. Release tensor and device resources.
    return cleanup(ACL_SUCCESS);
}
```
