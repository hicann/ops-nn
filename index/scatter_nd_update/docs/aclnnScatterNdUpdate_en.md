# aclnnScatterNdUpdate

📄 [View Source Code](https://gitcode.com/cann/ops-nn/tree/master/index/scatter_nd_update)

## Supported Products

| Product| Supported|
| :--- | :---: |
| Ascend 950PR/Ascend 950DT| √ |
| <term>Atlas A3 training products/Atlas A3 inference products</term>| √ |
| <term>Atlas A2 training products/Atlas A2 inference products</term>| √ |
| <term>Atlas 200I/500 A2 inference products</term>| × |
| <term>Atlas inference products</term>| × |
| <term>Atlas training products</term>| × |

## Function

Updates the values in **varRef** in sequence by referring to the values in the **updates** tensor based on the specified **indices**.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnScatterNdUpdateGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnScatterNdUpdate** is called to perform computation.

```Cpp
aclnnStatus aclnnScatterNdUpdateGetWorkspaceSize(
  aclTensor*       varRef,
  const aclTensor* indices,
  const aclTensor* updates,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScatterNdUpdate(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnScatterNdUpdateGetWorkspaceSize

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
        <td>varRef (aclTensor*) </td>
        <td>Input/Output</td>
        <td>Target tensor to be updated.</td>
        <td>The data type must be the same as that of updates.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32, INT64, BOOL, INT8</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices (aclTensor*) </td>
        <td>Input</td>
        <td>Index tensor.</td>
        <td>The index data in indices cannot be out of bounds.</td>
        <td>INT32 or INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>updates (aclTensor*) </td>
        <td>Input</td>
        <td>Tensor of updated data.</td>
        <td>The data type must be the same as that of varRef.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32, INT64, BOOL, INT8</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
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

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 300px">
    <col style="width: 134px">
    <col style="width: 716px">
    </colgroup>
    <thead>
      <tr>
        <th>Return</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The input varRef, indices, and updates are null pointers.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002/561103</td>
        <td>The shapes of varRef, indices, and updates do not meet the requirements.</td>
      </tr>
      <tr>
        <td>The data types of varRef, indices, and updates are not supported.</td>
      </tr>
    </tbody></table>

## aclnnScatterNdUpdate

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
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
        <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnScatterNdUpdateGetWorkspaceSize API.</td>
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
    </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnScatterNdUpdate** defaults to a deterministic implementation.

- Input shape restrictions:
  - **indices** must have at least two dimensions. The size of the last dimension must not exceed the dimension size of **varRef**.
  - Assume that the size of the last dimension of **indices** is **a**. The shape of **updates** is equal to the shape of **indices** excluding the last dimension plus the last dimension (**a**) of the shape of **varRef**. For example, if the shape of **varRef** is (4, 5, 6) and the shape of **indices** is (3, 2), the shape of **updates** must be (3, 6).

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_nd_update.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> varRefShape = {2, 3, 7};
  std::vector<int64_t> indicesShape = {2, 2};
  std::vector<int64_t> updatesShape = {2, 7};
  void* varRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  aclTensor* varRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  std::vector<float> varRefHostData = {0.3816, 0.3939, 0.8474, 0.1652, 0.6049, 0.3315, 0.4954,
                                     0.3284, 0.7060, 0.4359, 0.6514, 0.9476, 0.4708, 0.0656,
                                     0.9652, 0.9512, 0.6452, 0.1981, 0.4159, 0.9575, 0.1516,
                                     0.4987, 0.9107, 0.6635, 0.4119, 0.4845, 0.5558, 0.2749,
                                     0.6230, 0.1180, 0.2400, 0.9971, 0.4093, 0.5561, 0.4023,
                                     0.6612, 0.4109, 0.8470, 0.9733, 0.6947, 0.7980, 0.7957};
  std::vector<int64_t> indicesHostData = {5, 0, 1, 5};
  std::vector<float> updatesHostData = {0.7804, 0.3411, 0.6674, 0.8468, 0.6679, 0.5549, 0.9893,
                                    0.2086, 0.2473, 0.5110, 0.4549, 0.3113, 0.8490, 0.9217};
  // Create a varRef aclTensor.
  ret = CreateAclTensor(varRefHostData, varRefShape, &varRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an updates aclTensor.
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnScatterNdUpdate.
  ret = aclnnScatterNdUpdateGetWorkspaceSize(varRef, indices, updates, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterNdUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnScatterNdUpdate.
  ret = aclnnScatterNdUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterNdUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(varRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(varRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);
  // 7. Release device resources. Set the parameters based on the API definition.
  aclrtFree(varRefDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(updatesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
