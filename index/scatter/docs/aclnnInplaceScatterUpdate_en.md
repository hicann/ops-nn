# aclnnInplaceScatterUpdate

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/scatter)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √     |

## Function

Updates the values in **data** one by one by referring to the values in **updates** based on the specified **axis** and **indices**. The operator semantics are customized without corresponding TensorFlow or PyTorch APIs.

- Example:
  This operator has three inputs and one attribute: data, updates, indices, and axis. The data is the tensor to be updated, updates is the tensor that stores the updated data, indices indicates the update position, and axis indicates the specified update dimension. When **indices** is one-dimensional, there are two scenarios:

  **Scenario 1:** When **indices** is one-dimensional, **axis** specifies that the shape of the update dimension is 1 and **indices** specifies the offset of each batch dimension (the highest dimension) in the **axis** dimension.

  ```text
  Input example:
  data:(a, b, c, d)
  updates:(a, b, 1, d)
  indices:(a,)
  axis = -2
  ```

      data[i][j][indices[i]][k] = updates[i][j][0][k] # if dim=-2
      data[i][j][k][indices[i]] = updates[i][j][k][0] # if dim=-1

  **Scenario 2:** When **indices** is one-dimensional, **axis** specifies that the shape of the update dimension is greater than 1 and **indices** specifies the offset of each batch dimension (the highest dimension) in the **axis** dimension.

  ```text
  Input example:
  data:(a, b, c, d)
  updates:(a, b, e, d), indices[i] + e <= c
  indices:(a,)
  axis = -2 or 2
  ```

      data[i][j][indices[i]+k][l] = updates[i][j][k][l] # if dim=-2
      data[i][j][k][indices[i]+l] = updates[i][j][k][l] # if dim=-1

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnInplaceScatterUpdateGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnInplaceScatterUpdate** is called to perform computation.

```Cpp
aclnnStatus aclnnInplaceScatterUpdateGetWorkspaceSize(
    aclTensor*       data,
    const aclTensor* indices,
    const aclTensor* updates,
    int64_t          axis,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceScatterUpdate(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnInplaceScatterUpdateGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1477px"><colgroup>
    <col style="width: 147px">
    <col style="width: 120px">
    <col style="width: 233px">
    <col style="width: 277px">
    <col style="width: 270px">
    <col style="width: 121px">
    <col style="width: 164px">
    <col style="width: 145px">
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
        <td>data (aclTensor*) </td>
        <td>Input/Output</td>
        <td>Tensor to be updated.</td>
        <td>The number of dimensions must be the same as that of updates. Empty tensors are not supported.</td>
        <td>INT8, UINT8, FLOAT16, FLOAT32, INT32, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
        <td>ND</td>
        <td>2-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices (aclTensor*) </td>
        <td>Input</td>
        <td>Update position index. Only non-negative indexes are supported. Index data cannot be out of bounds.</td>
        <td>Empty tensors are not supported.</td>
        <td>INT32 or INT64</td>
        <td>ND</td>
        <td>0-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>updates (aclTensor*) </td>
        <td>Input</td>
        <td>Tensor for storing updated data.</td>
        <td>The data type must be the same as that of the input data, and the number of dimensions in the shape must be the same as that of the data shape. Empty tensors are not supported.</td>
        <td>INT8, UINT8, FLOAT16, FLOAT32, INT32, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
        <td>ND</td>
        <td>2-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>axis (int64_t) </td>
        <td>Input</td>
        <td>Dimension used for scatter.</td>
        <td>The value range is (–data_rank, data_rank), and axis cannot be 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
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

    - <term>Atlas training products</term>: The data type does not support UINT8 or BFLOAT16.
    - Ascend 950PR/Ascend 950DT: Data types such as FLOAT8_E4M3FN, FLOAT8_E5M2 and HIFLOAT8 are supported only by this model.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>The input data, indices, or updates is a null pointer. </td>
      </tr>
      <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of data, indices, or updates is not supported.</td>
      </tr>
      <tr>
      <td>The data types of data and updates are different.</td>
      </tr>
      <tr>
      <td>The number of dimensions of data is inconsistent with that of updates.</td>
      </tr>
      <tr>
      <td>The dimension of indices is not zero-dimensional, one-dimensional, or two-dimensional.</td>
      </tr>
      <tr>
      <td>When the dimension of indices is 0, the 0th axis of updates is not 1.</td>
      </tr>
      <tr>
      <td>data, indices, and updates are empty tensors.</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceScatterUpdate

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnInplaceScatterUpdateGetWorkspaceSize.</td>
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
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - **aclnnInplaceScatterUpdate** defaults to a deterministic implementation.

- The 0th axis of the updates shape must be consistent with that of the indices shape.
- If indices is zero-dimensional, the 0th axis of the updates shape must be 1.
- The 0th axis of the updates shape must be less than or equal to that of the data shape.
- The shapes of updates and data are the same except for the axis and 0th axis.
- When the indices shape is two-dimensional, the 1st axis of the shape must be 2.
- If the data type of indices is INT32, DtypeSize is 4. If the data type of indices is INT64, DtypeSize is 8. IndicesShapeSize is the product of the indices shape. The required UB is calculated as follows: UB = IndicesShapeSize x DtypeSize + 224. If the required UB size is greater than the total UB size of the corresponding AI processor version, the operation is not supported.
- If indices contains duplicates, the output at those positions is non-deterministic.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_update.h"

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
  // (Boilerplate) Initialize resources.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  int64_t axis = -2;
  std::vector<int64_t> selfRefShape = {1, 1, 2, 8};
  std::vector<int64_t> indicesShape = {1};
  std::vector<int64_t> updatesShape = {1, 1, 1, 8};
  void* selfRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  std::vector<float> selfRefHostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> indicesHostData = {1};
  std::vector<float> updatesHostData = {3, 3, 3, 3, 3, 3, 3, 3};

  // Create a selfRef aclTensor.
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
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
  // Call the first-phase API of aclnnInplaceScatterUpdate.
  ret = aclnnInplaceScatterUpdateGetWorkspaceSize(selfRef, indices, updates, axis, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceScatterUpdate.
  ret = aclnnInplaceScatterUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(selfRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);

  // 7. Release device resources.
  aclrtFree(selfRefDeviceAddr);
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
