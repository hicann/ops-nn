# aclnnMaxUnpool2d

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                         |    √ |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √   |
| <term>Atlas A2 training products/Atlas A2 inference products</term>    |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×  |

## Function

- Description: Performs the two-dimensional inverse operation of [aclnnMaxPool](../../../pooling/max_pool_v3/docs/aclnnMaxPool_en.md). **outputSize** determines the size of **out** along the H and W axes, fills elements of **self** in **out** based on **indices**, and sets other positions to **0**.

- Formula:

  $$
  out[n][c][indices[n][c][i]] = self[n][c][i]
  $$

  **out**, **indices**, and **self** are obtained by reshaping the axis combined from the last two axes, that is, i ∈ [0, H*W).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMaxUnpool2dGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMaxUnpool2d** is called to perform computation.

```Cpp
aclnnStatus aclnnMaxUnpool2dGetWorkspaceSize(
  const aclTensor*     self,
  const aclTensor*     indices, 
  const aclIntArray*   outputSize,
  aclTensor*           out, 
  uint64_t*            workspaceSize, 
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnMaxUnpool2d(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnMaxUnpool2dGetWorkspaceSize

- **Parameters:**

  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 165px">
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 115px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Discontinuous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Self in the formula.</td>
      <td class="tg-0pky">
        <ul>
          <li>The data type is the same as that of out.</li>
          <li>The shape is the same as that of indices.</li>
          <li>When the dimension is 3, the dimensions represent C, H, and W in sequence. When the dimension is 4, the dimensions represent N, C, H, and W in sequence. C indicates the channel, N indicates the batch, H indicates the height, and W indicates the width.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, DOUBLE</td>
      <td class="tg-0pky">ND, NCHW</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">indices (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">indices in the formula, indicating the index position of the input self element in the output result.</td>
      <td class="tg-0pky">
        <ul>
          <li>The shape is the same as that of self.</li>
          <li>When the dimension is 3, the dimensions represent C, H, and W in sequence. When the dimension is 4, the dimensions represent N, C, H, and W in sequence. C indicates the channel, N indicates the batch, H indicates the height, and W indicates the width.</li>
        </ul>
      </td>
      <td class="tg-0pky">INT64, INT32</td>
      <td class="tg-0pky">ND, NCHW</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">outputSize (aclIntArray*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Size of the output result in the H and W dimensions.</td>
      <td class="tg-0pky">The size is 2. The product of the two elements must be greater than or equal to the product of the sizes of self in the H and W dimensions.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">out in the formula.</td>
      <td class="tg-0pky">
        <ul>
          <li>The data type is the same as that of self.</li>
          <li>When the dimension is 3, the dimensions are C, H, and W in sequence. When the dimension is 4, the dimensions are N, C, H, and W in sequence. C indicates the channel, N indicates the batch, H indicates the height, and W indicates the width.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, DOUBLE</td>
      <td class="tg-0pky">ND, NCHW</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the size of the workspace to be allocated on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the operator executor, including the operator execution process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1147px"><colgroup>
  <col style="width: 286px">
  <col style="width: 123px">
  <col style="width: 738px">
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
      <td>The input self, indices, outputSize, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data types of self and indices are not supported.</td>
    </tr>
    <tr>
      <td>The data types of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of self is not 3 or 4.</td>
    </tr>
    <tr>
      <td>The shapes of self and indices are inconsistent.</td>
    </tr>
    <tr>
      <td>The size of outputSize is not 2.</td>
    </tr>
    <tr>
      <td>The product of the two elements of outputSize is less than the product of the sizes of self in the H and W dimensions.</td>
    </tr>
    <tr>
      <td>The size of out in the N and C dimensions is not completely the same as that of self.</td>
    </tr>
    <tr>
      <td>The size of out in the H and W dimensions is not equal to the two element values in outputSize.</td>
    </tr>
    <tr>
      <td>The product of outputSize is less than the product of the H and W axes of self. </td>
    </tr>
  </tbody>
  </table>

## aclnnMaxUnpool2d

- **Parameters:**
  
  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnLogAddExpGetWorkspaceSize.</td>
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
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnMaxUnpool2d** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool2d.h"

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
  std::vector<int64_t> selfShape = {1, 1, 2, 2};
  std::vector<int64_t> outShape = {1, 1, 4, 4};
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> outHostData = {0, 0, 0, 0.0, 0, 0, 0, 0,
                                    0, 0, 0, 0.0, 0, 0, 0, 0};
  std::vector<int64_t> indicesHostData = {3, 8, 11, 13};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, selfShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> arraySize = {4, 4};
  const aclIntArray *outputSize = aclCreateIntArray(arraySize.data(), arraySize.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMaxUnpool2d.
  ret = aclnnMaxUnpool2dGetWorkspaceSize(self, indices, outputSize, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMaxUnpool2d.
  ret = aclnnMaxUnpool2d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool2d failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
