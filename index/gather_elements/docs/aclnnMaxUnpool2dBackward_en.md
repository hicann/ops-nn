# aclnnMaxUnpool2dBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/gather_elements)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                         |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>    |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    √    |

## Function

- Description: Performs backpropagation of the MaxPool2d inverse operation ([aclnnMaxUnpool2d](../../scatter_elements/docs/aclnnMaxUnpool2d_en.md)), writing element values of **gradOutput** in **out** based on **indices**.

- Formula:

  $$
  out[n][c][i] = gradOutput[n][c][indices[n][c][i]]
  $$

  **out**, **gradOutput**, and **indices** are obtained by reshaping the axis combined from the last two axes, that is, $i ∈ [0, H*W)$. **H** and **W** represent the last two axes.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMaxUnpool2dBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMaxUnpool2dBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnMaxUnpool2dBackwardGetWorkspaceSize(
  const aclTensor*     gradOutput,
  const aclTensor*     self, 
  const aclTensor*     indices, 
  const aclIntArray*   outputSize,
  aclTensor*           out, 
  uint64_t*            workspaceSize, 
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnMaxUnpool2dBackward(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnMaxUnpool2dBackwardGetWorkspaceSize

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
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">gradOutput (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">gradOutput in the formula.</td>
      <td class="tg-0pky">
        <ul>
          <li>The data type must be convertible to out (see <a href="../../../docs/en/context/conversion_relationship.md" target="_blank">conversion relationship</a>).</li>
          <li>The data type must be the same as that of self.</li>
          <li>The shape must be (N, C, outputSize[0], outputSize[1]) or (C, outputSize[0], outputSize[1]).</li>
          <li>The dimension must be the same as that of self.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">self (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">
        <ul>
          <li>The data type must be the same as that of gradOutput.</li>
          <li>The shape must be (N, C, H, W) or (C, H, W), and the shape must be consistent with indices.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">indices (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">indices in the formula.</td>
      <td class="tg-0pky">The shape must be consistent with self.</td>
      <td class="tg-0pky">INT64, INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">outputSize (aclIntArray*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">The number of elements must be 2, and the element values must be consistent with the last two dimensions of the shape of gradOutput.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output in the formula.</td>
      <td class="tg-0pky">The shape must be the same as that of self.</td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be allocated on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the operator executor, which contains the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

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
      <td>The input gradOutput, self, indices, outputSize, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>gradOutput, self, indices, or out is not within the supported data types.</td>
    </tr>
    <tr>
      <td>gradOutput cannot be converted to the data type of out.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput and self are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of self is not 3D or 4D.</td>
    </tr>
    <tr>
      <td>The dimension of gradOutput is inconsistent with that of self.</td>
    </tr>
    <tr>
      <td>When self is 3D, the sizes of self and gradOutput in the C dimension are inconsistent.</td>
    </tr>
    <tr>
      <td>When self is 4D, the sizes of self and gradOutput in the N and C dimensions are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of self, indices, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The number of elements in outputSize is not 2.</td>
    </tr>
    <tr>
      <td>The sizes of the last two dimensions of gradOutput are inconsistent with the elements of outputSize.</td>
    </tr>
    <tr>
      <td>self and indices are not empty tensors, and gradOutput is an empty tensor.</td>
    </tr>
  </tbody>
  </table>

## aclnnMaxUnpool2dBackward

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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnLogAddExpGetWorkspaceSize API.</td>
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
  - **aclnnMaxUnpool2dBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool2d_backward.h"

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
  int64_t N = 1;
  int64_t C = 3;
  int64_t H = 2;
  int64_t W = 2;
  std::vector<int64_t> outputSizeData = {3, 1};
  std::vector<int64_t> gradOutputShape = {N, C, outputSizeData[0], outputSizeData[1]};
  std::vector<int64_t> selfShape = {N, C, H, W};
  std::vector<int64_t> indicesShape = {N, C, H, W};
  std::vector<int64_t> outShape = {N, C, H, W};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  aclIntArray* outputSize = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> selfHostData(12, 1);
  std::vector<int32_t> indicesHostData = {0, 1, 2, 1, 2, 0, 1, 1, 1, 0, 0, 0};
  std::vector<float> outHostData(12, 0);
  // Create a gradOutput aclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an outputSize aclIntArray.
  outputSize = aclCreateIntArray(outputSizeData.data(), 2);
  CHECK_RET(outputSize != nullptr, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMaxUnpool2dBackward.
  ret = aclnnMaxUnpool2dBackwardGetWorkspaceSize(gradOutput, self, indices, outputSize, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMaxUnpool2dBackward.
  ret = aclnnMaxUnpool2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(gradOutputDeviceAddr);
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
