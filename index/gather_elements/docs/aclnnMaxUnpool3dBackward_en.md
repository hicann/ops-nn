# aclnnMaxUnpool3dBackward

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

- Description: Performs backpropagation of the MaxPool3d inverse operation ([aclnnMaxUnpool3d](../../scatter_elements/docs/aclnnMaxUnpool3d_en.md)), writing element values of **gradOutput** in **out** based on **indices**.
- Formula:
  - When the input is four-dimensional, the dimensions are (N, D, H, W):
  $$
  out[N][i] = gradOutput[N][indices[N][i]]
  $$

  - When the input is five-dimensional, the dimensions are (N, C, D, H, W):
  $$
  out[N][C][i] = gradOutput[N][C][indices[N][C][i]]
  $$
  out, gradOutput, and indices are obtained by reshaping the last two axes into one axis, and i ∈ [0, D*H* W) is used.

## Prototype

  Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMaxUnpool3dBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMaxUnpool3dBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnMaxUnpool3dBackwardGetWorkspaceSize(
  const aclTensor*     gradOutput,
  const aclTensor*     self,
  const aclTensor*     indices, 
  const aclIntArray*   outputSize,
  const aclIntArray*   stride,
  const aclIntArray*   padding,
  aclTensor*           out, 
  uint64_t*            workspaceSize, 
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnMaxUnpool3dBackward(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnMaxUnpool3dBackwardGetWorkspaceSize

- **Parameters**

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
      <th class="tg-0pky">Non-consecutive Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">gradOutput (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">gradOutput in the formula.</td>
      <td class="tg-0pky">
        <ul>
          <li>The data type must be the same as that of self and out.</li>
          <li>The dimensions are (N, outputSize[0], outputSize[1], outputSize[2]) or (N, C, outputSize[0], outputSize[1], outputSize[2]).</li>
          <li>When the number of dimensions is 4, the dimensions are N, D, H, and W in sequence. When the number of dimensions is 5, the dimensions are N, C, D, H, and W in sequence.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, DOUBLE</td>
      <td class="tg-0pky">ND, NCDHW</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">self (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">
        <ul>
          <li>The data type must be the same as that of gradOutput and out.</li>
          <li>The dimensions are (N, D, H, W) or (N, C, D, H, W).</li>
          <li>The dimensions must be the same as those of gradOutput, and the shape must be the same as those of indices and out.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, DOUBLE</td>
      <td class="tg-0pky">ND, NCDHW</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">indices (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">indices in the formula indicates the index position of the element in gradOutput in the output result.</td>
      <td class="tg-0pky">The shape must be the same as that of self.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">ND, NCDHW</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">outputSize (aclIntArray*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the spatial size of the output result in the D, H, and W dimensions.</td>
      <td class="tg-0pky">The size is 3.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">stride (aclIntArray*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the stride size of the max pooling window in the D, H, and W dimensions.</td>
      <td class="tg-0pky">The size is 3.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">padding (aclIntArray*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Indicates the padding value of the max pooling window in the D, H, and W dimensions.</td>
      <td class="tg-0pky">The size is 3.</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">out (aclTensor*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Output in the formula.</td>
      <td class="tg-0pky">
        <ul>
          <li>The data type is the same as that of gradOutput and self.</li>
          <li>The shape must be the same as that of self and indices.</li>
        </ul>
      </td>
      <td class="tg-0pky">FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT8, DOUBLE</td>
      <td class="tg-0pky">ND, NCDHW</td>
      <td class="tg-0pky">4-5</td>
      <td class="tg-0pky">×</td>
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

- **Returns**

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
      <td>The input gradOutput, self, indices, outputSize, stride, padding, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="14">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="14">161002</td>
      <td>The out tensor is discontinuous.</td>
    </tr>
    <tr>
      <td>The data type of gradOutput, self, indices, or out is not supported.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput, self, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimension of self is not 4D or 5D.</td>
    </tr>
    <tr>
      <td>The dimensions of self, indices, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The shapes of self, indices, and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The size of each dimension of self except the N dimension is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The size of outputSize, stride, or padding is not equal to 3.</td>
    </tr>
    <tr>
      <td>The value of outputSize or stride is not greater than 0.</td>
    </tr>
    <tr>
      <td>The product of the three elements of outputSize is less than the product of the sizes of self in the D, H, and W dimensions.</td>
    </tr>
    <tr>
      <td>The size of gradOutput in the D, H, and W dimensions is not equal to the three elements of outputSize.</td>
    </tr>
    <tr>
      <td>The dimensions of gradOutput and self are inconsistent.</td>
    </tr>
    <tr>
      <td>When self is 4D, the size of gradOutput and self in the N dimension is inconsistent.</td>
    </tr>
    <tr>
      <td>When self is 5D, the size of gradOutput and self in the C or N dimension is inconsistent.</td>
    </tr>
  </tbody>
  </table>

## aclnnMaxUnpool3dBackward

- **Parameters**

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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnMaxUnpool3dBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool3d_backward.h"

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
  std::vector<int64_t> selfShape = {1, 1, 4, 4};
  std::vector<int64_t> indicesShape = {1, 1, 4, 4};
  std::vector<int64_t> gradShape = {1, 1, 4, 4};
  std::vector<int64_t> outShape = {1, 1, 4, 4};
  void* gradDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> gradHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData = {0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0};
  std::vector<int64_t> indicesHostData = {0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 11, 0, 0, 0, 13};
  // Create a grad aclTensor.
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> arraySize1 = {1, 4, 4};
  const aclIntArray *outputSize = aclCreateIntArray(arraySize1.data(), arraySize1.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  std::vector<int64_t> arraySize2 = {1, 2, 3};
  const aclIntArray *stride = aclCreateIntArray(arraySize2.data(), arraySize2.size());
  CHECK_RET(stride != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  const aclIntArray *padding = aclCreateIntArray(arraySize2.data(), arraySize2.size());
  CHECK_RET(padding != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnMaxUnpool3dBackward.
  ret = aclnnMaxUnpool3dBackwardGetWorkspaceSize(grad, self, indices, outputSize, stride, padding, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMaxUnpool3dBackward.
  ret = aclnnMaxUnpool3dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr, size * sizeof(outData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(grad);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. Release device resources.
  aclrtFree(gradDeviceAddr);
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
