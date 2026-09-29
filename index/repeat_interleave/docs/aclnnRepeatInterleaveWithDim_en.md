# aclnnRepeatInterleaveWithDim

📄 [View Source Code](https://gitcode.com/cann/ops-nn/tree/master/index/repeat_interleave)

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

- This API is used to repeat each element in the tensor for the number of times specified by the corresponding element in the repeats tensor along the specified dimension.

- Example: Assume that the input tensor is [[a, b], [c, d], [e, f]]. **repeats** is ([1, 2, 3]), and **dim** is 0.
  In this case, the generated tensor is ([[a, b], [c, d], [c, d], [e, f], [e, f], [e, f]]).
  In the dimension with dim = 0, a and b are repeated once, c and d are repeated twice, and e and f are repeated three times.

  Assume that the input tensor is ([[a, b], [c, d], [e, f]]). **repeats** is ([2]), and **dim** is 0.
  In this case, the generated tensor is [ [a, b], [a, b], [c, d], [c, d], [e, f], [e, f]].
  In the dimension with dim = 0, a and b are repeated twice, c and d are repeated twice, and e and f are repeated twice.
  Note: This scenario is equivalent to **repeats** being (2).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnRepeatInterleaveWithDimGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnRepeatInterleaveWithDim** is called to perform computation.

```Cpp
aclnnStatus aclnnRepeatInterleaveWithDimGetWorkspaceSize(
  const aclTensor* self,
  const aclTensor* repeats,
  int64_t          dim,
  int64_t          outputSize,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnRepeatInterleaveWithDim(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnRepeatInterleaveWithDimGetWorkspaceSize

- **Parameters:**

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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>Input tensor to be replicated in the description of the function.</td>
        <td>Empty tensors are supported.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, BFLOAT16, FLOAT</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>repeats (aclTensor*) </td>
        <td>Input</td>
        <td>Number of repetitions.</td>
        <td>Empty tensors are supported.<br>The value must be a 0D or 1D tensor.<br>For a one-dimensional tensor, the size of **repeats** must be 1 or equal to the size of the **dim** dimension of **self**.</td>
        <td>INT32 or INT64</td>
        <td>ND</td>
        <td>0-1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>Dimension to be repeated.</td>
        <td>The value range is [–self.dim(), self.dim() – 1].</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outputSize (int64_t) </td>
        <td>Input</td>
        <td>Final size of the repeated dim dimension.</td>
        <td>If repeats contains multiple values, the value of outputSize must be the sum of repeats.<br>If **repeats** contains only one element, the value of **outputSize** must be equal to **repeats** multiplied by the size of the **dim** dimension of **self**.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Output tensor when data replication is complete. For details, see the function description.</td>
        <td>The data type must be the same as that of self.<br>If repeats contains multiple values, the size of the out shape in the dim dimension is equal to the sum of all elements in repeats.<br>If repeats contains only one element, the size of the out shape in the dim dimension is equal to the product of repeats and the size of the dim dimension of self.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, BFLOAT16, FLOAT</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

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
        <td>The self, repeats, or out pointer is null. </td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data types of self and repeats are not supported.</td>
      </tr>
      <tr>
        <td>The data types of self and out are different.</td>
      </tr>
      <tr>
        <td>repeats is not a 0D or 1D tensor.</td>
      </tr>
      <tr>
        <td>The value of dim is not within the range of [–Number of dimensions of self, Number of dimensions of self – 1].</td>
      </tr>
      <tr>
        <td>When repeats is a 1D tensor, the size of repeats is not 1 and is not the size of the dimension specified by dim of self.</td>
      </tr>
      <tr>
        <td>The number of dimensions of self exceeds 8.</td>
      </tr>
      <tr>
        <td>When self is a 0D tensor, dim cannot be passed.</td>
      </tr>
    </tbody></table>

## aclnnRepeatInterleaveWithDim

- **Parameters:**

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
        <td>Workspace size allocated on the device, which is obtained by the first API aclnnRepeatInterleaveWithDimGetWorkspaceSize.</td>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnRepeatInterleaveWithDim** defaults to a deterministic implementation.
- Input shape restriction: repeats can only be a 0D or 1D tensor. For a one-dimensional tensor, the size of **repeats** must be 1 or equal to the size of the **dim** dimension of **self**.
- Input value range restriction: The value in the repeats tensor must be a natural number.
- Other restrictions: The value of outputSize must be calculated as follows: When there is only one element in repeats, outputSize = size of the dimension of self * value of repeats. When there are multiple values in repeats, outputSize = sum of the values in repeats.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat_interleave.h"

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
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> repeatsShape = {2};
  std::vector<int64_t> outShape = {3, 3};
  void* selfDeviceAddr = nullptr;
  void* repeatsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* repeats = nullptr;
  aclTensor* out = nullptr;
  int64_t dim = 0;
  int64_t output_size = 3;
  std::vector<int64_t> selfHostData = {3, 4, 5, -3, -4, -5};
  std::vector<int64_t> repeatsHostData = {1, 2};
  std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT64, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a repeats aclTensor.
  ret = CreateAclTensor(repeatsHostData, repeatsShape, &repeatsDeviceAddr, aclDataType::ACL_INT64, &repeats);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnRepeatInterleaveWithDim.
  ret = aclnnRepeatInterleaveWithDimGetWorkspaceSize(self, repeats, dim, output_size, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveWithDimGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRepeatInterleaveWithDim.
  ret = aclnnRepeatInterleaveWithDim(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveWithDim failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(repeats);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(repeatsDeviceAddr);
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
