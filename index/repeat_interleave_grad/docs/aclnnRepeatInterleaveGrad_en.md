# aclnnRepeatInterleaveGrad

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/index/repeat_interleave_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                         |    √  |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>    |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×   |

## Function

  - Function: This API is the reverse of the repeatInterleave operator. It performs ReduceSum on the axis dimension of the yGrad tensor based on the repeats.

  - Example:
    Assume that the ([[a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>, d<sub>1</sub>, e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub>, c<sub>2</sub>, d<sub>2</sub>, e<sub>2</sub>, f<sub>2</sub>]]), repeats of tensor yGrad is [1, 2, 2, 1] and the axis is 1.
    The final generated tensor is tensor([[a<sub>1</sub>, b<sub>1</sub> + c<sub>1</sub>, d<sub>1</sub> + e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub> + c<sub>2</sub>, d<sub>2</sub> + e<sub>2</sub>, f<sub>2</sub>]]). ReduceSum is performed on the axis of tensor yGrad based on repeats.

    Assume that the tensor yGrad is ([[a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>, d<sub>1</sub>, e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub>, c<sub>2</sub>, d<sub>2</sub>, e<sub>2</sub>, f<sub>2</sub>]]), repeats ([2]) and the axis is 1.
    The final generated tensor is tensor([[a<sub>1</sub> + b<sub>1</sub>, c<sub>1</sub> + d<sub>1</sub>, e<sub>1</sub> + f<sub>1</sub>, , [a<sub>2</sub> + b<sub>2</sub>, c<sub>2</sub> + d<sub>2</sub>, e<sub>2</sub> + f<sub>2</sub>]]]). ReduceSum is performed on every two axes of tensor yGrad based on the value of repeats.
    Note: This scenario is equivalent to **repeats** being (2).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnRepeatInterleaveGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnRepeatInterleaveGrad` is called to perform computation.

 ```cpp
  aclnnStatus aclnnRepeatInterleaveGradGetWorkspaceSize(
    const aclTensor *yGrad, 
    const aclTensor *repeats, 
    int64_t          axis, 
    const aclTensor *out, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
  ```

  ```cpp
  aclnnStatus aclnnRepeatInterleaveGrad(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
  ```

## aclnnRepeatInterleaveGradGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1714px"><colgroup>
  <col style="width: 138px">
  <col style="width: 120px">
  <col style="width: 304px">
  <col style="width: 424px">
  <col style="width: 291px">
  <col style="width: 132px">
  <col style="width: 160px">
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
      <th>Non-contiguous tensor Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>yGrad</td>
      <td>Input</td>
      <td>Input tensor to be reduced by Sum.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>repeats</td>
      <td>Input</td>
      <td>Number of repetitions.</td>
      <td>repeats must be a 0D or 1D tensor. If repeats is a 1D tensor and size is 1, then repeats supports broadcasting. If repeats is a 1D tensor and size is greater than 1, then the sum of elements in repeats is equal to the number of dimensions of yGrad. Empty tensors are not supported.</td>
      <td>INT32 or INT64</td>
      <td>ND</td>
      <td></td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>Dimension on which ReduceSum is performed.</td>
      <td>The value range of axis is [–n, n), where n is the number of dimensions of yGrad.</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor completed by ReduceSum in the function description.</td>
      <td>For details about the shape restrictions, see the restrictions.</td>
      <td>Same as yGrad</td>
      <td>ND</td>
      <td>-</td>
      <td></td>
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

  - Ascend 950PR/Ascend 950DT: The data type of yGrad can be FLOAT16, BFLOAT16, or FLOAT.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1039px"><colgroup>
  <col style="width: 292px">
  <col style="width: 138px">
  <col style="width: 609px">
  </colgroup>
  <thead>
    <tr>
      <th>561002</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input self, repeats, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of self and repeats are not supported.</td>
    </tr>
    <tr>
      <td>The data types of self and out are different.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>When self is not an empty tensor but repeats is an empty tensor.</td>
    </tr>
  </tbody>
  </table>

## aclnnRepeatInterleaveGrad

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1045px"><colgroup>
  <col style="width: 141px">
  <col style="width: 110px">
  <col style="width: 794px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnRepeatInterleaveGradGetWorkspaceSize.</td>
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
  - The aclnnRepeatInterleaveGrad is implemented in deterministic mode by default.

The following conditions must be met during computation:

  - If repeats is a 0D tensor or a 1D tensor with a size of 1, the element value of repeats must be a divisor of the dimension of yGrad on axis.
    If repeats is a 1D tensor with a size greater than 1, the sum of the elements of repeats must be the dimension of yGrad on axis.
    The value in the **repeats** tensor must be a natural number.
  - The shape of out must be the same as that of yGrad after ReduceSum is performed on the axis.
    For example, if the shape of yGrad is [64], the value of repeat is [2], and the value of axis is 0, the shape of out can be [32], [2, 16], or [2, 4, 4], as long as the shape of out is 32.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat_interleave_grad.h"

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
  std::vector<int64_t> yGradShape = {4, 3};
  std::vector<int64_t> repeatsShape = {2};
  std::vector<int64_t> outShape = {2, 3};
  void* yGradDeviceAddr = nullptr;
  void* repeatsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* yGrad = nullptr;
  aclTensor* repeats = nullptr;
  aclTensor* out = nullptr;
  int64_t axis = 0;
  std::vector<float> yGradHostData = {3, 4, 5, 3, 4, 5, -3, -4, -5, -3, -4, -5};
  std::vector<int64_t> repeatsHostData = {2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};

  // Create a yGrad aclTensor.
  ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_FLOAT, &yGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a repeats aclTensor.
  ret = CreateAclTensor(repeatsHostData, repeatsShape, &repeatsDeviceAddr, aclDataType::ACL_INT64, &repeats);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first part of the aclnnRepeatInterleaveGrad API.
  ret = aclnnRepeatInterleaveGradGetWorkspaceSize(yGrad, repeats, axis, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second part of the aclnnRepeatInterleaveGrad API.
  ret = aclnnRepeatInterleaveGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveGrad failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensors and aclScalars. Modify the code based on the API definition.
  aclDestroyTensor(yGrad);
  aclDestroyTensor(repeats);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(yGradDeviceAddr);
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
