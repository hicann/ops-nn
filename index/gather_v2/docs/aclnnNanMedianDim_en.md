# aclnnNanMedianDim

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/gather_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                         |    √  |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √   |

## Function

  - Function: After NANs are ignored, the median and position of the specified dimension of the tensor are returned.

  - Example:
    - Example 1:

      ```text
      If keepDim is set to True, the size of the corresponding dimension is set to 1. If keepDim is set to False, the corresponding dimension is deleted.
      If the shape of self is [2, 3, 4], dim = 1, and keepDim is true, then the output shape is [2, 1, 4].
      If the shape of self is [2, 3, 4], dim = 1, and keepDim is false, then the output shape is [2, 4].
      ```

    - Example 2:

      ```text
      Example of the output shape.
      If input
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) with shape [2, 4],
      dim = 0,
      keepDim = true,
      then output
      valuesOut = tensor([[-1., float('nan'), 3., 2.]]) with shape [1, 4],
      indicesOut = tensor([[1, 0, 0, 0]]) with shape [1, 4].
      ```

    - Example 3:

      ```text
      If input
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) with shape [2, 4],
      dim = 0,
      keepDim = false,
      then output
      valuesOut = tensor([-1., float('nan'), 3., 2.]) with shape [4],
      indicesOut = tensor([1, 0, 0, 0]) with shape [4].
      ```
      
    - Example 4:
    
      ```text
      If input
      self = tensor([[1, float('nan'), 3, 2],[-1, float('nan'), 3, 2]]) with shape [2, 4],
      dim = 1,
      keepDim = false,
      then output
      valuesOut = tensor ([2, 2]) with shape [2],
      indicesOut = tensor ([3, 3]) with shape [2].
      ```
    
## Prototype

  Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnNanMedianDimGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnNanMedianDim** is called to perform computation.

  ```cpp
  aclnnStatus aclnnNanMedianDimGetWorkspaceSize(
    const aclTensor* self, 
    int64_t          dim, 
    bool             keepDim, 
    aclTensor*       valuesOut, 
    aclTensor*       indicesOut, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
  ```
  
  ```cpp
  aclnnStatus aclnnNanMedianDim(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
  ```

## aclnnNanMedianDimGetWorkspaceSize

  - **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1800px"><colgroup>
    <col style="width: 140px">
    <col style="width: 127px">
    <col style="width: 268px">
    <col style="width: 418px">
    <col style="width: 387px">
    <col style="width: 134px">
    <col style="width: 171px">
    <col style="width: 155px">
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
        <td>self</td>
        <td>Input</td>
        <td></td>
        <td>-</td>
        <td>FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, BFLOAT16.</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Specified dimension</td>
        <td>The value range is [–self.dim(), self.dim() – 1].</td>
        <td></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepDim</td>
        <td>Input</td>
        <td>Whether to retain the dimensions of the input tensor in the output tensor</td>
        <td>If true, the size of the corresponding dimension is set to 1. If false, the corresponding dimension is deleted.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>valuesOut</td>
        <td>Output</td>
        <td>Median value</td>
        <td>If keepDim is true, the shape must be the same as the shape of self except the size of the specified dimension, and the size of the specified dimension is 1. If keepDim is false, the shape must be the same as the shape of self except the specified dimension. Non-contiguous tensors are supported. The data format can be ND.</td>
        <td>FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, BFLOAT16.</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indicesOut</td>
        <td>Output</td>
        <td>Index of the median</td>
        <td></td>
        <td>INT64</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
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

    - <term>Atlas training products</term>: The data type cannot be BFLOAT16.

  - **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter verification. The following errors may be thrown:
      
    <table style="undefined;table-layout: fixed; width: 1415px"><colgroup>
    <col style="width: 314px">
    <col style="width: 161px">
    <col style="width: 940px">
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
        <td>The input self, valuesOut, or indicesOut is a null pointer. </td>
      </tr>
      <tr>
        <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="10">161002</td>
        <td>The data type of self, valuesOut, or indicesOut is not supported.</td>
      </tr>
      <tr>
        <td>The data types of self and valuesOut are different.</td>
      </tr>
      <tr>
        <td>The value of dim is out of the range [–self.dim(), self.dim() – 1].</td>
      </tr>
      <tr>
        <td>The number of dimensions of self, valuesOut, or indicesOut exceeds 8.</td>
      </tr>
      <tr>
        <td>The size of the dimension specified by dim in self cannot be 0.</td>
      </tr>
      <tr>
        <td>When keepDim is true, the number of dimensions of valuesOut or indicesOut is inconsistent with that of self.</td>
      </tr>
      <tr>
        <td>When keepDim is false, the number of dimensions of valuesOut or indicesOut is not 1 less than that of self.</td>
      </tr>
      <tr>
        <td>When keepDim is true, the shape of valuesOut or indicesOut is inconsistent with that of self in terms of the size of the dimension except dim.</td>
      </tr>
      <tr>
        <td>When keepDim is true, the size of the dimension specified by dim in valuesOut or indicesOut is not 1.</td>
      </tr>
      <tr>
        <td>When keepDim is false, the shape of valuesOut or indicesOut is inconsistent with that of self in terms of the size of the dimension except dim.</td>
      </tr>
    </tbody></table>

## aclnnNanMedianDim

  - **Parameters**
      
  <table style="undefined;table-layout: fixed; width: 1042px"><colgroup>
  <col style="width: 141px">
  <col style="width: 110px">
  <col style="width: 791px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnNanMedianDimGetWorkspaceSize API.</td>
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

  - If the data type of **self** is not FLOAT, FLOAT16, or BFLOAT16, the operator execution may time out due to an overlarge tensor size (an AI CPU error is reported, with reason=[aicpu timeout]). The maximum size for each data type (closely related to the remaining memory of the machine) is as follows:
    - INT64: 150000000
    - UINT8, INT8, INT16, INT32: 725000000

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_median.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> valuesOutShape = {2};
  std::vector<int64_t> indicesOutShape = {2};
  void* selfDeviceAddr = nullptr;
  void* valuesOutDeviceAddr = nullptr;
  void* indicesOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valuesOut = nullptr;
  aclTensor* indicesOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, NAN};
  std::vector<float> valuesOutHostData = {0, 0};
  std::vector<int64_t> indicesOutHostData = {0, 0};
  int64_t dim = 0;
  bool keepDim = false;
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a valuesOut aclTensor.
  ret = CreateAclTensor(valuesOutHostData, valuesOutShape, &valuesOutDeviceAddr, aclDataType::ACL_FLOAT, &valuesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an indicesOut aclTensor.
  ret = CreateAclTensor(indicesOutHostData, indicesOutShape, &indicesOutDeviceAddr, aclDataType::ACL_INT64, &indicesOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnNanMedianDim.
  ret = aclnnNanMedianDimGetWorkspaceSize(self, dim, keepDim, valuesOut, indicesOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNanMedianDimGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnNanMedianDim.
  ret = aclnnNanMedianDim(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNanMedianDim failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(valuesOutShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valuesOutDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("valuesOut[%ld] is: %f\n", i, resultData[i]);
  }

  std::vector<int64_t> indicesData(size, 0);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indicesOutDeviceAddr,
                    size * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("indicesOut[%ld] is: %ld\n", i, indicesData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(valuesOut);
  aclDestroyTensor(indicesOut);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(valuesOutDeviceAddr);
  aclrtFree(indicesOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
