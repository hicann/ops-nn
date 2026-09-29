# aclnnUnique2

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                         |    √  |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×   |

## Function

Removes duplicate elements from the input tensor self and returns the unique elements in self. This is an enhanced unique function, with a new return value countsOut indicating the number of occurrences for where elements in the input self map to in valueOut, controlled by the returnCounts parameter.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnUnique2GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnUnique2** is called to perform computation.

```cpp
aclnnStatus aclnnUnique2GetWorkspaceSize(
  const aclTensor* self, 
  bool             sorted, 
  bool             returnInverse, 
  bool             returnCounts, 
  aclTensor*       valueOut, 
  aclTensor*       inverseOut, 
  aclTensor*       countsOut, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnUnique2(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnUnique2GetWorkspaceSize

* **Parameters**

  <table style="undefined;table-layout: fixed; width: 1403px"><colgroup>
  <col style="width: 128px">
  <col style="width: 122px">
  <col style="width: 299px">
  <col style="width: 171px">
  <col style="width: 292px">
  <col style="width: 106px">
  <col style="width: 136px">
  <col style="width: 149px">
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
      <td>self</td>
      <td>Input</td>
      <td>Tensor to be deduplicated.</td>
      <td>-</td>
      <td>BOOL, FLOAT, FLOAT16, DOUBLE, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, BFLOAT16</td>
      <td>ND</td>
      <td>1~8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sorted</td>
      <td>Input</td>
      <td>Whether to sort valueOut in ascending order.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnInverse</td>
      <td>Input</td>
      <td>Whether to return the index of each element in the input data in valueOut.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnCounts</td>
      <td>Input</td>
      <td>Whether to return the number of each unique element in valueOut in the original input tensor.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valueOut</td>
      <td>Output</td>
      <td>First output tensor, which stores the unique elements in the input tensor. The shape supports only one dimension, and the number of elements is the same as that of self.</td>
      <td>-</td>
      <td>BOOL, FLOAT, FLOAT16, DOUBLE, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inverseOut</td>
      <td>Output</td>
      <td>Second output tensor. This parameter is valid only when returnInverse is set to True or returnCounts is set to True. It returns the index of each element in self in valueOut. The shape is the same as that of self.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>countsOut</td>
      <td>Output</td>
      <td>Third output tensor. This parameter is valid only when returnCounts is set to True. It returns the number of times that each element in valueOut appears in self. The shape is the same as that of valueOut.</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
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

* **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:
        
    <table style="undefined;table-layout: fixed; width: 1015px"><colgroup>
    <col style="width: 257px">
    <col style="width: 101px">
    <col style="width: 657px">
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
        <td>The input self, valueOut, inverseOut, or countsOut is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="4">161002</td>
        <td>The data type of self or valueOut is not supported.</td>
      </tr>
      <tr>
        <td>self is a non-contiguous tensor.</td>
      </tr>
      <tr>
        <td>returnInverse is True, and the shape of inverseOut is inconsistent with that of self.</td>
      </tr>
      <tr>
        <td>returnCounts is True, and countsOut and valueOut have different shapes.</td>
      </tr>
    </tbody>
    </table>
  
## aclnnUnique2

* **Parameters**

  <table style="undefined;table-layout: fixed; width: 950px"><colgroup>
  <col style="width: 122px">
  <col style="width: 105px">
  <col style="width: 723px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnUnique2GetWorkspaceSize.</td>
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
 
* **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnUnique2** defaults to a deterministic implementation.

  * Ascend 950PR/Ascend 950DT:
      * Due to differences in the implementation of the deduplication algorithm, the operator ignores the value of the sorted input parameter and sorts the output results in ascending order when all of the following conditions are met:
          * The self input is 1D.
          * The data type of self is FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, or BFLOAT16.
      * Due to differences in the implementation of the deduplication algorithm, the inverseOut output of the operator is meaningless when all of the following conditions are met:
          - The returnInverse input is false.
          - The self input is 1D.
          - The data type of self is FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, or BFLOAT16.
  * For <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>, if the input self contains 0, the output of the operator may contain both positive 0 and negative 0, instead of only one 0.
  * For <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>, when the data volume of self exceeds 200 million, the execution time is long and the operation may time out.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_unique2.h"
  
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
  // 1. (Required) Initialize the device and stream. For details, see the AscendCL API Reference.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output according to the API.
  std::vector<int64_t> selfShape = {8};
  std::vector<int64_t> valueShape = {8};
  std::vector<int64_t> inverseShape = {8};
  std::vector<int64_t> countsShape = {8};
  void* selfDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* inverseDeviceAddr = nullptr;
  void* countsDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* valueOut = nullptr;
  aclTensor* inverseOut = nullptr;
  aclTensor* countsOut = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> valueHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> inverseHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> countsHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  bool sorted = false;
  bool returnInverse = false;
  bool returnCounts = false;
  
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a valueOut aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &valueOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an inverseOut aclTensor.
  ret = CreateAclTensor(inverseHostData, inverseShape, &inverseDeviceAddr, aclDataType::ACL_INT64, &inverseOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a countsOut aclTensor.
  ret = CreateAclTensor(countsHostData, countsShape, &countsDeviceAddr, aclDataType::ACL_INT64, &countsOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnUnique2.
  ret = aclnnUnique2GetWorkspaceSize(self, sorted, returnInverse, returnCounts, valueOut, inverseOut, countsOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnique2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnUnique2.
  ret = aclnnUnique2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnique2 failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. The code needs to be modified based on the API definition.
  auto size = GetShapeSize(valueShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), valueDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  
  // 6. Release the aclTensor and aclScalar. You need to modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(valueOut);
  aclDestroyTensor(inverseOut);
  aclDestroyTensor(countsOut);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(inverseDeviceAddr);
  aclrtFree(countsDeviceAddr);
  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
