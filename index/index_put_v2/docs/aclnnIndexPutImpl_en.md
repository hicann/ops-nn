# aclnnIndexPutImpl

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/index_put_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                         |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    √   |

## Function

- Function: Replaces or accumulates the data of the coordinates corresponding to the input self with the input values based on the indices.
- Formula:

  - If **accumulate** is set to **False**:

    $$
    self[indices] = values
    $$

  - If **accumulate** is set to **True**:
    
    $$
    self[indices]  = self[indices]  + values
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnIndexPutImplGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnIndexPutImpl** is called to perform computation.

```Cpp
aclnnStatus aclnnIndexPutImplGetWorkspaceSize(
 aclTensor*             selfRef,
 const aclTensorList*   indices,
 const aclTensor*       values,
 const bool             accumulate,
 const bool             unsafe,
 uint64_t*              workspaceSize,
 aclOpExecutor**        executor)
```

```Cpp
aclnnStatus aclnnIndexPutImpl(
 void*             workspace,
 uint64_t          workspaceSize,
 aclOpExecutor*    executor,
 aclrtStream       stream)
```

## aclnnIndexPutImplGetWorkspaceSize

* **Parameters**

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
        <td>selfRef</td>
        <td>Input</td>
        <td>Self in the formula.</td>
        <td>The data type is the same as that of values.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>Input</td>
        <td>Indices in the formula.</td>
        <td>-</td>
        <td>INT32, INT64, BOOL</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>values</td>
        <td>Input</td>
        <td>Values in the formula.</td>
        <td>-</td>
        <td>Same as selfRef</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>accumulate</td>
        <td>Input</td>
        <td>Operation type flag for accumulation or update, which is a Boolean value on the host.</td>
        <td><ul><li>If accumulate is True, accumulation is performed.</li><li>If accumulate is False, update is performed.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>unsafe</td>
        <td>Input</td>
        <td>Whether the index is within the valid range.</td>
        <td><ul><li>If unsafe is set to True, an error is reported and the execution exits when an index out-of-bounds occurs.</li><li>If unsafe is set to False and an index out-of-bounds occurs, a runtime exception may occur.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
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
    - For the Ascend 950PR/Ascend 950DT, if accumulate is set to False and indices have duplicate index values, the result is not order-preserving.

* **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.
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
      <td>The input selfRef, indices, or values is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types of selfRef and values are not supported.</td>
      </tr>
      <tr>
      <td>The data types of selfRef and values are different.</td>
      </tr>
      <tr>
      <td>The data formats of selfRef and values are different.</td>
      </tr>
    </tbody>
    </table>

## aclnnIndexPutImpl

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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnIndexPutImplGetWorkspaceSize API.</td>
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
  - **aclnnIndexPutImpl** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computation.

- The input parameters **selfRef**, **indices**, and **values** have the following constraints:
  - The number of tensors in indices cannot exceed the number of dimensions of selfRef.
  - The dimensions of values must meet the following formula or the following formula after broadcasting:
      - values.Dims() = indices[i].Dims() + (selfRef.Dims() - indices.size())
      - The first half of the dimensions of **values** must be the same as the tensor dimensions in **indices** (the tensors in **indices** are broadcast to the same shape), and the second half of the dimensions must be the same as the dimensions of **selfRef** minus the number of tensors in **indices**.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_put_impl.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on the API definition.
  std::vector<int64_t> selfShape = {3, 4};
  std::vector<int64_t> indexShape = {1};
  std::vector<int64_t> valueShape = {1};
  void* selfDeviceAddr = nullptr;
  void* indexOneDeviceAddr = nullptr;
  void* indexTwoDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indexOne = nullptr;
  aclTensor* indexTwo = nullptr;
  aclTensor* value= nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> valueHostData = {1};
  std::vector<int64_t> indexOneHostData = {0};
  std::vector<int64_t> indexTwoHostData = {2};
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an index aclTensor.
  ret = CreateAclTensor(indexOneHostData, indexShape, &indexOneDeviceAddr, aclDataType::ACL_INT64, &indexOne);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexTwoHostData, indexShape, &indexTwoDeviceAddr, aclDataType::ACL_INT64, &indexTwo);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* indexs[] = {indexOne, indexTwo};
  auto indexTensorList = aclCreateTensorList(indexs, 2);
  // value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnIndexPutImpl API call example
  // 3. Call the CANN operator library API, which needs to be replaced with a specific operator API.
  // Call the first-phase API of aclnnIndexPutImpl.
  ret = aclnnIndexPutImplGetWorkspaceSize(self, indexTensorList, value, true, false, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexPutImplGetWorkspaceSizefailed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnIndexPutImpl.
  ret = aclnnIndexPutImpl(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexPutImplfailed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the code based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensorList(indexTensorList);
  aclDestroyTensor(value);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexOneDeviceAddr);
  aclrtFree(indexTwoDeviceAddr);
  aclrtFree(valueDeviceAddr);
  if (workspaceSize > 0) {
   aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
