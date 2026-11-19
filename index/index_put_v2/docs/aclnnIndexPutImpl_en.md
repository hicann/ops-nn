# aclnnIndexPutImpl

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:23:24.347Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Replaces or accumulates data in the input self at positions specified by indices with the input values.
- Formula:

  - accumulate = False:

    $$
    self[indices] = values
    $$

  - accumulate = True:

    $$
    self[indices]  = self[indices]  + values
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnIndexPutImplGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the "aclnnIndexPutImpl" API to perform the computation.

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
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>selfRef</td>
        <td>Input</td>
        <td>self in the formula.</td>
        <td>The data type is consistent with values.</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>Input</td>
        <td>indices in the formula.</td>
        <td>The number of tensors in indices cannot exceed the number of dimensions of selfRef (maximum 8 dimensions). When the data type is BOOL, each tensor serves as a mask input and corresponds to the dimensions of selfRef in sequence. The size of each dimension of each mask tensor must be exactly consistent with the size of the corresponding dimension in selfRef</td>
        <td>INT32, INT64, BOOL</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>values</td>
        <td>Input</td>
        <td>values in the formula.</td>
        <td> - </td>
        <td>Consistent with selfRef</td>
        <td>ND</td>
        <td>number of dimensions = number of dimensions of indices after broadcasting + (number of dimensions of selfRef - number of indices)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>accumulate</td>
        <td>Input</td>
        <td>Operation type flag for accumulation or update, Boolean value on the Host side.</td>
        <td><ul><li>If accumulate is True, accumulation is performed;</li><li>If accumulate is False, update is performed.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>unsafe</td>
        <td>Input</td>
        <td>Flag that indicates whether to skip the index out-of-bounds check.</td>
        <td><ul><li>If unsafe is True, the index out-of-bounds check is not performed;</li><li>If unsafe is False, the index out-of-bounds check is performed.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be applied on the side of Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

    <!-- npu="910" id7 -->
    - <term>Atlas training products</term>: The BFLOAT16 data type is not supported.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: When accumulate is False and indices contain duplicate index values, the result is not order-preserving.
    <!-- end id8 -->

* **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
    </colgroup>
    <thead>
      <tr>
      <th>Return Value</th>
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
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of selfRef, values, or indices is not supported.</td>
      </tr>
      <tr>
      <td>The data types of selfRef and values are different.</td>
      </tr>
      <tr>
      <td>The number of dimensions of selfRef exceeds 8.</td>
      </tr>
      <tr>
      <td>The number of tensors in indices exceeds the number of dimensions of selfRef.</td>
      </tr>
      <tr>
      <td>The shape of the BOOL-type index (mask) in indices is not consistent with the corresponding dimension of selfRef.</td>
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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace applied on the Device side.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnIndexPutImplGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the computation flow of the operator.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnIndexPutImpl defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.

- The input parameters selfRef, indices, and values generally have the following constraints:
  - The number of tensors in indices cannot exceed the dimension of selfRef (a maximum of 8 dimensions).
  - When the data type of indices is BOOL, each tensor serves as a mask input and corresponds to the dimensions of selfRef in order. The size of each dimension of each mask tensor must be completely consistent with the size of the corresponding dimension in selfRef. For example, if the shape of selfRef is (3,4,5) and indices contains two tensors of the BOOL type, the shape of the first mask must be (3,) and the shape of the second mask must be (4,).
  - The dimensions of values must satisfy the following formula or satisfy it after broadcasting:
      - values.Dims() = indices[i].Dims() + (selfRef.Dims() - indices.size())
      - This means that the first half of the values dimensions must match the dimensions of the tensors in indices (the tensors in indices are broadcast to the same shape), and the second half must match the dimensions of selfRef minus the number of tensors in indices.

- Non-contiguous tensor support:
  - selfRef, indices, and values support non-contiguous tensors (that is, tensors whose strides do not meet the contiguous storage condition).
  - The non-contiguous use case has the following restrictions:
      - The number of dimensions of selfRef, indices, and values does not exceed 4.
      - When accumulate is True and the data type is FLOAT16/BFLOAT16/INT8/UINT8, non-contiguous optimization does not take effect (the high-precision path is used).

- Index element count limit:
  - Generally, the number of elements in a single index tensor does not exceed 60,000,000.
  - The high-dimensional use case has stricter limits: no more than 54,247,424 for 5 dimensions; no more than 48,611,328 for 6 dimensions; no more than 44,384,256 for 7 dimensions; and no more than 40,861,696 for 8 dimensions.
  - When the IndexPutWithSortV2 optimization path is used, the index value range does not exceed 16,777,216, and the number of elements in the trailing dimension of self does not exceed 2,147,483,647.

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
  // Boilerplate for resource initialization.
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
  // Call aclrtMalloc to allocate Device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy Host-side data to Device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  // Calculate the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                           shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Process the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
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
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the index aclTensor.
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

  // Example of calling the aclnnIndexPutImpl API.
  // 3. Call the CANN operator library API. Change it to the specific operator API.
  // Call the first-phase API of aclnnIndexPutImpl.
  ret = aclnnIndexPutImplGetWorkspaceSize(self, indexTensorList, value, true, false, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexPutImplGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnIndexPutImpl.
  ret = aclnnIndexPutImpl(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexPutImpl failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value. Copy the result from the Device side to the Host side. Modify the code according to the specific API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor. Modify the code according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensorList(indexTensorList);
  aclDestroyTensor(value);

  // 7. Release device resources. Modify the code according to the specific API definition.
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
