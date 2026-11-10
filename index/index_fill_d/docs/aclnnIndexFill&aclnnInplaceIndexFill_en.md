# aclnnIndexFill&aclnnInplaceIndexFill

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-08T11:20:53.185Z pushedAt=2026-10-09T02:02:11.398Z -->

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Replaces the values at the positions specified by index along the given axis dim of the input self with value. The computation is deterministic.
- Example: The input self is:

      [[1, 2, 3],

      [4, 5, 6],

      [7, 8, 9]]

  If dim = 0, index = [0, 2], and value = 0, the operator's computation result is:

      [[0, 0, 0],

      [4, 5, 6],

      [0, 0, 0]]

  If dim = 1, index = [0, 2], and value = 0, the operator's computation result is:

      [[0, 2, 0],

      [0, 5, 0],

      [0, 8, 0]]

## Function Prototype

- aclnnIndexFill and aclnnInplaceIndexFill implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.
    - aclnnIndexFill: Create a new output tensor object to store the computation result.
    - aclnnInplaceIndexFill: No need to create a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator is divided into [two-phase APIs](../../../docs/en/context/two_phase_api.md). First call "aclnnIndexFillGetWorkspaceSize" or "aclnnInplaceIndexFillGetWorkspaceSize" to obtain the workspace size required for computation and the executor containing the operator computation process, and then call "aclnnIndexFill" or "aclnnInplaceIndexFill" to perform the computation.

```Cpp
  aclnnStatus aclnnIndexFillGetWorkspaceSize(
   const aclTensor*        self,
   int64_t           dim,
   const aclTensor*  index,
   const aclScalar*  value,
   aclTensor*        out,
   uint64_t*         workspaceSize,
   aclOpExecutor**   executor)
  ```

  ```Cpp
  aclnnStatus aclnnIndexFill(
   void          *workspace,
   uint64_t       workspaceSize,
   aclOpExecutor *executor,
   aclrtStream    stream)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceIndexFillGetWorkspaceSize(
   aclTensor*       self,
   int64_t          dim,
   const aclTensor* index,
   const aclScalar* value,
   uint64_t*        workspaceSize,
   aclOpExecutor**  executor)
  ```

  ```Cpp
  aclnnStatus aclnnInplaceIndexFill(
   void             *workspace,
   uint64_t          workspaceSize,
   aclOpExecutor    *executor,
   aclrtStream       stream)
  ```

## aclnnIndexFillGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1457px"><colgroup>
    <col style="width: 147px">
    <col style="width: 120px">
    <col style="width: 233px">
    <col style="width: 257px">
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
        <th>Dim (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>input</td>
        <td>self in the function example, that is, the tensor whose values at specified positions are to be replaced with value.</td>
        <td>-</td>
        <td>FLOAT16, FLOAT, INT32, INT64, BOOL, BFLOAT16, INT8, UINT8, INT16, DOUBLE</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>input</td>
        <td>Specifies the dimension along which self is filled.</td>
        <td>If self is 1-8 dimensional, dim is in the range of [-self.dim(), self.dim()). If self is 0-dimensional, dim is in the range of [-1, 1).</td>
        <td>int64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>Input</td>
        <td>Specifies the indices to fill self along dim.</td>
        <td>Each element value is smaller than the dimension size of self along dim. Negative values are supported. Out-of-bound values are allowed and directly skipped. When index is an empty array (array length is 0), no position is filled and the output is the same as the input.</td>
        <td>INT32, INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Specifies the fill value.</td>
        <td>Must be convertible to the data type of self.</td>
        <td>Same range as the data type of self</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>output</td>
        <td>The specified output tensor.</td>
        <td>-</td>
        <td>Same as self</td>
        <td>ND</td>
        <td>Same as self</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>output</td>
        <td>Returns the workspace size to be allocated on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>output</td>
        <td>Returns the op executor containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>
    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Data types supported: FLOAT16, FLOAT, INT32, INT64, BOOL, BFLOAT16.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: Data types supported: FLOAT16, FLOAT, INT32, INT64, BOOL, BFLOAT16, INT8, UINT8, INT16, DOUBLE.
    <!-- end id8 -->

- **Return Value**

  aclnnStatus: Return status code. For details, see [aclnn return code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
    </colgroup>
    <thead>
      <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Null pointer passed for self, index, value, or out.</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of self, index, or value are not within the supported range.</td>
      </tr>
      <tr>
      <td>The absolute value of dim exceeds the maximum dim value of self or exceeds 8 dimensions.</td>
      </tr>
      <tr>
      <td>The shapes of self and out are not equal.</td>
      </tr>
      <tr>
      <td>The data types of self and out are not equal.</td>
      </tr>
    </tbody>
    </table>

## aclnnIndexFill

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
      <td>Workspace memory address allocated on the Device side.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnIndexFillGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>input</td>
      <td>op executor. It contains the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>input</td>
      <td>Stream used to execute the task.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: Return status code. For details, see [aclnn return codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceIndexFillGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1457px"><colgroup>
    <col style="width: 147px">
    <col style="width: 120px">
    <col style="width: 233px">
    <col style="width: 257px">
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
        <th>Dimensions (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>input</td>
        <td>Tensor whose values at specified positions are replaced with value.</td>
        <td>-</td>
        <td>FLOAT16, FLOAT, INT32, INT64, BOOL, BFLOAT16, INT8, UINT8, INT16, DOUBLE</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>input</td>
        <td>Specifies the dimension along which self is filled.</td>
        <td>When self is 1-8 dimensional, the value range of dim is [-self.dim(), self.dim()). When self is 0-dimensional, the value range of dim is [-1, 1).</td>
        <td>int64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>input</td>
        <td>Specifies the indices used to fill self along the dim dimension.</td>
        <td>Each element value is smaller than the size of the dim dimension in self. Negative values are supported. Values that exceed the dimension size are supported and skipped directly. When index is an empty array (array length is 0), no position is filled and the output is the same as the input.</td>
        <td>INT32, INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>value</td>
        <td>input</td>
        <td>Specifies the data value to fill.</td>
        <td>-</td>
        <td>Same data type range as self</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>output</td>
        <td>Returns the workspace size to be allocated on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>output</td>
        <td>Returns the op executor containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>
    <!-- npu="A3,910b" id9 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The supported data types are FLOAT16, FLOAT, INT32, INT64, BOOL, and BFLOAT16.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: The supported data types are FLOAT16, FLOAT, INT32, INT64, BOOL, BFLOAT16, INT8, UINT8, INT16, and DOUBLE.
    <!-- end id10 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn return codes](../../../docs/en/context/aclnn_return_code.md).

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
      <td>Null pointer passed for self, index, or value.</td>
      </tr>
      <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data type of self, index, or value is not within the supported range.</td>
      </tr>
      <tr>
      <td>The absolute value of dim exceeds the maximum dim value of self or exceeds 8 dimensions.</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceIndexFill

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
      <td>Workspace memory address allocated on the Device side.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnInplaceIndexFillGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md)..

## Constraints

- deterministic computation:
  - aclnnIndexFill&aclnnInplaceIndexFill default to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_fill.h"

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
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host side to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
  // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {3, 3};
  std::vector<int64_t> indexShape = {1};
  std::vector<int64_t> outShape = selfShape;
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclScalar* value = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> indexHostData = {0};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int64_t dim = 1;
  float fillVal = 10;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the value aclScalar.
  value = aclCreateScalar(&fillVal, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);

  // 3. Call the CANN operator library API. Replace the name with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnIndexFill.
  ret = aclnnIndexFillGetWorkspaceSize(self, dim, index, value, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexFillGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnIndexFill.
  ret = aclnnIndexFill(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexFill failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the allocated variables. Modify the code according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyTensor(out);
  aclDestroyScalar(value);

  // 7. Release device resources. Modify the code according to the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
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

**aclnnInplaceIndexFill API call example:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_fill.h"

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
// Boilerplate: resource initialization.
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

// Calculate the strides of the contiguous tensor.
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
  // Specify deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output. Customize the construction based on the API definition.
  std::vector<int64_t> selfShape = {3, 3};
  std::vector<int64_t> indexShape = {1};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclScalar* value = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int64_t> indexHostData = {0};
  int64_t dim = 1;
  float fillVal = 10;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the value aclScalar.
  value = aclCreateScalar(&fillVal, aclDataType::ACL_FLOAT);
  CHECK_RET(value != nullptr, return ret);

// 3. Call the CANN operator library API. Replace the API name with the specific one.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnInplaceIndexFill.
  ret = aclnnInplaceIndexFillGetWorkspaceSize(self, dim, index, value, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceIndexFillGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
  ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
  // Call the second-phase API of aclnnInplaceIndexFill.
  ret = aclnnInplaceIndexFill(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceIndexFill failed. ERROR: %d\n", ret); return ret);
  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value, copy the result from the device side to the host side, and modify the code based on the specific API definition.
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
  size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
  LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

// 6. Release the allocated variables. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyScalar(value);

// 7. Release device resources. Modify the code based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
  if (workspaceSize > 0) {
  aclrtFree(workspaceAddr);
    }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
  }
```
