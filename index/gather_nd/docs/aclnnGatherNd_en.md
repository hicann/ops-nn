# aclnnGatherNd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:17:19.610Z -->

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

- Description: For an input tensor `self` with dimension **r≥1** and an input tensor `indices` with dimension **q≥1**, data slices are gathered into an output tensor `out` with dimension **(q-1) + (r - indices_shape[-1])**. `indices` is a **q**-dimensional integer tensor, which can be viewed as a **q-1**-dimensional special tensor composed of **Index Pairs** (each **Index Pair** is a one-dimensional tensor of length **indices_shape[-1]**, and each **Index Pair** points to a slice in **self**).
- Formula:
  - If indices_shape[-1] > r, it is an invalid scenario.
  - If indices_shape[-1] = r, the dimension of the output tensor out is q-1, that is, the shape of out is [indices_shape[0:q-1]], and the elements in out are the elements at the index pair positions of self. (See Example 1.)
  - If indices_shape[-1] < r, the dimension of the output tensor out is (q-1) + (r - indices_shape[-1]). Let c=indices_shape[-1]. That is, the shape of out is [indices_shape[0:q-1],self_shape[c:r]], and `out` consists of slices of `self` at the index pair positions. (See Examples 2, 3, and 4)

  The constraints on **r**, **q**, and **indices_shape[-1]** are as follows:
  - r ≥ 1 and q ≥ 1 must be satisfied.
  - The value of indices_shape[-1] must be in the range [1, r] (inclusive).
  - Each element of `indices` must be within the range [-s, s-1] (where s is the value on each axis of self_shape), i.e., -self_shape[i] ≤ indices[...,i] ≤ self_shape[i]-1.

- Example:

  ```text
  Example 1:
    self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
    indices: [[0, 0], [1, 1]]   # indices_shape=[2, 2], q=2, indices_shape[-1]=2
    out: [0, 3]                 # out_shape=[2]
  Example 2:
    self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
    indices: [[1], [0]]         # indices_shape=[2, 1], q=2, indices_shape[-1]=1
    out: [[2, 3], [0, 1]]       # out_shape=[2, 2]
  Example 3:
    self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
    indices: [[0, 1], [1, 0]]                  # indices_shape=[2, 2], q=2, indices_shape[-1]=2
    out: [[2, 3], [4, 5]]                      # out_shape=[2, 2]
  Example 4:
    self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
    indices: [[[0, 1]], [[1, 0]]]              # indices_shape=[2, 1, 2], q=3, indices_shape[-1]=2
    out: [[[2, 3]], [[4, 5]]]                  # out_shape=[2, 1, 2]
  ```

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call the "aclnnGatherNdGetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, call the "aclnnGatherNd" API to execute the computation.

```Cpp
aclnnStatus aclnnGatherNdGetWorkspaceSize(
 const aclTensor *self,
 const aclTensor *indices,
 bool             negativeIndexSupport,
 aclTensor       *out,
 uint64_t        *workspaceSize,
 aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGatherNd(
 void          *workspace,
 uint64_t       workspaceSize,
 aclOpExecutor *executor,
 aclrtStream    stream)
```

## aclnnGatherNdGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1479px"><colgroup>
    <col style="width: 183px">
    <col style="width: 120px">
    <col style="width: 265px">
    <col style="width: 299px">
    <col style="width: 197px">
    <col style="width: 114px">
    <col style="width: 156px">
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
        <td>self</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>-</td>
        <td>INT64, INT32, INT8, UINT8, BOOL, FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT16, UINT16, UINT32, UINT64</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>Input</td>
        <td>Input aclTensor.</td>
        <td>-</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>negativeIndexSupport</td>
        <td>Input</td>
        <td>Whether the ONNX model contains negative indexes.</td>
        <td>If True, the ONNX model contains negative indexes. If False, indexes are within the valid range and no negative index exists.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output aclTensor.</td>
        <td>The data type must be the same as that of self.</td>
        <td>Same as self</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: DOUBLE, INT16, UINT16, UINT32, and UINT64 data types are not supported.
    <!-- end id7 -->

- **Return Value**

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
      <td>Parameters self and out are null pointers.</td>
      </tr>
      <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of self is not in the supported range.</td>
      </tr>
      <tr>
      <td>The number of dimensions of self or indices is less than 1 or greater than 8.</td>
      </tr>
      <tr>
      <td>The data types of self and out are inconsistent.</td>
      </tr>
    </tbody>
    </table>

## aclnnGatherNd

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
      <td>Memory address of the workspace allocated on the device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnGatherNdGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op that contains the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
      </tr>
      </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGatherNd defaults to a deterministic implementation.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather_nd.h"

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
  // Boilerplate for resource initialization
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
  // Call aclrtMemcpy to copy host-side data to the device memory.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize the construction according to the API.
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> indicesShape = {2, 2};
  std::vector<int64_t> outShape = {2};
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  bool negativeIndexSupport = true;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<int64_t> indicesHostData = {0, 0, 1, 1};
  std::vector<float> outHostData = {0, 3};
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnGatherNd.
  ret = aclnnGatherNdGetWorkspaceSize(self, indices, negativeIndexSupport, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherNdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnGatherNd.
  ret = aclnnGatherNd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherNd failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device side to the host side. Modify it based on the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify it based on the definition of the specific API.
  aclDestroyTensor(self);
  aclDestroyTensor(indices);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify it based on the definition of the specific API.
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
