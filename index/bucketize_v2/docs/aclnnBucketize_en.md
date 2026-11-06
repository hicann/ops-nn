# aclnnBucketize

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:08:50.425Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Determines the interval index of each element in the input tensor based on the given boundaries array. For input value $x$ and boundaries array $boundaries = [b_0, b_1, ..., b_{n-1}]$, the output index $y_i$ satisfies:
  - If $x \le b_0$, then $y_i = 0$
  - If $b_{i-1} < x \le b_i$, then $y_i = i$ (default: left-open right-closed interval)
  - If $x > b_{n-1}$, then $y_i = n$
- Formula:

  When right=False (default, left-open right-closed interval):
  $$
  y_i = \min\{j \mid x_i \le b_j\}
  $$

  When right=True (left-closed right-open interval):
  $$
  y_i = \min\{j \mid x_i < b_j\}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnBucketizeGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator's computation process, and then call the "aclnnBucketize" API to perform the computation.

```cpp
aclnnStatus aclnnBucketizeGetWorkspaceSize(
  const aclTensor  *self,
  const aclTensor  *boundaries,
  const bool        outInt32,
  const bool        right,
  aclTensor        *out,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnBucketize(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnBucketizeGetWorkspaceSize

- **Parameters**

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
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor, the data whose interval indices are to be determined.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data type must satisfy the deduction relationship with the data type of boundaries.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>boundaries (aclTensor*)</td>
      <td>Input</td>
      <td>Boundary array, must be a one-dimensional tensor and monotonically increasing.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Must be a one-dimensional tensor.</li><li>Must be monotonically increasing with no duplicate values.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outInt32 (bool)</td>
      <td>Input</td>
      <td>Control parameter for the output data type.</td>
       <td><ul><li>True indicates that the output data type is INT32.</li><li>False indicates that the output data type is INT64.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>right (bool)</td>
      <td>Input</td>
      <td>Interval boundary control parameter.</td>
       <td><ul><li>True indicates that the interval includes the left boundary (left-closed right-open).</li><li>False indicates that the interval does not include the left boundary (left-open right-closed).</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Output tensor, with each element indicating the corresponding interval index.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape must be the same as self.</li><li>The data type is determined by outInt32.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>Same as self</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> ACLNN_ERR_PARAM_NULLPTR </td>
      <td> 161001 </td>
      <td>self, boundaries, and out are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="5"> ACLNN_ERR_PARAM_INVALID </td>
      <td rowspan="5"> 161002 </td>
      <td>The data type of self or boundaries is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self and boundaries do not meet the deduction relationship.</td>
    </tr>
    <tr>
      <td>The dimension of boundaries is not 1.</td>
    </tr>
    <tr>
      <td>The shape of out does not match the shape of self.</td>
    </tr>
    <tr>
      <td>The data type of out does not match the outInt32 parameter.</td>
    </tr>
  </tbody></table>

## aclnnBucketize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
   <col style="width: 200px">
   <col style="width: 130px">
   <col style="width: 770px">
  </colgroup>
  <thead>
            <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
        </thead>
        <tbody>
            <tr><td>workspace</td><td>Input</td><td>Workspace memory address allocated on the device.</td></tr>
            <tr><td>workspaceSize </td><td>Input</td><td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnBucketizeGetWorkspaceSize.</td></tr>
            <tr><td>executor</td><td>Input</td><td> Op executor, which contains the computation flow of the operator. </td></tr>
            <tr><td>stream </td><td>Input</td><td> Stream for executing the task. </td></tr>
        </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnBucketize defaults to a deterministic implementation.
- boundaries must be a one-dimensional tensor.
- boundaries must be monotonically increasing and contain no duplicate values (cannot be verified at the software level; you must ensure this).
- The data types of self and boundaries must satisfy the deduction relationship.
- The shape of out must be the same as that of self.
- If outInt32 is True, the data type of out must be INT32; if outInt32 is False, the data type of out must be INT64.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:

```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_bucketize.h"

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
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
         strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
      return 0;
    }

   int main() {
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> selfShape = {3, 2};
      std::vector<int64_t> boundariesShape = {3};
      std::vector<int64_t> outShape = {3, 2};
      void* selfDeviceAddr = nullptr;
      void* boundariesDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* self = nullptr;
      aclTensor* boundaries = nullptr;
      aclTensor* out = nullptr;

      std::vector<float> selfHostData = {-5, 10000, 150, 10, 5, 100};
      std::vector<float> boundariesHostData = {0, 10, 100};
      std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0};

      ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(boundariesHostData, boundariesShape, &boundariesDeviceAddr, aclDataType::ACL_FLOAT, &boundaries);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      bool outInt32 = false;
      bool right = false;

      ret = aclnnBucketizeGetWorkspaceSize(self, boundaries, outInt32, right, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBucketizeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
      }

      ret = aclnnBucketize(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBucketize failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      auto size = GetShapeSize(outShape);
      std::vector<int64_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int64_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
      }

      aclDestroyTensor(self);
      aclDestroyTensor(boundaries);
      aclDestroyTensor(out);

      aclrtFree(selfDeviceAddr);
      aclrtFree(boundariesDeviceAddr);
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

<!-- end id7 -->

## Reference

### Pytorch AtenIR

```Cpp
- func: bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool outInt32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU: bucketize_out_cpu
    CUDA: bucketize_out_cuda
```

### AtenIR Parameter Description

| Type | Name | Description | Data Types Supported on GPU | Data Types Supported on CPU |
| ---- | ------ | ---------------------------- | ----------------------------------------------------------------------------------------------- |---|
| Input | input | Input tensor. | FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, DOUBLE | FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, DOUBLE|
| Input | boundaries | Boundary array, must be one-dimensional and monotonically increasing. | FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, DOUBLE | FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, DOUBLE|
| Attribute | outInt32 | Whether the output data type is int32. | bool | bool |
| Attribute | right | Whether the interval includes the left boundary. | bool | bool |
| Input | out | Specified output tensor. | INT32, INT64 |INT32, INT64|
| Output | output | Output tensor. | INT32, INT64 |INT32, INT64|
