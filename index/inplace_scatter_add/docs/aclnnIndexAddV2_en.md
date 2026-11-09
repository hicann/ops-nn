# aclnnIndexAddV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:26:28.480Z -->

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

Adds values from the source tensor to the values at the corresponding positions in the input tensor along a specified dimension based on the given indices.

This operator uses in-place semantics: the computation is accumulated directly onto the storage of self, and self is overwritten in place after the call (equivalent to executing self[index] += alpha × source in place). The underlying operator's output is bound to the storage of self (in-place binding), and the aclnn API copies the result to out for delivery.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnIndexAddV2GetWorkspaceSize" to obtain the workspace size required for computation and the executor containing the operator computation process, and then call "aclnnIndexAddV2" to perform computation.

```Cpp
aclnnStatus aclnnIndexAddV2GetWorkspaceSize(
 const aclTensor*  self,
 const int64_t     dim,
 const aclTensor*  index,
 const aclTensor*  source,
 const aclScalar*  alpha,
 int64_t           mode,
 aclTensor*        out,
 uint64_t*         workspaceSize,
 aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnIndexAddV2(
 void*          workspace,
 uint64_t       workspaceSize,
 aclOpExecutor* executor,
 aclrtStream    stream)
```

## aclnnIndexAddV2GetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1451px"><colgroup>
    <col style="width: 143px">
    <col style="width: 120px">
    <col style="width: 227px">
    <col style="width: 265px">
    <col style="width: 274px">
    <col style="width: 117px">
    <col style="width: 160px">
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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>Input tensor.</td>
        <td>1 to 8 dimensions are supported. A 0-D self has no valid dim value (the value range [-self.dim(), self.dim()-1] is empty when self is 0-D), and it is rejected by the first-phase validation.</td>
        <td>FLOAT, FLOAT16, INT32, INT16, INT8, UINT8, DOUBLE, INT64, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>The specified dimension.</td>
        <td>Value range: [-self.dim(), self.dim()-1].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index (aclTensor*)</td>
        <td>Input</td>
        <td>Index.</td>
        <td>The shape size of index must be equal to the shape value of source on the dim dimension.</td>
        <td>INT64, INT32</td>
        <td>ND</td>
        <td>0-1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>source (aclTensor*)</td>
        <td>Input</td>
        <td>Source tensor.</td>
        <td>Except for the dim dimension, other dimension values of the source shape must be equal to those of the self shape.</td>
        <td>Consistent with self</td>
        <td>ND</td>
        <td>Consistent with self</td>
        <td>√</td>
      </tr>
      <tr>
        <td>alpha (aclScalar*)</td>
        <td>Input</td>
        <td>Scaling factor.</td>
        <td>The data type can be converted to the data type after <a href="../../../docs/en/context/deduction_relationship.md">deduction</a> between self and source.<br>A floating-point alpha with an integer self is accepted: the alpha value is truncated according to the conversion semantics toward the self data type (for example, 0.5 is converted to 0 for an integer self). When the alpha value is outside the representable range of the self data type (for example, alpha is inf, nan, or a floating-point value beyond the representable range while self is an integer type), the behavior is undefined.<br>When self is BOOL, alpha supports only BOOL (BOOL supports one-way conversion to other data types; numeric/floating-point types cannot be converted to BOOL).<br>In a non-deterministic computation use case where mode is 0, alpha does not participate in the computation.</td>
        <td>Scalar type (must be convertible to the data type of self; complex types are not supported)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>mode (int64_t)</td>
        <td>Input</td>
        <td>Computation mode.</td>
        <td>When the value is 0, the high-performance mode is used. Other values indicate the non-high-performance mode. For limitations of the high-performance mode, see Constraints.<br>In deterministic computation use cases, the value of mode is ignored and the deterministic computation mode is used.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Output aclTensor.</td>
        <td>The underlying operator binds the output to the storage of self in place (self is overwritten in place), and the API copies the result to out.</td>
        <td>Consistent with self</td>
        <td>ND</td>
        <td>Consistent with self</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator computation flow.</td>
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
      <td>self, index, source, alpha, or out is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data types of self, index, source, and out are not within the supported range.</td>
      </tr>
      <tr>
      <td>The data types of self, source, and out are not consistent.</td>
      </tr>
      <tr>
      <td>The value of dim is not within the range [-self.dim(), self.dim()-1].</td>
      </tr>
      <tr>
      <td>The data type of alpha cannot be converted to the data type of self (for example, alpha is not BOOL while self is BOOL, or alpha is a complex type).</td>
      </tr>
      <tr>
      <td>Except for the dim dimension, the shape values of self and source are not equal on other dimensions.</td>
      </tr>
      <tr>
      <td>index is not 0- or 1-dimensional.</td>
      </tr>
      <tr>
      <td>The shape size of index is not equal to the shape size of source on dimension dim.</td>
      </tr>
      <tr>
      <td>The shape of out is not equal to the shape of self.</td>
      </tr>
      <tr>
      <td>When mode=0, the high-performance mode restrictions in Constraints are not met.</td>
      </tr>
    </tbody>
    </table>

## aclnnIndexAddV2

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
      <td>Workspace memory address requested on the device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device, obtained by the first-phase API aclnnIndexAddV2GetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op. It contains the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnIndexAddV2 defaults to a non-deterministic implementation. You can enable deterministic computation through aclrtSetSysParamOpt.
- In-place semantics
  - self is overwritten in place after the call (see Function); GE graph mode (GEIR) is not within the supported scope of this operator.
- Index value range
  - The index value is expected to fall within the range [0, self.shape[dim]). The first-phase API does not validate the index value range: negative indices are silently skipped by the kernel-side guard (not accumulated), and out-of-bounds indices result in undefined behavior (an execution-time exception may be triggered when the InplaceIndexAdd path is used).
- Empty tensor
  - When source is an empty tensor, the API copies self to out (no accumulation is performed); when self is an empty tensor, out is an empty tensor.
- Capacity limit
  - The 32-bit fast path of the operator requires the element scale involved in the computation to be within the uint32 domain (2^32-1). Beyond that, the computation is automatically routed to the 64-bit implementation with identical results. The actual supported tensor scale limit depends on the data type and Device memory.
- High-performance mode (mode=0) has the following additional constraints:
  - dim must be 0 or -2.
  - The data type of self must be FLOAT, FLOAT16, INT32, INT16, or BFLOAT16.
  - self must be a two-dimensional tensor.
  - If mode=0 and the constraints in the parameter description are met but the above three constraints are not, the API throws an error.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md). For the complete sample code, see [test_aclnn_index_add_v2.cpp](../examples/test_aclnn_index_add_v2.cpp).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_add_v2.h"

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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); aclrtFree(*deviceAddr);
            *deviceAddr = nullptr; return ret);

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

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> indexShape = {4};
  std::vector<int64_t> sourceShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* sourceDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* source = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> indexHostData = {0, 1, 2, 3};
  std::vector<float> sourceHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  int64_t dim = 0;
  int64_t mode = 0;
  float alphaValue = 1.0f;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int32_t exitCode = 0;
  int64_t resultSize = 0;
  std::vector<float> resultData;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor(self) failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);
  // Create the index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT32, &index);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor(index) failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);
  // Create the source aclTensor.
  ret = CreateAclTensor(sourceHostData, sourceShape, &sourceDeviceAddr, aclDataType::ACL_FLOAT, &source);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor(source) failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor(out) failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);
  // Create the alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("create alpha failed.\n"); exitCode = -1; goto cleanup);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  // Call the first-phase API of aclnnIndexAddV2.
  ret = aclnnIndexAddV2GetWorkspaceSize(self, dim, index, source, alpha, mode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexAddV2GetWorkspaceSize failed. ERROR: %d\n", ret);
            exitCode = -1; goto cleanup);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); exitCode = -1;
              goto cleanup);
  }
  // Call the second-phase API of aclnnIndexAddV2.
  ret = aclnnIndexAddV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexAddV2 failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);

  // 5. Obtain the output value and copy the result from the device side to the host side. Change it according to the specific API definition.
  resultSize = GetShapeSize(outShape);
  resultData.assign(resultSize, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    resultSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
            exitCode = -1; goto cleanup);
  for (int64_t i = 0; i < resultSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

cleanup:
  // 6. Release the aclTensor and aclScalar. Change it according to the specific API definition.
  // (Objects that have been created are also released on failure paths.)
  if (self != nullptr) {
    aclDestroyTensor(self);
  }
  if (index != nullptr) {
    aclDestroyTensor(index);
  }
  if (source != nullptr) {
    aclDestroyTensor(source);
  }
  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  if (out != nullptr) {
    aclDestroyTensor(out);
  }

  // 7. Release device resources. Modify the code based on the specific API definition.
  // (Resources that have been allocated are also released on failure paths.)
  if (selfDeviceAddr != nullptr) {
    aclrtFree(selfDeviceAddr);
  }
  if (indexDeviceAddr != nullptr) {
    aclrtFree(indexDeviceAddr);
  }
  if (sourceDeviceAddr != nullptr) {
    aclrtFree(sourceDeviceAddr);
  }
  if (outDeviceAddr != nullptr) {
    aclrtFree(outDeviceAddr);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return exitCode;
}
```
