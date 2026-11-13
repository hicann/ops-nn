# aclnnMaskedSoftmaxWithRelPosBias

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:26:20.254Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Replaces the softmax computation using window attention in swinTransformer.

- Formula:

  $$
  out = \operatorname{softmax}(scaleValue * x + attenMaskOptional + relativePosBias)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call the "aclnnMaskedSoftmaxWithRelPosBias" API to perform computation.

```Cpp
aclnnStatus aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *attenMaskOptional,
  const aclTensor *relativePosBias,
  double           scaleValue,
  int64_t          innerPrecisionMode,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnMaskedSoftmaxWithRelPosBias(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 300px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input to be computed, corresponding to x in the formula.</td>
      <td>4D (B*W, N, S1, S2) or 5D (B, W, N, S1, S2)</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Input</td>
      <td>Input to be computed, corresponding to attenMaskOptional in the formula.</td>
      <td>3D (W, S1, S2), 4D (W, 1, S1, S2), or 5D (1, W, 1, S1, S2); passing nullptr is allowed, which is equivalent to computing without a mask</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>√</td>
    </tr>
    <tr>
      <td>relativePosBias</td>
      <td>Input</td>
      <td>Input to be computed, corresponding to relativePosBias in the formula.</td>
      <td>shape is 3D (N, S1, S2), 4D (1, N, S1, S2), or 5D (1, 1, N, S1, S2).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>Input to be computed, corresponding to scaleValue in the formula.</td>
      <td>None</td>
      <td>DOUBLE</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecisionMode</td>
      <td>Input</td>
      <td>Precision mode parameter. Reserved field. Currently, only the value 0 is supported.</td>
      <td>None</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output to be computed, corresponding to out in the formula.</td>
      <td>Same shape as x. Only contiguous tensors are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
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
    </tr>
  </tbody>
  </table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: BFLOAT16 is not supported.
  <!-- end id7 -->

- **Return Value**
  <p>aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).</p>
  <p>The first-phase API performs input parameter validation. Errors are reported in the following use cases:</p>
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x, relativePosBias, or out is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of x, attenMaskOptional (when not null), relativePosBias, or out is not within the supported range, or is inconsistent with the data type of x.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shape of an input or output parameter does not match or is not within the supported dimension range.</td>
    </tr>
  </tbody></table>

## aclnnMaskedSoftmaxWithRelPosBias

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Size of the workspace allocated on the Device side, obtained by the first-phase API aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor containing the operator computation flow.</td>
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
  - aclnnMaskedSoftmaxWithRelPosBias defaults to a deterministic implementation.

- Data type constraint:
  - x supports only FLOAT, FLOAT16, and BFLOAT16. The data types of attenMaskOptional (when not empty), relativePosBias, and out must be the same as that of x; otherwise, the operator reports an error.

- Data format constraint:
  - x, attenMaskOptional, relativePosBias, and out support only the ND format.

- Attribute value range constraint:
  - scaleValue must be a finite value (any DOUBLE value except NaN, +Inf, and -Inf is supported). The operator reports an error if NaN or ±Inf is passed in.
  - innerPrecisionMode is a reserved field. Currently, only the value 0 is supported. The operator reports an error if any other value is passed in.

- Non-contiguous tensor support:
  - x, attenMaskOptional, and relativePosBias support non-contiguous tensors (that is, tensors whose strides do not meet the contiguous storage condition). After being passed in, they are automatically converted to contiguous tensors by the framework before computation.
  - out does not support non-contiguous tensors and must be a contiguous tensor.

<!-- npu="310p" id8 -->
- <term>Atlas inference products</term>: The scenario where the last dimension S2 of the input parameter x is not aligned to 32 bytes is not supported.
<!-- end id8 -->

- Ensure that the UB space required by the shape passed to the operator is smaller than the total UB size of the AI processor version. The total size of the UB space required by this operator, minComputeSize, is as follows, where s2AlignedSize indicates the result of aligning S2 to 32 bytes.

  - When attenMaskOptional exists:

    ```tex
    For FLOAT type, the formula is as follows:
    dtypeSize = 4;
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288;
    minComputeSize = xSize * 8 + softMaskMinTmpSize;
    For FLOAT16 type, the formula is as follows:
    dtypeSize = 2;
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288;
    minComputeSize = xSize * 16 + softMaskMinTmpSize;
    ```

  - When attenMaskOptional does not exist:

    ```tex
    For FLOAT type, the formula is as follows:
    dtypeSize = 4;
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288;
    minComputeSize = xSize * 6 + softMaskMinTmpSize;
    For FLOAT16 type, the formula is as follows:
    dtypeSize = 2;
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288;
    minComputeSize = xSize* 12 + softMaskMinTmpSize;
    ```

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: For the BFLOAT16 type, the formula is consistent with that of the FLOAT16 type.
  <!-- end id9 -->

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_masked_softmax_with_rel_pos_bias.h"

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
  // Boilerplate. Resource initialization.
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
  // Call aclrtMalloc to allocate memory on the Device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy Host-side data to Device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API.
  // Set deviceId based on the actual Device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API definition.
  std::vector<int64_t> xShape = {1, 1, 1, 2, 16};
  std::vector<int64_t> attenMaskOptionalShape = {1, 2, 16};
  std::vector<int64_t> relativePosBiasShape = {1, 2, 16};
  std::vector<int64_t> outShape = {1, 1, 1, 2, 16};

  void* xDeviceAddr = nullptr;
  void* attenMaskOptionalDeviceAddr = nullptr;
  void* relativePosBiasDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* attenMaskOptional = nullptr;
  aclTensor* relativePosBias = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> xHostData = {1.08, -1.56, -1.3, -2.01, 2.18, -2.23, -3.58, 3.22, 1.25, -0.56, -0.3, -1.01, 1.08, -1.13, -3.08, -2.22, -0.08, -2.56, 1.35, 1.01, 0.35, -1.03, -1.28, 1.22, 0.08, -2.56, -1.01, -1.01, -0.18, -6.23, 4.55, -1.82};
  std::vector<float> attenMaskOptionalHostData = {2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,};
  std::vector<float> relativePosBiasHostData = {1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenMaskOptionalHostData, attenMaskOptionalShape, &attenMaskOptionalDeviceAddr, aclDataType::ACL_FLOAT, &attenMaskOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(relativePosBiasHostData, relativePosBiasShape, &relativePosBiasDeviceAddr, aclDataType::ACL_FLOAT, &relativePosBias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnMaskedSoftmaxWithRelPosBias.
  ret = aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(x, attenMaskOptional, relativePosBias, 1.0, 0, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnMaskedSoftmaxWithRelPosBias.
  ret = aclnnMaskedSoftmaxWithRelPosBias(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBias failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, and copy the result on the device side to the host side. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify them based on the specific API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(attenMaskOptional);
  aclDestroyTensor(relativePosBias);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(xDeviceAddr);
  aclrtFree(attenMaskOptionalDeviceAddr);
  aclrtFree(relativePosBiasDeviceAddr);
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
