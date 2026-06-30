# aclnnMaskedSoftmaxWithRelPosBias

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/masked_softmax_with_rel_pos_bias)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    √    |
| <term>Atlas training series products</term>                             |    ×    |

## Function

- Description: Replaces the part that uses window attention to compute softmax in swinTransformer.

- Formula:

$$
out = \operatorname{softmax}(scaleValue * x + attenMaskOptional + relativePosBias)
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMaskedSoftmaxWithRelPosBias** is called to perform computation.

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

- **Parameters:**
  
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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input parameter for computation, corresponding to x in the formula.</td>
      <td>The shape is four-dimensional (B*W, N, S1, S2) or five-dimensional (B, W, N, S1, S2).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Input</td>
      <td>Input parameter for computation, corresponding to attenMaskOptional in the formula.</td>
      <td>The shape is three-dimensional (W, S1, S2), four-dimensional (W, 1, S1, S2), or five-dimensional (1, W, 1, S1, S2).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>relativePosBias</td>
      <td>Input</td>
      <td>Input parameter for computation, corresponding to relativePosBias in the formula.</td>
      <td>The shape is three-dimensional (N, S1, S2), four-dimensional (1, N, S1, S2), or five-dimensional (1, 1, N, S1, S2).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Input</td>
      <td>Input parameter for computation, corresponding to scaleValue in the formula.</td>
      <td>None</td>
      <td>DOUBLE</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecisionMode</td>
      <td>Input</td>
      <td>Precision mode.</td>
      <td>None</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter for computation, corresponding to out in the formula.</td>
      <td>Its shape is the same as that of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
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
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas inference series products</term>: BFLOAT16 is not supported.
  
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed x, attenMaskOptional, relativePosBias, or out is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The data type, data format, or shape of the input or output parameter is not supported.</td>
    </tr>
  </tbody></table>

## aclnnMaskedSoftmaxWithRelPosBias

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize.</td>
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

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnMaskedSoftmaxWithRelPosBias** defaults to a deterministic implementation.

- <term>Atlas inference series products</term>: The last dimension S2 of the input parameter x must be 32-byte aligned.

- Ensure that the size of the UB required by the shape passed to an operator is less than the total size of the UB provided by the AI processor. In the following example, minComputeSize is the total size of the UB required by the operator, and s2AlignedSize indicates the 32-byte alignment result of S2.
  
  - If attenMaskOptional exists:
    ```
    For the FLOAT type, the formula is as follows:
    dtypeSize = 4;
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288;
    minComputeSize = xSize * 8 + softMaskMinTmpSize;
    For the FLOAT16 type, the formula is as follows:
    dtypeSize = 2;
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288;
    minComputeSize = xSize * 16 + softMaskMinTmpSize;
    ```
  - If attenMaskOptional does not exist:
    ```
    For the FLOAT type, the formula is as follows:
    dtypeSize = 4;
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288;
    minComputeSize = xSize * 6 + softMaskMinTmpSize; 
    For the FLOAT16 type, the formula is as follows:
    dtypeSize = 2;
    xSize = s2AlignedSize * dtypeSize;
    softMaskMinTmpSize = 288;
    minComputeSize = xSize* 12 + softMaskMinTmpSize;
    ```
  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: If the data type is BFLOAT16, the formula is the same as that when FLOAT16 is used.


## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
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

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnMaskedSoftmaxWithRelPosBias.
  ret = aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(x, attenMaskOptional, relativePosBias, 1.0, 0, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnMaskedSoftmaxWithRelPosBias.
  ret = aclnnMaskedSoftmaxWithRelPosBias(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedSoftmaxWithRelPosBias failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
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
