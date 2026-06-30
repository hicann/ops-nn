# aclnnWeightQuantBatchMatmul

**This API will be deprecated in later versions. Use [aclnnWeightQuantBatchMatmulV2](../../weight_quant_batch_matmul_v2/docs/aclnnWeightQuantBatchMatmulV2_en.md) and [aclnnWeightQuantBatchMatmulV3](../../weight_quant_batch_matmul_v2/docs/aclnnWeightQuantBatchMatmulV3_en.md) instead.**
[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/weight_quant_batch_matmul)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √       |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |     ×     |
| <term>Atlas inference series products</term>                            |   ×     |
| <term>Atlas training series products</term>                             |   ×     |

## Function

- Description: Quantizes **mat2** in **self * mat2** (matmul/batchmatmul), which is a process of fake-quantization.
- Formula:

  $$
  result = self@mat2+bias
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnWeightQuantBatchMatmulGetWorkspaceSize** is called to obtain the input parameters and to compute the workspace size required for computation. Then, **aclnnWeightQuantBatchMatmul** is called to perform computation.
```cpp
aclnnStatus aclnnWeightQuantBatchMatmulGetWorkspaceSize(
  const aclTensor *x1, 
  const aclTensor *x2, 
  const aclTensor *diagonalMatrix, 
  const aclTensor *deqOffset, 
  const aclTensor *deqScale, 
  const aclTensor *addOffset, 
  const aclTensor *mulScale, 
  const aclTensor *bias, 
  bool             transposeX1, 
  bool             transposeX2, 
  float            antiquantScale, 
  float            antiquantOffset, 
  aclTensor       *out, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```
```cpp
aclnnStatus aclnnWeightQuantBatchMatmul(
  void              *workspace, 
  uint64_t           workspaceSize, 
  aclOpExecutor     *executor, 
  const aclrtStream  stream)
```

## aclnnWeightQuantBatchMatmulGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1526px"><colgroup>
  <col style="width: 155px">
  <col style="width: 123px">
  <col style="width: 318px">
  <col style="width: 328px">
  <col style="width: 124px">
  <col style="width: 122px">
  <col style="width: 209px">
  <col style="width: 147px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Input `self` in the formula.</td>
      <td>The shape can only be two-dimensional and does not support the batch axis. The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x2.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input `mat2` in the formula after processing.</td>
      <td>The shape can only be two-dimensional and does not support the batch axis. The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x1.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diagonalMatrix</td>
      <td>Input</td>
      <td>Input `mat2` in the formula after dequantization on x2.</td>
      <td>The shape is (32, 32), which is an identity matrix. When m is greater than 64, this parameter is not involved in the computation and can be empty.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>Two-dimensional, with the shape being (32, 32).</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqOffset</td>
      <td>Input</td>
      <td>Input `mat2` in the formula after dequantization on x2. It is computed based on addOffset, antiquantOffset, and antiquantScale. For details about the compute method, see the sample code.</td>
      <td>The shape can be 1, n, (1, 1), (1, n), or (n, 1), and must meet the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x2. When m is greater than 64, this parameter is not involved in the computation and can be empty.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>n, (1, 1), (1, n), or (n, 1)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale</td>
      <td>Input</td>
      <td>Input `mat2` in the formula after dequantization on x2. It is computed using the aclnnTransQuantParam API. For details about the compute method, see the sample code.</td>
      <td>The shape can be 1, n, (1, 1), (1, n), or (n, 1), and must meet the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x2. When m is greater than 64, this parameter is not involved in the computation and can be empty.</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>1, n, (1, 1), (1, n), or (n, 1)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>addOffset</td>
      <td>Input</td>
      <td>Input `mat2` in the formula after dequantization on x2.</td>
      <td>The shape can be 1, n, (1, 1), (1, n), or (n, 1), and must meet the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x2. When m is less than 64, this parameter is not involved in the computation and can be empty in any case.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1, n, (1, 1), (1, n), or (n, 1)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mulScale</td>
      <td>Input</td>
      <td>Input `mat2` in the formula after dequantization on x2.</td>
      <td>The shape can be 1, n, (1, 1), (1, n), or (n, 1), and must meet the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with x2. When m is less than 64, this parameter is not involved in the computation and can be empty in any case.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1, n, (1, 1), (1, n), or (n, 1)</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Input `bias` in the formula.</td>
      <td>The shape is one-dimensional and the value is N. It can be empty.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>Input</td>
      <td>Whether to transpose x1.</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Whether to transpose x2.</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScale</td>
      <td>Input</td>
      <td>Input `mat2` in the formula after dequantization on x2.</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffset</td>
      <td>Input</td>
      <td>Input `mat2` in the formula after dequantization on x2.</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>`result` in the formula.</td>
      <td>The data type must be convertible from the deduced data type of x1 and x2. The shape must be the broadcast result of x1 and x2.</td>
      <td>FLOAT16,INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
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

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
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
      <td>The passed x1, x2, diagonalMatrix (m < 64), deqOffset (m < 64), or deqScale (m < 64) is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of a passed non-empty tensor is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnWeightQuantBatchMatmul

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnWeightQuantBatchMatmulGetWorkspaceSize.</td>
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
- Deterministic description:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnWeightQuantBatchMatmul** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_trans_quant_param.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul.h"

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

  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> x1Shape = {128, 4};
  std::vector<int64_t> x2Shape = {4, 4};
  std::vector<int64_t> addOffsetShape = {4};
  std::vector<int64_t> mulScaleShape = {4};
  std::vector<int64_t> diagonalMatrixShape = {32, 32};
  std::vector<int64_t> deqOffsetShape = {4};
  std::vector<int64_t> deqScaleShape = {4};
  std::vector<int64_t> outShape = {128, 4};

  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* addOffsetDeviceAddr = nullptr;
  void* mulScaleDeviceAddr = nullptr;
  void* diagonalMatrixDeviceAddr = nullptr;
  void* deqOffsetDeviceAddr = nullptr;
  void* deqScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* x1Fp16DeviceAddr = nullptr;
  void* addOffsetFp16DeviceAddr = nullptr;
  void* mulScaleFp16DeviceAddr = nullptr;
  void* outFp16DeviceAddr = nullptr;

  std::vector<float> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<float> x2HostData = {1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1};

  std::vector<float> outHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  bool transposeX1 = false;
  bool transposeX2 = false;
  float antiquantOffset = 0;
  float antiquantScale = 1;

  std::vector<float> addOffsetHostData = {1.0, 1.0, 1.0, 0.0};
  float* addOffsetDate = addOffsetHostData.data();
  uint64_t addOffsetSize = 4;
  std::vector<float> mulScaleHostData = {2.0, 2.0, 1.0, 1.0};
  float* mulScaleDate = mulScaleHostData.data();
  uint64_t mulScaleSize = 4;

  // diagonalMatrixData
  uint64_t n = 32;
  uint64_t diagonalMatrixSize = n*n;
  int8_t *diagonalMatrixData = (int8_t *)calloc(diagonalMatrixSize, sizeof(int32_t));
  for (int64_t i = 0; i < n; i++) {
    diagonalMatrixData[i * n + i] = 1;
  }
  std::vector<int8_t> diagonalMatrixHostData(diagonalMatrixData, diagonalMatrixData + diagonalMatrixSize);

  // Get deqOffset
  uint64_t deqOffsetSize = addOffsetSize;
  int32_t *deqOffsetData = (int32_t *)calloc(deqOffsetSize, sizeof(int32_t));
  for (int64_t i = 0; i < deqOffsetSize; i++) {
    deqOffsetData[i] = static_cast<int32_t>(round(addOffsetDate[i] / antiquantScale - antiquantOffset));
  }
  std::vector<int32_t> deqOffsetHostData(deqOffsetData, deqOffsetData + deqOffsetSize);

  // Get deqScale
  uint64_t deqScaleSize = mulScaleSize;
  uint64_t *deqScaleData = (uint64_t *)calloc(deqScaleSize, sizeof(uint64_t));
  for (int64_t i = 0; i < deqScaleSize; i++) {
    mulScaleDate[i] = mulScaleDate[i] * antiquantScale;
  }
  std::vector<uint64_t> deqScaleHostData(deqScaleData, deqScaleData + deqScaleSize);

  // creat aclTensor
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* addOffset = nullptr;
  aclTensor* mulScale = nullptr;
  aclTensor* diagonalMatrix = nullptr;
  aclTensor* deqOffset = nullptr;
  aclTensor* deqScale = nullptr;
  aclTensor* out = nullptr;
  aclTensor* x1Fp16 = nullptr;
  aclTensor* addOffsetFp16 = nullptr;
  aclTensor* mulScaleFp16 = nullptr;
  aclTensor* outFp16 = nullptr;

  // Create an x1 aclTensor.
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an x1Fp16 aclTensor.
  ret = CreateAclTensor(x1HostData, x1Shape, &x1Fp16DeviceAddr, aclDataType::ACL_FLOAT16, &x1Fp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an x2 aclTensor.
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an addOffset aclTensor.
  ret = CreateAclTensor(addOffsetHostData, addOffsetShape, &addOffsetDeviceAddr, aclDataType::ACL_FLOAT, &addOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an addOffsetFp16 aclTensor.
  ret = CreateAclTensor(addOffsetHostData, addOffsetShape, &addOffsetFp16DeviceAddr, aclDataType::ACL_FLOAT16, &addOffsetFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a mulScale aclTensor.
  ret = CreateAclTensor(mulScaleHostData, mulScaleShape, &mulScaleDeviceAddr, aclDataType::ACL_FLOAT, &mulScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a mulScaleFp16 aclTensor.
  ret = CreateAclTensor(mulScaleHostData, mulScaleShape, &mulScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &mulScaleFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a diagonalMatrix aclTensor.
  ret = CreateAclTensor(diagonalMatrixHostData, diagonalMatrixShape, &diagonalMatrixDeviceAddr, aclDataType::ACL_INT8, &diagonalMatrix);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a deqOffset aclTensor.
  ret = CreateAclTensor(deqOffsetHostData, deqOffsetShape, &deqOffsetDeviceAddr, aclDataType::ACL_INT32, &deqOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a deqScale aclTensor.
  ret = CreateAclTensor(deqScaleHostData, deqScaleShape, &deqScaleDeviceAddr, aclDataType::ACL_UINT64, &deqScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an outFp16 aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outFp16DeviceAddr, aclDataType::ACL_FLOAT16, &outFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // aclnnWeightQuantBatchMatmul API call example
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // aclnn cast fp16
  //x1
  ret = aclnnCastGetWorkspaceSize(x1, aclDataType::ACL_FLOAT16, x1Fp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // addOffset
  ret = aclnnCastGetWorkspaceSize(addOffset, aclDataType::ACL_FLOAT16, addOffsetFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // mulScale
  ret = aclnnCastGetWorkspaceSize(mulScale, aclDataType::ACL_FLOAT16, mulScaleFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Call the first-phase API of aclnnWeightQuantBatchMatmul.
  ret = aclnnWeightQuantBatchMatmulGetWorkspaceSize(x1Fp16, x2, diagonalMatrix, deqOffset, deqScale, addOffsetFp16, mulScaleFp16, nullptr, transposeX1, transposeX2, antiquantScale, antiquantOffset, outFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);


  // Allocate device memory based on workspaceSize computed by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnWeightQuantBatchMatmul.
  ret = aclnnWeightQuantBatchMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // fp16 to fp 32
  ret = aclnnCastGetWorkspaceSize(outFp16, aclDataType::ACL_FLOAT, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(x1);
  aclDestroyTensor(x1Fp16);
  aclDestroyTensor(x2);
  aclDestroyTensor(addOffset);
  aclDestroyTensor(addOffsetFp16);
  aclDestroyTensor(mulScaleFp16);
  aclDestroyTensor(mulScale);
  aclDestroyTensor(out);
  aclDestroyTensor(outFp16);


  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(addOffsetDeviceAddr);
  aclrtFree(deqScaleDeviceAddr);
  aclrtFree(mulScaleDeviceAddr);
  aclrtFree(diagonalMatrixDeviceAddr);
  aclrtFree(deqOffsetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(x1Fp16DeviceAddr);
  aclrtFree(addOffsetFp16DeviceAddr);
  aclrtFree(mulScaleFp16DeviceAddr);
  aclrtFree(outFp16DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  free(diagonalMatrixData);
  free(deqOffsetData);
  free(deqScaleData);
  return 0;
}
```
