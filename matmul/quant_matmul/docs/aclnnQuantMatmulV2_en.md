# aclnnQuantMatmulV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:40:07.203Z -->

**Notice: This API will be deprecated in later versions. Use the latest aclnnQuantMatmulV5 API instead.**

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Description: Performs quantized matrix multiplication. Supports a maximum input dimension of 3. Similar APIs include [aclnnMm](../../mat_mul_v3/docs/aclnnMm_en.md) (matrix multiplication supporting only 2D tensors as input) and [aclnnBatchMatMul](../../batch_mat_mul_v3/docs/aclnnBatchMatMul_en.md) (matrix multiplication supporting only 3D tensors, where the first dimension is the batch dimension).
- Formula:

$$
out = (x1@x2 + bias) * deqScale
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the aclnnQuantMatmulV2GetWorkspaceSize API to obtain the workspace size required for computation and the executor containing the operator computation process, and then call the aclnnQuantMatmulV2 API to execute the computation.

```cpp
aclnnStatus aclnnQuantMatmulV2GetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *bias,
  const aclTensor *deqScale,
  bool             adjX1,
  bool             adjX2,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulV2(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  const aclrtStream  stream)
```

## aclnnQuantMatmulV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1535px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 250px">
  <col style="width: 350px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 130px">
  <col style="width: 153px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data type</th>
      <th>Data format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Input x1 in the formula.</td>
      <td>If adjX1 is false, the dimensions are (batch, m, k).<br>If adjX1 is true, the dimensions are (batch, k, m), where batch can be omitted.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 in the formula.</td>
      <td>If adjX2 is false, the dimensions denote (batch, k, n).<br>If adjX2 is true, the dimensions denote (batch, n, k); batch may be absent, and k is consistent with k in the shape of x1.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>The input bias in the formula.</td>
      <td>The shape is 1D (n,). n is consistent with n of x2.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale</td>
      <td>Input</td>
      <td>Quantization parameter. Input deqScale in the formula.</td>
      <td>1D shape (t,). t = align(n, 16), where n is consistent with the n of x2.</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>adjX1</td>
      <td>Input</td>
      <td>Indicates whether the input shape of x1 includes transpose.</td>
      <td>When adjX1 is false, the dimensions are (batch, m, k).<br>When adjX1 is true, the dimensions are (batch, k, m), and batch is optional.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>adjX2</td>
      <td>Input</td>
      <td>Indicates whether the input shape of x2 contains transpose.</td>
      <td>When adjX2 is false, dimensions indicate: (batch, k, n).<br>When adjX2 is true, dimensions indicate: (batch, n, k), and batch can be absent.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output out in the formula.</td>
      <td>(batch, m, n), where batch is optional. The batch dimensions of x1 and x2 support broadcasting. The output batch is consistent with the batch after broadcasting. m and n are consistent with the m of x1 and the n of x2, respectively.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
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
      <td>The passed-in x1, x2, deqScale, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type or data format of x1, x2, bias, deqScale, or out is out of the supported range.</td>
    </tr>
    <tr>
      <td>The shape of x1 or x2 is not 2D or 3D.</td>
    </tr>
    <tr>
      <td>The shape of bias or deqScale is not 1D.</td>
    </tr>
    <tr>
      <td>The size of deqScale is not the n size in x2 rounded up to a multiple of 16.</td>
    </tr>
    <tr>
      <td>bias exists and is an empty tensor where m and n are both non-zero but k is 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnQuantMatmulV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>Memory address of the workspace allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnQuantMatmulV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: aclnnQuantMatmulV2 defaults to a deterministic implementation.

  <!-- end id7 -->

Method for migrating this API to aclnnQuantMatmulV4:

- Inputs x1, x2, bias, adjX1, and adjX2 can be directly mapped to x1, x2, bias, transposeX1, and transposeX2 in the aclnnQuantMatmulV4 API.
- The input deqScale is a UINT64 aclTensor, and its data type is consistent with scale in the aclnnQuantMatmulV4 API. The deqScale shape of the aclnnQuantMatmulV2 API is 1D (t,), where t = align(n, 16). The scale shape in the aclnnQuantMatmulV4 API is 1D (t,), where t = 1 or n. Call aclnnTransQuantParam to directly convert the original FLOAT quantization parameters into a UINT64 aclTensor with shape (n,) (see [aclnnQuantMatmulV4 Example](../../quant_batch_matmul_v3/docs/aclnnQuantMatmulV4_en.md#example)), denoted as **scale**, which corresponds to scale in the aclnnQuantMatmulV4 API.
- The optional input offset/pertokenScaleOptional in the aclnnQuantMatmulV4 API is set to nullptr.
- The API parameters are set to `aclnnQuantMatmulV4GetWorkspaceSize(x1, x2, scale, nullptr, nullptr, bias, adjX1, adjX2, out, workspaceSize, executor)`.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <memory>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_matmul.h"
#include "aclnnop/aclnn_trans_quant_param.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      Finalize(deviceId, stream);\
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host side to device-side memory.
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

void Finalize(int32_t deviceId, aclrtStream stream) {
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnQuantMatmulV2Test(int32_t deviceId, aclrtStream &stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> x1Shape = {5, 2};
  std::vector<int64_t> x2Shape = {2, 3};
  std::vector<int64_t> biasShape = {3};
  std::vector<int64_t> outShape = {5, 3};
  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* out = nullptr;
  std::vector<int8_t> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int8_t> x2HostData = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> biasHostData = {1, 1, 1};
  std::vector<uint16_t> outHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // Actually in float16 half precision.
  // Create the x1 aclTensor.
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the x2 aclTensor.
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the bias aclTensor.
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  bool adjX1 = false;
  bool adjX2 = false;

  // Convert the scale data from float to the uint64_t type required by the hardware, and create the corresponding tensor as the deqScale input parameter.
  float scaleArray[3] = {1.0, 1.0, 1.0};
  uint64_t scaleSize = 3;
  float *offsetArray = nullptr;
  uint64_t offsetSize = 0;
  uint64_t *deqScaleArray = nullptr;
  uint64_t deqScaleSize = 0;
  ret = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &deqScaleArray, &deqScaleSize);
  std::unique_ptr<uint64_t> deqScaleArrayPtr(deqScaleArray);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParam failed. ERROR: %d\n", ret); return ret);

  std::vector<uint64_t> deqScaleVector(deqScaleArray, deqScaleArray + deqScaleSize);
  std::vector<int64_t> deqScaleShape = {deqScaleSize};
  void* deqScaleDeviceAddr = nullptr;
  aclTensor* deqScale = nullptr;
  ret = CreateAclTensor(deqScaleVector, deqScaleShape, &deqScaleDeviceAddr, aclDataType::ACL_UINT64, &deqScale);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> deqScaleTensorPtr(deqScale, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deqScaleDeviceAddrPtr(deqScaleDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call a CANN operator library API. Change it to the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnQuantMatmulV2.
  ret = aclnnQuantMatmulV2GetWorkspaceSize(x1, x2, bias, deqScale, adjX1, adjX2, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Request device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnQuantMatmulV2.
  ret = aclnnQuantMatmulV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from device memory to host memory, and modify it based on the definition of the specific API.
  auto size = GetShapeSize(outShape);
  std::vector<uint16_t> resultData(size, 0); // The fp16 data cannot be printed directly in the C language. Read it as uint16 and manually convert the binary data to fp16.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Fill in deviceId based on your actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnQuantMatmulV2Test(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV2Test failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}

```
