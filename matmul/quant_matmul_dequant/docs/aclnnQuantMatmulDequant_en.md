# aclnnQuantMatmulDequant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:54:08.043Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id7 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id7 -->
<!-- npu="310b" id3 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Not supported
<!-- end id5 -->

## Function

- Description: performs quantization, matmul, and dequantization on input x.
- Formula:
  1 If smoothScaleOptional is passed in,

  $$
      x = x\cdot scale_{smooth}
  $$

  2 If xScaleOptional is not passed in, dynamic quantization is used, and the quantization coefficient of x needs to be calculated.

  $$
      scale_{x}=row\_max(abs(x))/max_{quantDataType}
  $$

  3 Quantization

  $$
      x_{quantized}=round(x/scale_{x})
  $$

  4 Matmul+Dequantization

  - 4.1 If the data type of the input $scale_{weight}$ is FLOAT32:

    $$
      out = (x_{quantized}@weight_{quantized} + bias) * scale_{weight} * scale_{x}
    $$

  - 4.2 If the data type of the input $scale_{weight}$ is INT64:

    $$
      scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.int32).tobytes(), dtype=np.float32)) \\
      out = (x_{quantized}@weight_{quantized} + bias) * scale_{weight}
    $$

    Special note: In the preceding 4.2 use case, $scale_{weight}$ has already been multiplied by $scale_{x}$ before input, so the operator omits this step during internal computation. This requires the pertensor static quantization use case. That is, $scale_{weight}$ must be processed as follows before input to obtain data of the INT64 type:

    $$
    scale_{weight} = scale_{weight} * scale_{x} \\
    scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.float32). \\tobytes(), dtype=np.int32).astype(np.int64))
    $$

## Function Prototype

Each operator is split into a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnQuantMatmulDequantGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the process, and then call the "aclnnQuantMatmulDequant" API to perform the computation.

```Cpp
aclnnStatus aclnnQuantMatmulDequantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *weightScale,
  const aclTensor *biasOptional,
  const aclTensor *xScaleOptional,
  const aclTensor *xOffsetOptional,
  const aclTensor *smoothScaleOptional,
  char            *xQuantMode,
  bool             transposeWeight,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantMatmulDequant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulDequantGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1491px"><colgroup>
  <col style="width: 201px">
  <col style="width: 120px">
  <col style="width: 240px">
  <col style="width: 350px">
  <col style="width: 177px">
  <col style="width: 120px">
  <col style="width: 138px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the left matrix of the input, which is x in the formula.</td>
      <td><ul><li>The shape supports 2 dimensions, and each dimension indicates (m, k).</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the right matrix of the input, which is weight_{quantized} in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8</td>
      <td>FRACTAL_NZ, ND</td>
      <td>2, 4</td>
      <td>√</td>
    </tr>
      <tr>
      <td>weightScale (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the quantization scale of weight, scale_{weight} in the formula.</td>
      <td><ul><li>The shape is 1D (n,), where n is consistent with the n of weight.</li><li>Empty tensors are supported.</li><li>When the data type is INT64, the data type of xScaleOptional must be FLOAT16 and the value of xQuantMode must be pertensor.</li></ul></td>
      <td>FLOAT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>biasOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the bias of the calculation, which is bias in the formula.</td>
      <td>Currently, only null pointers are supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>xScaleOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the quantization factor of x, that is, scale_{x} in the formula.</td>
      <td><ul><li>When xQuantMode is pertensor, the shape is 1 dimension (1,); when xQuantMode is pertoken, the shape is 1 dimension (m,), where m is consistent with m of input x. If empty, dynamic quantization is used.</li><li>Supports empty tensor.</li><li>When the data type is FLOAT16, this parameter must require the data type of weightScale to be INT64 and the value of xQuantMode to be pertensor.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
       <tr>
      <td>xOffsetOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the offset of x.</td>
      <td>Currently, only null pointers are supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>smoothScaleOptional (aclTensor*)</td>
      <td>Optional input</td>
      <td>Indicates the smooth factor of x, which is scale_{smooth} in the formula.</td>
      <td><ul><li>shape is 1D (k,), where k is consistent with the k of x.</li><li>Supports empty tensor.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>xQuantMode (char*)</td>
      <td>Input</td>
      <td>Specifies the quantization mode of input x.</td>
      <td>The value can be pertoken/pertensor. For dynamic quantization, only pertoken is supported. pertoken indicates that each token (a row) has its own quantization parameters; pertensor indicates that the entire tensor uses unified quantization parameters.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeWeight (bool)</td>
      <td>Input</td>
      <td>Indicates whether the input weight is transposed.</td>
      <td>Currently only true is supported.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Calculation result, the out in the formula.</td>
      <td>Supports 2 dimensions, with dimensions: (m, n). m is consistent with the m of x, and n is consistent with the n of weight.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
       <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

    - For the weight parameter in ND format, the shape supports 2 dimensions.
      - When transposeWeight is true, the dimensions indicate: (n, k).
      - When transposeWeight is false, the dimensions indicate: (k, n).
    - For the weight parameter in FRACTAL_NZ format, the shape supports 4 dimensions.
      - When transposeWeight is true, the dimensions indicate: (k1, n1, n0, k0), where k0 = 32, n0 = 16, and k1 must meet the following relationship with k of x: ceilDiv(k, 32) = k1.
      - When transposeWeight is false, each dimension indicates: (n1, k1, k0, n0), where k0 = 16, n0 = 32, and k1 and k of x must satisfy the following relationship: ceilDiv(k, 16) = k1.
      - Use the aclnnCalculateMatmulWeightSizeV2 API and the aclnnTransMatmulWeight API to convert the input format from ND to FRACTAL_NZ.

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If the passed-in parameter is a required input, output, or required attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td><ul><li>If a parameter of the aclTensor type is passed in and its data type is not within the supported range.</li><li>n or k in the shape of weight is not divisible by 16.</li></ul></td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>If a parameter of the aclTensor type is passed in and its shape is inconsistent with the preceding parameter descriptions.</td>
    </tr>
  </tbody>
  </table>

## aclnnQuantMatmulDequant

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied on the Device side, obtained by the first-phase API aclnnQuantMatmulDequantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the computation process of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="910,310p" id6 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: aclnnQuantMatmulDequant defaults to a deterministic implementation.

  <!-- end id6 -->

- Both n and k must be multiples of 16.
- When the data type of weightScale is INT64, the data type of xScaleOptional must be FLOAT16 and the value of xQuantMode must be pertensor; when the data type of xScaleOptional is FLOAT16, the data type of weightScale must be INT64 and the value of xQuantMode must be pertensor.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_matmul_dequant.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: resource initialization
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
  // Call aclrtMalloc to apply for memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of a contiguous tensor.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. You need to customize the construction based on the API.
  int M = 64;
  int K = 256;
  int N = 512;

  char quantMode[16] = "pertoken";
  bool transposeWeight = true;

  std::vector<int64_t> xShape = {M,K};
  std::vector<int64_t> weightShape = {N,K};
  std::vector<int64_t> weightScaleShape = {N};
  std::vector<int64_t> xScaleShape = {M};
  std::vector<int64_t> smoothScaleShape = {K};
  std::vector<int64_t> outShape = {M,N};

  void* xDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* weightScaleDeviceAddr = nullptr;
  void* xScaleDeviceAddr = nullptr;
  void* smoothScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* weightScale = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* xScale = nullptr;
  aclTensor* xOffset = nullptr;
  aclTensor* smoothScale = nullptr;
  aclTensor* out = nullptr;

  std::vector<uint16_t> xHostData(GetShapeSize(xShape));
  std::vector<uint8_t> weightHostData(GetShapeSize(weightShape));
  std::vector<uint16_t> weightScaleHostData(GetShapeSize(weightScaleShape));
  std::vector<uint16_t> xScaleHostData(GetShapeSize(xScaleShape));
  std::vector<uint16_t> smoothScaleHostData(GetShapeSize(smoothScaleShape));
  std::vector<uint16_t> outHostData(GetShapeSize(outShape));

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT, &weightScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT, &xScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(smoothScaleHostData, smoothScaleShape, &smoothScaleDeviceAddr, aclDataType::ACL_FLOAT16, &smoothScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, and replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnQuantMatmulDequant.
  ret = aclnnQuantMatmulDequantGetWorkspaceSize(x, weight, weightScale,
                                                bias, xScale, xOffset, smoothScale,
                                                quantMode, transposeWeight, out,
                                                &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulDequantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnQuantMatmulDequant.
  ret = aclnnQuantMatmulDequant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulDequant failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, and copy the result from device memory to host memory. Modify the code based on the specific API definition.
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(weightScale);
  aclDestroyTensor(xScale);
  aclDestroyTensor(smoothScale);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(weightScaleDeviceAddr);
  aclrtFree(xScaleDeviceAddr);
  aclrtFree(smoothScaleDeviceAddr);
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
