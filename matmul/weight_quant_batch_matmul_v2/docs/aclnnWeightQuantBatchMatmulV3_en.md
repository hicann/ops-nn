# aclnnWeightQuantBatchMatmulV3

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/weight_quant_batch_matmul_v2)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    ×    |

## Function

- **Description**: Performs input matrix multiplication in a fake-quantization scenario and implements output quantization. Compared with the **aclnnWeightQuantBatchMatmulV2** API, this API has the following changes:

  The **innerPrecise** parameter is added to support the selection of high-precision or high-performance compute mode. To improve performance in the A16W4 per_group scenario, this parameter can be set to **1** when **batchSize** is less than or equal to **16**.
- **Formula**:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  In the formula, $weight$ is the input of the fake-quantization scenario, and the dequantization formula $ANTIQUANT(weight)$ is as follows:

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  When quantScaleOptional is configured, the output is quantized using the following formula:

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  If quantScaleOptional is set to nullptr, the out is as follows:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnWeightQuantBatchMatmulV3GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnWeightQuantBatchMatmulV3** is called to perform computation.
```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *weight, 
  const aclTensor *antiquantScale, 
  const aclTensor *antiquantOffsetOptional, 
  const aclTensor *quantScaleOptional, 
  const aclTensor *quantOffsetOptional, 
  const aclTensor *biasOptional, 
  int              antiquantGroupSize, 
  int              innerPrecise, 
  const aclTensor *y, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV3(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)`
```

## aclnnWeightQuantBatchMatmulV3GetWorkspaceSize

- **Parameters**
  <table style="table-layout: fixed; width: 1550px">
    <colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
    </colgroup>
    <thread>
      <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Input `x` in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>The shape can be two-dimensional (m, k), where the Reduce dimension **k** must be the same as that of `weight`.</td>
        <td>Non-contiguous tensors are supported only in the transpose scenario.</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Input `weight` in the formula. When the weight data format is FRACTAL_NZ and the data type is INT4 or INT32, or when the weight data format is ND and the data type is INT32, this parameter is supported only in the INT4Pack scenario. The aclnnConvertWeightToINT4Pack API needs to be used for INT32-to-INT4Pack conversion and ND-to-FRACTAL_NZ conversion. For details, see the <a href="../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack_en.md" target="_blank">example</a>. If the data type is INT4, the inner axis of weight must be an even number.
        <td>For different fake-quantization algorithm modes, the weight data format FRACTAL_NZ is supported only in the following scenarios:
        <ul><li>per_channel mode:
              <ul><li>The weight data type is INT8, and the y data type is not INT8.</li>
              <li>The weight data type is INT4 or INT32, weight is transposed, and the y data type is not INT8.</li></ul>
            <li>per_group mode: The weight data type is INT4 or INT32, weight and x are not transposed, antiquantGroupSize is 64 or 128, k is a multiple of antiquantGroupSize, n is a multiple of 64, and the y data type is not INT8.</li></ul></td>
        <td>INT8, INT4, INT32</td>
        <td>ND, FRACTAL_NZ</td>
        <td>(k, n) is supported.</td>
        <td>Non-contiguous tensors are supported only in the transpose scenario.</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>Input</td>
        <td>Dequantization scale parameter and the input `antiquantScale` in the dequantization formula.</td>
        <td>When the data type is FLOAT16 or BFLOAT16, the data type must be the same as that of the input `x`. When the data type is UINT64 or INT64, x supports only FLOAT16 and is not transposed, weight supports only int8 and is transposed in ND format, only the per_channel mode is supported, quantScaleOptional and quantOffsetOptional must be empty, m value range supports only [1, 96], and k and n must be multiples of 64. The aclnnCast API needs to be used to to perform FLOAT16-to-FLOAT32 conversion, and then the aclnnTransQuantParamV2 API needs to be used to perform FLOAT32-to-UINT64 conversion. For details, see <a href="../../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2_en.md" target="_blank">TransQuantParamV2</a>.</td>
        <td>FLOAT16, BFLOAT16, UINT64, INT64</td>
        <td>ND</td>
        <td>
          <ul>
            <li>per_tensor mode: The input shape is (1,) or (1, 1).</li>
            <li>per_channel mode: The input shape is (1, n) or (n,).</li>
            <li>per_group mode: The input shape is (ceil(k, group_size), n).</li>
          </ul>
        </td>
        <td>Non-contiguous tensors are supported only in the transpose scenario.</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional</td>
        <td>Input</td>
        <td>Dequantization offset parameter and the input `antiquantOffset` in the dequantization formula.</td>
        <td>When the data type is FLOAT16 or BFLOAT16, the data type must be the same as that of the input `x`. When the data type is INT32, the data range is limited to [–128, 127]. x supports only FLOAT16, weight supports only int8, and antiquantScale supports only UINT64 and INT64. It is an optional parameter. When it is not required, pass a null pointer to it.</td>
        <td>FLOAT16, BFLOAT16, INT32</td>
        <td>ND</td>
        <td>Must be the same as that of `antiquantScale`.</td>
        <td>Non-contiguous tensors are supported only in the transpose scenario.</td>
      </tr>
      <tr>
        <td>quantScaleOptional</td>
        <td>Input</td>
        <td>Quantization parameter, which is converted from the data of quantScale and quantOffset in the quantization formula through the `aclnnTransQuantParam` API.
        <td>Converted from the data of `quantScale` and `quantOffset` in the quantization formula through the `aclnnTransQuantParam` API.</td>
        <td>UINT64</td>
        <td>ND</td>
        <td>
          <ul>
            <li>per_tensor mode: The input shape is (1, ) or (1, 1).</li>
            <li>per_channel mode: The input shape is (1, n) or (n,).</li>
          </ul></td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffsetOptional</td>
        <td>Input</td>
        <td>Quantization offset parameter, the input `quantOffset` in the quantization formula.</td>
        <td>It is an optional parameter. When it is not required, pass a null pointer to it.</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>Same as `quantScaleOptional`.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>Bias input, `bias` in the formula. When the data type of `x` is BFLOAT16, the data type of this parameter must be FLOAT. When the data type of `x` is FLOAT16, the data type of this parameter must be FLOAT16.</td>
        <td>It is an optional parameter. When it is not required, pass a null pointer to it.</td>
        <td>FLOAT, FLOAT16</td>
        <td>ND</td>
        <td>1-2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantGroupSize</td>
        <td>Input</td>
        <td>groupSize input for dequantizing the input `weight` in per_group mode of the fake quantization algorithm. It describes the size of the data to be dequantized corresponding to a group of dequantization parameters in the Reduce direction. If the fake-quantization algorithm mode is not per_group, pass 0. If the fake-quantization algorithm mode is per_group, the value range is [32, k – 1] and the value must be a multiple of 32.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Whether fake quantization is in high-precision or high-performance computing mode. Only 0 or 1 can be passed. To improve performance in the A16W4 per_group scenario when batchSize is less than or equal to 16, this parameter can be set to 1 and the weight data format can be set to FRACTAL_NZ. In other scenarios, this parameter is not recommended, and you are advised to pass 0.</td>
        <td>-</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
       <tr>
        <td>y</td>
        <td>Output</td>
        <td>Compute output, corresponding to `y` in the formula.</td>
        <td>If `quantScaleOptional` exists, the data type is INT8. If `quantScaleOptional` does not exist, the data type can be FLOAT16 or BFLOAT16, and must be the same as the data type of the input `x`.</td>
        <td>INT8,FLOAT16,BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace required to be allocated on the device.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </tbody>
  </table>

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
      <td>The mandatory input is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="14">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="14">161002</td>
      <td>The shape dimensions of the passed x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional, or y do not meet requirements.</td>
    </tr>
    <tr>
      <td>The data type of the passed x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional, or y is not supported.</td>
    </tr>
    <tr>
      <td>The reduce dimensions (k) of x and weight are different.</td>
    </tr>
    <tr>
      <td>When antiquantOffsetOptional exists, the shape is different from that of antiquantScale.</td>
    </tr>
    <tr>
      <td>When quantOffsetOptional exists, the shape is different from that of quantScale.</td>
    </tr>
    <tr>
      <td>The shape of biasOptional does not meet requirements.</td>
    </tr>
    <tr>
      <td>The value of antiquantGroupSize does not meet requirements.</td>
    </tr>
    <tr>
      <td>The value of innerPrecise does not meet requirements.</td>
    </tr>
    <tr>
      <td>When quantOffsetOptional exists, quantScaleOptional is a null pointer.</td>
    </tr>
    <tr>
      <td>The input k and n values are not within the [1, 65535] range.</td>
    </tr>
    <tr>
      <td>When the x matrix is not transposed, m is not in the range of [1, 2^31-1]. When the x matrix is transposed, m is not in the range of [1, 65535].</td>
    </tr>
    <tr>
      <td>The tensor is empty, which is not supported.</td>
    </tr>
    <tr>
      <td>The data format of the input tensor is not supported.</td>
    </tr>
    <tr>
      <td>The continuity of the input x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional and y does not meet the requirements.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The product model is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnWeightQuantBatchMatmulV3

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnWeightQuantBatchMatmulV3GetWorkspaceSize.</td>
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
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnWeightQuantBatchMatmulV3** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

- per_channel mode: To improve performance, you are advised to use the **weight** input after transpose. If the value range of m is [65, 96], **antiquantScale** of the UINT64/INT64 data type is recommended.

- per_group mode: In the A16W4 scenario where **batchSize** is less than or equal to 16, you can set **innerPrecise** to **1** and set the **weight** data format to FRACTAL_NZ to improve performance, but the accuracy may drop.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_v3.h"

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
  std::vector<int64_t> xShape = {16, 32};
  std::vector<int64_t> weightShape = {32, 16};
  std::vector<int64_t> yShape = {16, 16};
  void* xDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* y = nullptr;
  int32_t innerPrecise = 1;
  std::vector<float> xHostData(512, 1);
  std::vector<int8_t> weightHostData(512, 1);
  std::vector<float> yHostData(256, 0);

  std::vector<int64_t> antiquantScaleShape = {16};
  void* antiquantScaleDeviceAddr = nullptr;
  aclTensor* antiquantScale = nullptr;
  std::vector<float> antiquantScaleHostData(16, 1);


  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an other aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a y aclTensor.
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an antiquantScale aclTensor.
  ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an xFp16 aclTensor.
  void* xFp16DeviceAddr = nullptr;
  aclTensor* xFp16 = nullptr;
  ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an antiquantScale aclTensor.
  void* antiquantScaleFp16DeviceAddr = nullptr;
  aclTensor* antiquantScaleFp16 = nullptr;
  ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a yFp16 aclTensor.
  void* yFp16DeviceAddr = nullptr;
  aclTensor* yFp16 = nullptr;
  ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  void* workspaceAddr = nullptr;

  // Call cast to generate the FP16 input.
  ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Call the first-phase API of aclnnWeightQuantBatchMatmulV3.
  ret = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(xFp16, weight, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, innerPrecise, yFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnWeightQuantBatchMatmulV3.
  ret = aclnnWeightQuantBatchMatmulV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3 failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

 // Convert the output into FP32.
  ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(antiquantScale);
  aclDestroyTensor(y);
  aclDestroyTensor(xFp16);
  aclDestroyTensor(antiquantScaleFp16);
  aclDestroyTensor(yFp16);

  // 7. Release device resources.
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(antiquantScaleDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(xFp16DeviceAddr);
  aclrtFree(antiquantScaleFp16DeviceAddr);
  aclrtFree(yFp16DeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
