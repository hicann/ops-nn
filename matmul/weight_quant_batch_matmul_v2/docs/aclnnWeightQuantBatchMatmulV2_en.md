# aclnnWeightQuantBatchMatmulV2

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/weight_quant_batch_matmul_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- **Description**: Performs input matrix multiplication in a fake-quantization scenario and implements output quantization.
- **Formula**:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  In the formula, $weight$ is the input of the fake-quantization scenario, and the dequantization formula $ANTIQUANT(weight)$ is as follows:

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  When the output needs to be quantized, the quantization formula is as follows:

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  When the output does not need to be quantized, the formula is as follows:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnWeightQuantBatchMatmulV2GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnWeightQuantBatchMatmulV2** is called to perform computation.
  - `aclnnStatus aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *antiquantScale, const aclTensor *antiquantOffsetOptional, const aclTensor *quantScaleOptional, const aclTensor *quantOffsetOptional, const aclTensor *biasOptional, int antiquantGroupSize, const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnWeightQuantBatchMatmulV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnWeightQuantBatchMatmulV2GetWorkspaceSize

- **Parameters**

  - **x** (aclTensor *, compute input): left input matrix of matrix multiplication, input `x` in the formula, and aclTensor on the device. The [data format](../../../docs/en/context/data_formats.md) can be ND. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported only in the transpose scenario.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: Two dimensions are supported. The shape can be (m, k), where **m** indicates the size of the first dimension of the matrix, and **k** indicates the size of the second dimension of the matrix. The Reduce dimension **k** must be the same as the Reduce dimension **k** of `weight`. The data type can be FLOAT16 or BFLOAT16. When the `x` matrix is not transposed, **m** is in the range of [1, 2^31-1]. When the `x` matrix is transposed, **m** is in the range of [1, 65535].
    - <term>Atlas inference series products</term>: The data type can be FLOAT16. The shape can be two- to six-dimensional. The input shape must be (batch, m, k), where **batch** indicates the batch size of the matrix and can be zero- to four-dimensional. **m** indicates the size of the first dimension of the single batch matrix, and **k** indicates the size of the second dimension of the single batch matrix. The batch dimension must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with the batch dimension of `weight`. When the fake quantization algorithm mode is pertensor [quantization mode](../../../docs/en/context/quantization_introduction.md), m × k cannot exceed 512000000.

  - **weight** (aclTensor *, compute input): right input matrix of matrix multiplication, input `weight` in the formula, and aclTensor on the device. The [data format](../../../docs/en/context/data_formats.md) can be ND or FRACTAL_NZ.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: Two dimensions are supported. The reduce dimension k must be the same as the reduce dimension k of `x`. The data type can be INT8, INT4, or INT32. When the `weight` [data format](../../../docs/en/context/data_formats.md) is FRACTAL_NZ and the data type is INT4 or INT32, or when the `weight` [data format](../../../docs/en/context/data_formats.md) is ND and the data type is INT32, this parameter is supported only in the INT4Pack scenario. `aclnnConvertWeightToINT4Pack` is also used for INT32-to-INT4Pack conversion and ND-to-FRACTAL_NZ conversion. For details, see the [example](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack_en.md). If the data type is INT4, the inner axis of `weight` must be an even number. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported only in the transpose scenario. The shape can be (k, n), where **k** indicates the size of the first dimension of the matrix, and **n** indicates the size of the second dimension of the matrix.
      For different fake-quantization algorithm modes, the `weight` [data format](../../../docs/en/context/data_formats.md) FRACTAL_NZ is supported only in the following scenarios:
      - perchannel [quantization mode](../../../docs/en/context/quantization_introduction.md):
        - The `weight` data type is INT8, and the y data type is not INT8.
        - The `weight` data type is INT4 or INT32, `weight` is transposed, and the y data type is not INT8.
      - pergroup [quantization mode](../../../docs/en/context/quantization_introduction.md): The `weight` data type is INT4 or INT32, `weight` and `x` are not transposed, antiquantGroupSize is 64 or 128, k is a multiple of antiquantGroupSize, n is a multiple of 64, and the y data type is not INT8.
    - <term>Atlas inference series products</term>: Two to six dimensions are supported. The batch dimension must meet the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md) with the batch dimension of `x`. The data type can be INT8. Details are as follows:
      - If the [data format](../../../docs/en/context/data_formats.md) is ND, the input shape must be (batch, k, n), where **batch** indicates the batch size of the matrix and can be zero- to four-dimensional, **k** indicates the size of the first dimension of the single batch matrix, and **n** indicates the size of the second dimension of the single batch matrix.
      - If the [data format](../../../docs/en/context/data_formats.md) is FRACTAL_NZ:
        - The input shape must be (batch, n, k), where **batch** indicates the batch size of the matrix and can be zero- to four-dimensional, **k** indicates the size of the first dimension of the single batch matrix, and **n** indicates the size of the second dimension of the single batch matrix.
        - **aclnnCalculateMatmulWeightSizeV2** and **aclnnTransMatmulWeight** are also used to convert the input format from ND to FRACTAL_NZ. For details, see the [example].
  - **antiquantScale** (aclTensor *, compute input): dequantization scale parameter, the input `antiquantScale` in the dequantization formula, and aclTensor on the device. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The supported data types are FLOAT16, BFLOAT16, UINT64, and INT64. When the data type is FLOAT16 or BFLOAT16, the data type must be the same as that of the input `x`. When the data type is UINT64 or INT64, `x` supports only FLOAT16 and is not transposed, `weight` supports only INT8 and is transposed in ND format, and the quantization mode must be perchannel [quantization mode](../../../docs/en/context/quantization_introduction.md). Null pointers must be passed for **quantScaleOptional** and **quantOffsetOptional**. The value of **m** ranges from 1 to 96. The values of **k** and **n** must be multiples of 64. First, the **aclnnCast** API is used to perform the FLOAT16-to-FLOAT32 conversion. For details, see [Cast]. Then, the **aclnnTransQuantParamV2** API is used to perform the FLOAT32-to-UINT64 conversion. For details, see [TransQuantParamV2](../../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2_en.md). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported only in the transpose scenario.
      For different fake-quantization algorithm modes, `antiquantScale` supports the following shapes:
      - pertensor [quantization mode](../../../docs/en/context/quantization_introduction.md): The input shape is (1,) or (1, 1).
      - perchannel [quantization mode](../../../docs/en/context/quantization_introduction.md): The input shape is (1, n) or (n,).
      - pergroup [quantization mode](../../../docs/en/context/quantization_introduction.md): The input shape is (⌈k/group_size⌉, n), where **group_size** indicates the size of each group to which **k** is to be grouped.
    - <term>Atlas inference series products</term>: The supported data type is FLOAT16. The data type must be the same as that of the input `x`.
      For different fake-quantization algorithm modes, `antiquantScale` supports the following shapes:
      - pertensor [quantization mode](../../../docs/en/context/quantization_introduction.md): The input shape is (1,) or (1, 1).
      - perchannel [quantization mode](../../../docs/en/context/quantization_introduction.md): The input shape is (n, 1) or (n, ). [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported.
      - pergroup [quantization mode](../../../docs/en/context/quantization_introduction.md): The input shape is related to the data format of `weight` as follows:
        - When the data format of `weight` is ND, the input shape is (⌈k/group_size⌉, n), where **group_size** indicates the size of each group to which **k** is to be grouped.
        - When the data format of `weight` is FRACTAL_NZ, the input shape is (n, ⌈k/group_size⌉), where **group_size** indicates the size of each group to which **k** is to be grouped.
  - **antiquantOffsetOptional** (aclTensor*, compute input): dequantization offset parameter, `antiquantOffset` in the dequantization formula, and aclTensor on the device. It is an optional parameter. When it is not required, pass a null pointer to it. When it is required, the shape must be the same as that of `antiquantScale`. The [data format](../../../docs/en/context/data_formats.md) can be ND.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The supported data types are FLOAT16, BFLOAT16, and INT32. When the data type is FLOAT16 or BFLOAT16, it must be the same as the data type of the input `x`. When the data type is INT32, the value range is [–128, 127], the data type of **x** can only be FLOAT16, the data type of **weight** can only be INT8, and the data type of `antiquantScale` can only be UINT64 or INT64. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported only in the transpose scenario.
    - <term>Atlas inference series products</term>: The supported data type is FLOAT16. The data type must be the same as that of the input `x`.
  - **quantScaleOptional** (aclTensor *, compute input): quantization parameter, aclTensor on the device, which is converted from the data of `quantScale` and `quantOffset` in the quantization formula through the `aclnnTransQuantParam` API.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The supported data type is UINT64, and the supported [data format](../../../docs/en/context/data_formats.md) is ND. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. It is an optional parameter. When it is not required, pass a null pointer to it. For different fake-quantization algorithm modes, the supported shapes are as follows:
      - pertensor [quantization mode](../../../docs/en/context/quantization_introduction.md): The input shape is (1,) or (1, 1).
      - perchannel [quantization mode](../../../docs/en/context/quantization_introduction.md): The input shape is (1, n) or (n,).
    - <term>Atlas inference series products</term>: This parameter is reserved and not used currently. It is fixed as a null pointer.
  - **quantOffsetOptional** (aclTensor*, compute input): quantization offset parameter, `quantOffset` in the quantization formula, and aclTensor on the device.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: The supported data type is FLOAT, and the supported [data format](../../../docs/en/context/data_formats.md) is ND. It is an optional parameter. When it is not required, pass a null pointer to it. If it is required, the shape must be the same as that of `quantScaleOptional`. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported.
    - <term>Atlas inference series products</term>: This parameter is reserved and not used currently. It is fixed as a null pointer.
  - **biasOptional** (aclTensor *, compute input): bias input, `bias` in the formula, and aclTensor on the device. It is an optional parameter. When it is not required, pass a null pointer to it. The [data format](../../../docs/en/context/data_formats.md) can be ND. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: One or two dimensions are supported, and the shape (n,) or (1, n) is supported. The data type can be FLOAT16 or FLOAT. When the data type of `x` is BFLOAT16, the data type of this parameter must be FLOAT. When the data type of `x` is FLOAT16, the data type of this parameter must be FLOAT16.
    - <term>Atlas inference series products</term>: The data type can be FLOAT16. One to six dimensions are supported. When **batch** is used, the input shape must be (batch, 1, n), where **batch** must be the same as the batch after the batch dimensions of **x** and **weight** are broadcast. When **batch** is not used, the input shape must be (n,) or (1, n).
  - **antiquantGroupSize** (int, compute input): groupSize input for dequantizing the input `weight` in pergroup or mx [quantization mode](../../../docs/en/context/quantization_introduction.md) of the fake quantization algorithm. It describes the size of the data to be dequantized corresponding to a group of dequantization parameters in the Reduce direction. If the fake quantization algorithm is not in pergroup or mx [quantization mode](../../../docs/en/context/quantization_introduction.md), pass **0**. If the fake quantization algorithm is pergroup [quantization mode](../../../docs/en/context/quantization_introduction.md), the value range is [32, k – 1] and the value must be a multiple of 32. In the mx [quantization mode](../../../docs/en/context/quantization_introduction.md), only 32 is supported.

  - **y** (aclTensor\*, compute output): compute output, `y` in the formula, aclTensor on the device. The [data format](../../../docs/en/context/data_formats.md) can be ND. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: Two dimensions are supported, and shape (m, n) is supported. The data type can be FLOAT16, BFLOAT16, or INT8. If `quantScaleOptional` exists, the data type is INT8. If `quantScaleOptional` does not exist, the data type can be FLOAT16 or BFLOAT16, and must be the same as the data type of the input `x`.
    - <term>Atlas inference series products</term>: The data type can be FLOAT16. Two to six dimensions are supported, and the shape can be (batch, m, n), where **batch** is optional. The batch dimensions of **x** and **weight** can be broadcast. The output **batch** is the same as the broadcast **batch**. **m** and **n** are the same as **m** of **x** and **n** of **weight**, respectively.
  - **workspaceSize** (uint64_t *, output): size of the workspace required to be allocated on the device.

  - **executor** (aclOpExecutor **, output): operator executor, containing the operator computation process.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

 ```
The first-phase API implements input parameter verification. The following errors may be thrown:
161001 (ACLNN_ERR_PARAM_NULLPTR): The passed parameter is a mandatory input, output, or attribute, but is a null pointer.
161002 (ACLNN_ERR_PARAM_INVALID):
    - The shape dimension of passed x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional, or y does not meet requirements.
    - The data type of passed x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional, or y is not supported.
    - The reduce dimensions (k) of x and weight are different.
    - When antiquantOffsetOptional exists, the shape is different from that of antiquantScale.
    - When quantOffsetOptional exists, the shape is different from that of quantScale.
    - The shape of x does not meet requirements.
    - The shape of biasOptional does not meet requirements.
    - The value of antiquantGroupSize does not meet requirements.
    - When quantOffsetOptional exists, quantScaleOptional is a null pointer.
    - The value of m, n, or k is not supported.
    - The tensor is empty, which is not supported.
    - The data format of the input tensor is not supported.
    - The continuity of the passed x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional and y does not meet the requirements.
  361001 (ACLNN_ERR_RUNTIME_ERROR): The product model is not supported.
 ```
## aclnnWeightQuantBatchMatmulV2

- **Parameters**

  - **workspace** (void *, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API `aclnnWeightQuantBatchMatmulV2GetWorkspaceSize`.
  - **executor** (aclOpExecutor *, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).


## Constraints
- Deterministic description:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnWeightQuantBatchMatmulV2** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.

Performance optimization suggestions:
- <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
  - pertensor [quantization mode](../../../docs/en/context/quantization_introduction.md): When the [data format](../../../docs/en/context/data_formats.md) is ND, the transposed `weight` input is recommended. When the [data format](../../../docs/en/context/data_formats.md) is FRACTAL_NZ, the non-transposed `weight` input is recommended.
  - pergroup [quantization mode](../../../docs/en/context/quantization_introduction.md): The non-transposed weight input is recommended.
  - perchannel [quantization mode](../../../docs/en/context/quantization_introduction.md): When the [data format](../../../docs/en/context/data_formats.md) is ND, the transposed `weight` input is recommended. When the [data format](../../../docs/en/context/data_formats.md) is FRACTAL_NZ, the non-transposed `weight` input is recommended. If the value range of m is [65, 96], **antiquantScale** of the UINT64 or INT64 data type is recommended.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  // Call the first-phase API of aclnnWeightQuantBatchMatmulV2.
  ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weight, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnWeightQuantBatchMatmulV2.
  ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

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
