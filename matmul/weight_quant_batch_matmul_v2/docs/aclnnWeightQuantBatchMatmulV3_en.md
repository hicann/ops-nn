# aclnnWeightQuantBatchMatmulV3

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                            |     √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- **Description**: Performs input matrix multiplication in a fake-quantization scenario and implements output quantization. Compared with the **aclnnWeightQuantBatchMatmulV2** API, this API has the following changes:

  The **innerPrecise** parameter is added to support the selection of high-precision or high-performance compute mode. In the A16W4 per_group scenario, this parameter can be set to 1 when batchSize <= 16 to improve performance.
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
  aclrtStream    stream)
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
        <td>Input `weight` in the formula</td>
        <td>-</td>
        <td>INT8, INT4, INT32, FLOAT8_E4M3FN<sup>2</sup>, HIFLOAT8<sup>2</sup>, FLOAT<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
        <td>ND, FRACTAL_NZ</td>
        <td>(k, n) is supported.</td>
        <td>Non-contiguous tensors are supported only in the transpose scenario.</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>Input</td>
        <td>Dequantization scale parameter and the input `antiquantScale` in the dequantization formula.</td>
        <td>When the data type is FLOAT16 or BFLOAT16, the data type must be the same as that of the input `x`. When the data type is UINT64 or INT64, x supports only FLOAT16 and is not transposed, weight supports only int8 and is transposed in ND mode, the mode supports only per_channel, quantScaleOptional and quantOffsetOptional must be empty, m supports only [1, 96], k and n must be 64-aligned. The aclnnCast API must be used to convert FLOAT16 to FLOAT32, and then the aclnnTransQuantParamV2 API must be used to convert FLOAT32 to UINT64. <a href="../../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2_en.md" target="_blank">See details</a></td>.
        <td>FLOAT16, BFLOAT16, UINT64<sup>1</sup>, INT64<sup>1</sup>, FLOAT8_E8M0<sup>2</sup></td>
        <td>ND</td>
        <td>
          <ul>
            <li>per_tensor mode: The input shape is (1,) or (1, 1).</li>
            <li>per_channel mode: The input shape is (1, n) or (n,).</li>
            <li>per_group mode: The input shape is (ceil(k, group_size), n)</li>.
          </ul>
        </td>
        <td>Non-contiguous tensors are supported only in the transpose scenario.</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional</td>
        <td>Input</td>
        <td>Dequantization offset parameter and the input `antiquantOffset` in the dequantization formula.</td>
        <td>When the data type is FLOAT16 or BFLOAT16, the data type must be the same as that of the input `x`. When the data type is INT32, the data range is limited to [–128, 127]. x supports only FLOAT16, weight supports only int8, and antiquantScale supports only UINT64 and INT64. It is an optional parameter. When it is not required, pass a null pointer to it.</td>
        <td>FLOAT16, BFLOAT16, INT32<sup>1</sup></td>
        <td>ND</td>
        <td>Must be the same as that of `antiquantScale`.</td>
        <td>Non-contiguous tensors are supported only in the transpose scenario.</td>
      </tr>
      <tr>
        <td>quantScaleOptional</td>
        <td>Input</td>
        <td>Quantization parameter, which is converted from the data of quantScale and quantOffset in the quantization formula through the `aclnnTransQuantParam` API.</td>
        <td>Converted from the data of `quantScale` and `quantOffset` in the quantization formula through the `aclnnTransQuantParam` API.</td>
        <td>UINT64<sup>1</sup></td>
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
        <td>FLOAT<sup>1</sup></td>
        <td>ND</td>
        <td>Same as `quantScaleOptional`.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>Bias input, `bias` in the formula. When the data type of `x` is BFLOAT16, the data type of this parameter must be FLOAT. When the data type of `x` is FLOAT16, the data type of this parameter must be FLOAT16.</td>
        <td>It is an optional parameter. When it is not required, pass a null pointer to it.</td>
        <td>FLOAT, FLOAT16, BFLOAT16<sup>2</sup></td>
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
        <td>INT8<sup>1</sup>, FLOAT16, BFLOAT16</td>
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

  - Ascend 950PR/Ascend 950DT:

    - The superscript "1" in the data type column of the preceding table indicates that the data type is not supported by this series.

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    - The superscript "2" in the "Data Type" column of the table above indicates data types that are not supported by the products.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
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
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
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

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<a id="a2_a3_Series"></a>

<details>
<summary><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></summary>

  - **Deterministic description**: The non-deterministic implementation is used by default. You can enable deterministic implementation by calling aclrtCtxSetSysParamOpt.

  - **Common constraints**
    - When the weight data format is FRACTAL_NZ and the data type is INT4 or INT32, or when the weight data format is ND and the data type is INT32, this parameter is supported only in the INT4Pack scenario. The aclnnConvertWeightToINT4Pack API needs to be used for INT32-to-INT4Pack conversion and ND-to-FRACTAL_NZ conversion. <a href="../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack_en.md" target="_blank">See details</a>. If the data type is INT4, the inner axis of weight must be an even number.
    - For different fake-quantization algorithm modes, the weight data format FRACTAL_NZ is supported only in the following scenarios:
      - per_channel mode:
          The weight data type is INT8, and the y data type is not INT8.
          The weight data type is INT4 or INT32, weight is transposed, and the y data type is not INT8.
      - per_group mode: The weight data type is INT4/INT32, weight and x are not transposed, antiquantGroupSize is 64 or 128, k is antiquantGroupSize aligned, n is 64-aligned, and the y data type is not INT8.

  - **Performance Optimization Suggestions**
    - per_channel mode: To improve performance, you are advised to use the **weight** input after transpose. If the value range of m is [65, 96], **antiquantScale** of the UINT64/INT64 data type is recommended.
    - per_group mode: In the A16W4 scenario where **batchSize** is less than or equal to 16, you can set **innerPrecise** to **1** and set the **weight** data format to FRACTAL_NZ to improve performance, but the accuracy may drop.

</details>

<a id="ascend_950pr_ascend950dt"></a>

<details>
<summary>Ascend 950PR/Ascend 950DT</summary>

  - **Deterministic description**: The default deterministic implementation is used.

   <a id="common-constraints"></a>

  - **Common constraints**
    - The sizes of m, k, and n in the `x` and `weight` matrices are in the range of [1, 2^31 – 1]. The Reduce dimension k of `weight` must be the same as that of `x`.
    - The following quantization modes are supported: pertensor [quantization mode](../../../docs/en/context/quant_more_introduction.md), perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md), pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md), and mx [quantization mode](../../../docs/en/context/quant_more_introduction.md).
    - `x` does not support transposition. Therefore, non-continuous tensors (../../../docs/en/context/non_contiguous_tensor.md) are not supported. weight supports discontinuous tensors only in the transposition scenario. The discontinuous tensors of antiquantScale and antiquantOffsetOptional support only the transposition scenario, and the continuity requirements must be the same as those of weight.
    - The shapes supported by different quantization modes of `antiquantScale` are as follows:
      - pertensor [quantization mode](../../../docs/en/context/quant_more_introduction.md): (1,) or (1, 1).
      - perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (1, n) or (n,).
      - pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (⌈k/group_size⌉, n), where **group_size** indicates the size of each group to which **k** is to be grouped.
      - mx [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (⌈k/group_size⌉, n), where group_size indicates the size of each group to which k is to be grouped. Only 32 is supported.
    - `quantScaleOptional` and `quantOffsetOptional` are reserved parameters and are not used currently. Null pointers are always passed.

    <a id="constraints-for-a16w8-scenario"></a>

    <details>
    <summary>Constraints for the A16W8 scenario</summary>

    - **Input and output data type combinations**

    | x        | weight            | weight Format | antiquantScale | antiquantOffsetOptional | quantScaleOptional | quantOffsetOptional | biasOptional | antiquantGroupSize | y    | Scenario Description|
    | ----     | ------------------| --------------| -------------- | ------------------------| ------------------ | ------------------- | ------------ | ------------------ | ---- | ------- |
    | FLOAT16/BFLOAT16 | INT8 | ND | Consistent with x| Consistent with x or null| null | null | Consistent with x or FLOAT (only when x is BFLOAT16) or null| PerGroupQuantization: [32, k-1] and a multiple of 32<br>Others: 0| Consistent with x| T & C & G quantization|
    | FLOAT16/BFLOAT16 | HIFLOAT8/FLOAT8_E4M3FN | ND | Same as x| null | null | null | Same as x or null| pergroup: [32, k-1], and a multiple of 32<br>Others: 0| Same as x| C quantization|

    </details>

    <a id="constraints-for-a16w4-scenario"></a>

    <details>
    <summary>Constraints for the A16W4 scenario</summary>

    - **Input and Output Data Type Combinations**

    | x        | weight            | weight Format | antiquantScale | antiquantOffsetOptional | quantScaleOptional | quantOffsetOptional | biasOptional | antiquantGroupSize | y    | Scenario Description|
    | ----     | ------------------| --------------| -------------- | ------------------------| ------------------ | ------------------- | ------------ | ------------------ | ---- | ------- |
    | FLOAT16/BFLOAT16 | INT4/INT32 | ND | Same as x| Same as x or null| null | null | Same as x, FLOAT (only when x is BFLOAT16), or null| 0 | Same as x| T quantization|
    | FLOAT16/BFLOAT16 | INT4/INT32 | ND | Same as x| Same as x or null| null | null | Same as x, FLOAT (only when x is BFLOAT16), or null| pergroup: [32, k-1], and a multiple of 32<br>Others: 0| Same as x| C & G quantization|
    | FLOAT16/BFLOAT16 | FLOAT4_E2M1 | ND | FLOAT8_E8M0 | null | null | null | Same as x or null| 32 | Same as x| MX quantization|
    | FLOAT16/BFLOAT16 | FLOAT | ND | FLOAT8_E8M0 | null | null | null | Same as x, FLOAT (only when x is BFLOAT16), or null| 32 | Same as x| MX quantization|

    - **Constraints**

      In addition to the common restrictions, the restrictions in the A16W4 scenario are as follows:
      - If the `weight` data type is INT4 or FLOAT4_E2M1, the last dimension of the weight must be 2-aligned. If the `weight` data type is INT32 or FLOAT, the last dimension of the weight must be 8-aligned.
      - If the `weight` data type is INT32 or FLOAT, the `aclnnConvertWeightToINT4Pack` API must be used to convert the data from INT32 or FLOAT to tightly packed INT4 or FLOAT4_E2M1. [For details, see the sample](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack_en.md).
  
  <a id="ascend_950pr_ascend950dt_Optimization Suggestions"></a>

  - **Performance Optimization Suggestions**

    - pertensor [quantization mode](../../../docs/en/context/quant_more_introduction.md): When  [Data Format](../../../docs/en/context/data_format.md) is ND, the transposed `weight` input is recommended.
    - perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md): When the  [Data Format](../../../docs/en/context/data_format.md) is ND, the transposed `weight` input is recommended.
    - pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md) and mx [quantization mode](../../../docs/en/context/quant_more_introduction.md): The non-transposed `weight` input is recommended.

    </details>

</details>

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>and Ascend 950PR/Ascend 950DT:
A16W8 calling example.

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

- Ascend 950PR/Ascend 950DT:
A16MxFp4 calling example.

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v3.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  template <typename T1>
  inline T1 CeilDiv(T1 a, T1 b)
  {
      return b == 0 ? a : (a + b - 1) / b;
  };
  template <typename T1>
  inline T1 CeilAlign(T1 a, T1 b)
  {
      return (a + b - 1) / b * b;
  };

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(
      const aclTensor* weight, const aclTensor* weightInt4Pack, uint64_t* workspaceSize, aclOpExecutor** executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(
      void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream)
  {
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
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
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
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorB4(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor, aclFormat format)
  {
      auto size = hostData.size() * sizeof(T);
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
      if (format == aclFormat::ACL_FORMAT_ND) {
          *tensor = aclCreateTensor(
              shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
              shape.size(), *deviceAddr);
      } else {
          std::vector<int64_t> nzShape;
          if (dataType == aclDataType::ACL_INT4 || dataType == aclDataType::ACL_FLOAT4_E2M1) {
              nzShape = {CeilDiv(shape[1], (int64_t)16), CeilDiv(shape[0], (int64_t)16), 16, 16};
          } else {
              nzShape = {CeilDiv(shape[1], (int64_t)2), CeilDiv(shape[0], (int64_t)16), 16, 2};
          }
          *tensor = aclCreateTensor(
              shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(),
              nzShape.size(), *deviceAddr);
      }

      return 0;
  }

  void PrintMat(std::vector<float> resultData, std::vector<int64_t> resultShape)
  {
      int64_t m = resultShape[0];
      int64_t n = resultShape[1];
      for (size_t i = 0; i < m; i++) {
          printf(i == 0 ? "[[" : " [");
          for (size_t j = 0; j < n; j++) {
              printf(j == n - 1 ? "%.1f" : "%.1f, ", resultData[i * n + j]);
              if (j == 2 && j + 3 < n) {
                  printf("..., ");
                  j = n - 4;
              }
          }
          printf(i < m - 1 ? "],\n" : "]]\n");
          if (i == 2 && i + 3 < m) {
              printf(" ... \n");
              i = m - 4;
          }
      }
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnWeightQuantBatchMatmulV3Test(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      aclDataType weightPackedDtype = aclDataType::ACL_FLOAT; // Optional: ACL_FLOAT type
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      int64_t m = 16;
      int64_t k = 64;
      int64_t n = 64;
      int64_t groupSize = 32;
      int64_t weightDim0 = k;
      int64_t weightDim1 = n;
      bool isWeightTransposed = false;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {k, n};
      std::vector<int64_t> antiquantScaleShape = {k / groupSize, n};
      std::vector<int64_t> yShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* weightB4PackDeviceAddr = nullptr;
      void* antiquantScaleDeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* y = nullptr;
      aclTensor* antiquantScale = nullptr;
      std::vector<int64_t> weightPackedShape;
      weightPackedShape = {weightDim0, weightDim1 / 8};
      std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0b0011110000000000); // fp16 1.0
      xHostData[0] = 0; // fp16 0 to check whether the verification result meets the requirements
      std::vector<float> weightHostData(GetShapeSize(weightShape), 1.0); // fp32 1.0, which is converted to fp4_e2m1 1.0 after int4pack
      std::vector<float> yHostData(GetShapeSize(yShape), 0);

      std::vector<uint8_t> antiquantScaleHostData(GetShapeSize(antiquantScaleShape), 0b01111111); // fp8_e8m0 1.0

      // Create an x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an other aclTensor.
      ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a y aclTensor.
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an antiquantScale aclTensor.
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a yFp16 aclTensor.
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      aclFormat weightFormat = aclFormat::ACL_FORMAT_ND; // Optional: ACL_FORMAT_FRACTAL_NZ
      aclTensor* weightPacked = nullptr;

      std::vector<int8_t> weightB4PackHostData(n * k / 2, 0); // One B8 data stores two B4 data, so the value is divided by 2.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
          weightB4PackHostData.resize(CeilAlign(weightDim1 / 2, (int64_t)8) * CeilAlign(weightDim0, (int64_t)16), 0);
      }
      // Create a weightInt4Pack aclTensor.
      ret = CreateAclTensorB4(
          weightB4PackHostData, weightPackedShape, &weightB4PackDeviceAddr, weightPackedDtype, &weightPacked,
          weightFormat);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPackedTensorPtr(weightPacked, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightPackedDeviceAddrPtr(weightB4PackDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Convert weight from INT32 to INT4 pack format.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightPacked, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      void* workspacePackAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspacePackAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspacePackAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspacePackAddrPtr.reset(workspacePackAddr);
      }
      ret = aclnnConvertWeightToINT4Pack(workspacePackAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnWeightQuantBatchMatmulV3.
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
          x, weightPacked, antiquantScale, nullptr, nullptr, nullptr, nullptr, groupSize, 0, yFp16, &workspaceSize,
          &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnWeightQuantBatchMatmulV3.
      ret = aclnnWeightQuantBatchMatmulV3(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3 failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Convert the output into FP32.
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceCastAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceCastAddrPtr.reset(workspaceCastAddr);
      }
      ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      PrintMat(resultData, yShape);
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnWeightQuantBatchMatmulV3Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3Test failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
  