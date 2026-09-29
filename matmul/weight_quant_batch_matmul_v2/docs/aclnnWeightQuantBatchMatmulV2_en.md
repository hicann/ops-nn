# aclnnWeightQuantBatchMatmulV2

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/weight_quant_batch_matmul_v2)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                            |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>|      ×     |
| <term>Atlas inference products</term>|      √     |
| <term>Atlas training products</term>|      ×     |

## Function

- **Function**: performs matrix multiplication in the fake-quantization scenario and quantizes the output.
- **Formula**:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  In the formula, $weight$ is the input of the fake-quantization scenario, and the dequantization formula $ANTIQUANT(weight)$ is as follows:

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  - If the output does not need to be quantized, the calculation formula is as follows:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  - If the output needs to be quantized, the quantization formula is as follows:

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnWeightQuantBatchMatmulV2GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnWeightQuantBatchMatmulV2** is called to perform computation.

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *antiquantScale,
  const aclTensor *antiquantOffsetOptional,
  const aclTensor *quantScaleOptional,
  const aclTensor *quantOffsetOptional,
  const aclTensor *biasOptional,
  int              antiquantGroupSize,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnWeightQuantBatchMatmulV2GetWorkspaceSize

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
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th style="white-space: nowrap">Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Left input matrix of matrix multiplication, which is the input `x` in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2D, supported shape: (m, k)</td>
        <td>Supported only in the transpose scenario</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Right input matrix of matrix multiplication, which is the input `weight` in the formula.</td>
        <td>-</td>
        <td>INT8, INT4, FLOAT8_E4M3FN<sup>2</sup>, HIFLOAT8<sup>2</sup>, INT32, FLOAT<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
        <td>ND, FRACTAL_NZ</td>
        <td>2D, supporting (k, n)</td>
        <td>Supported only in transpose scenarios</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>Input</td>
        <td>Scale parameter for input dequantization calculation, which is used to implement the input dequantization calculation. It is the input `antiquantScale` in the dequantization formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, FLOAT8_E8M0<sup>2</sup>, UINT64<sup>1</sup>, INT64<sup>1</sup></td>
        <td>ND</td>
        <td>1D or 2D</td>
        <td>Supported only in transpose scenarios</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional</td>
        <td>Optional input</td>
        <td>Offset parameter for input dequantization calculation, which is used to implement the input dequantization calculation. It is the `antiquantOffset` in the dequantization formula.</td>
        <td>A null pointer if not required.</td>
        <td>FLOAT16, BFLOAT16, INT32<sup>1</sup></td>
        <td>ND</td>
        <td>The value must be the same as that of `antiquantScale`.</td>
        <td>Supported only in the transpose scenario.</td>
      </tr>
      <tr>
        <td>quantScaleOptional</td>
        <td>Optional input</td>
        <td>Quantization parameters for implementing output quantization calculation.</td>
        <td>The value is obtained by converting the data of `quantScale` and `quantOffset` in the quantization formula using the `aclnnTransQuantParam` API. If not required, it is a null pointer.</td>
        <td>UINT64<sup>1</sup></td>
        <td>ND</td>
        <td>1D or 2D</td>
        <td>Not supported.</td>
      </tr>
      <tr>
        <td>quantOffsetOptional</td>
        <td>Optional input</td>
        <td>Quantization offset parameter for implementing output quantization calculation, which is `quantOffset` in the quantization formula.</td>
        <td>If not required, it is a null pointer.</td>
        <td>FLOAT<sup>1</sup></td>
        <td>ND</td>
        <td>The value must be the same as that of `quantScaleOptional`.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Optional input</td>
        <td>Bias input, which is `bias` in the formula.</td>
        <td>Null pointer when it is not required.</td>
        <td>FLOAT16, FLOAT, BFLOAT16<sup>2</sup></td>
        <td>ND</td>
        <td>1-2 dimensions</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantGroupSize</td>
        <td>Input</td>
        <td>Group size for dequantizing the input `weight` in fake-quantization pergroup and mx <a href="../../../docs/en/context/quant_more_introduction.md" target="_blank">quantization modes</a>. It describes the size of the data block to be dequantized along the reduce axis corresponding to a single set of dequantization parameters.</td>
        <td>If the fake-quantization algorithm is neither pergroup nor mx <a href="../../../docs/en/context/quant_more_introduction.md" target="_blank">quantization mode</a>, pass 0.<br>If the fake-quantization algorithm is pergroup <a href="../../../docs/en/context/quant_more_introduction.md" target="_blank">quantization mode</a>, the value range is [32, k – 1] and the value must be a multiple of 32.<br>In mx <a href="../../../docs/en/context/quant_more_introduction.md" target="_blank">quantization mode</a>, only 32 is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>The `y` in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8<sup>1</sup></td>
        <td>ND</td>
        <td>2</td>
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
    </tbody>
  </table>

  - Ascend 950PR/Ascend 950DT:

    - The superscript "1" in the data type column of the preceding table indicates that the data type is not supported by this series.

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    - The superscript "2" in the "Data Type" column of the table above indicates data types that are not supported by the products.

  - <term>Atlas inference products</term>

    - The superscript "3" in the data type column of the preceding table indicates that the data type is not supported by this series.

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
        <td rowspan="13">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="13">161002</td>
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
        <td>When x is of type bfloat16 and weight is of type float4_e2m1 or float32, the bias data type can only be bfloat16.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>The product model is not supported.</td>
      </tr>
    </tbody>
  </table>

## aclnnWeightQuantBatchMatmulV2

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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnWeightQuantBatchMatmulV2GetWorkspaceSize API.</td>
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

<a id="a2_a3_series_product_"></a>

<details>
<summary><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></summary>

  - **Deterministic description**: The default implementation is non-deterministic. You can enable deterministic implementation by calling aclrtCtxSetSysParamOpt.

  - `x` (aclTensor *, input for computation): When the matrix is not transposed, the value of m is within the range of [1, 2^31 – 1]. When the matrix is transposed, the value of m is within the range of [1, 65535].
  - `weight` (aclTensor *, input for computation): The dimension can be 2D. The dimension k of Reduce must be the same as that of `x`. The data type can be INT8, INT4, or INT32. When the `weight`  [Data Format](../../../docs/en/context/data_format.md) is FRACTAL_NZ and the data type is INT4 or INT32, or when the `weight`  [Data Format](../../../docs/en/context/data_format.md) is ND and the data type is INT32, this parameter is supported only in the INT4Pack scenario. `aclnnConvertWeightToINT4Pack` is also used for INT32-to-INT4Pack conversion and ND-to-FRACTAL_NZ conversion. For details, see the [example](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack_en.md). If the data type is INT4, the inner axis of `weight` must be an even number.[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported only in the transpose scenario. The shape can be (k, n), where **k** indicates the size of the first dimension of the matrix, and **n** indicates the size of the second dimension of the matrix.
    For different fake-quantization algorithm modes, the `weight`  [Data Format](../../../docs/en/context/data_format.md) FRACTAL_NZ is supported only in the following scenarios:
    - perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md):
      - The `weight` data type is INT8, and the y data type is not INT8.
      - The `weight` data type is INT4 or INT32, `weight` is transposed, and the y data type is not INT8.
    - pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md): The `weight` data type is INT4 or INT32, `weight` and `x` are not transposed, antiquantGroupSize is 64 or 128, k is a multiple of antiquantGroupSize, n is a multiple of 64, and the y data type is not INT8.
  - antiquantScale (aclTensor *, compute input): The supported data types are FLOAT16, BFLOAT16, UINT64, and INT64. When the data type is FLOAT16 or BFLOAT16, the data type must be the same as that of the input x. When the data type is UINT64 or INT64, x supports only FLOAT16 and is not transposed, weight supports only INT8 and is transposed in ND format, and the quantization mode must be perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md). Null pointers must be passed for **quantScaleOptional** and **quantOffsetOptional**. The value of **m** ranges from 1 to 96. The values of **k** and **n** must be multiples of 64. First, the **aclnnCast** API is used to perform the FLOAT16-to-FLOAT32 conversion. For details, see [Cast](https://gitcode.com/cann/ops-math/blob/master/math/cast/docs/aclnnCast.md). Then, the **aclnnTransQuantParamV2** API is used to perform the FLOAT32-to-UINT64 conversion. For details, see [TransQuantParamV2](../../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2_en.md).[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported only in the transpose scenario.
    For different fake-quantization algorithm modes, `antiquantScale` supports the following shapes:
    - pertensor [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (1,) or (1, 1).
    - perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (1, n) or (n,).
    - pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (⌈k/group_size⌉, n), where **group_size** indicates the size of each group to which **k** is to be grouped.
  - `antiquantOffsetOptional` (aclTensor *, input for computation): The data type can be FLOAT16, BFLOAT16, or INT32. When the data type is FLOAT16 or BFLOAT16, the data type must be the same as that of the input `x`. When the data type is INT32, the value range is restricted to [–128, 127]. x supports only FLOAT16, weight supports only INT8, and `antiquantScale` supports only UINT64 or INT64.[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are supported only in the transpose scenario.
  - `quantScaleOptional` (aclTensor *, input for computation): The data type can be UINT64, and the data format can be ND.[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported. It is an optional parameter. When it is not required, pass a null pointer to it. For different fake-quantization algorithm modes, the supported shapes are as follows:
    - pertensor [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (1,) or (1, 1).
    - perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (1, n) or (n,).
  - `quantOffsetOptional` (aclTensor *, input for computation): The data type can be FLOAT, and the data format can be ND. It is an optional parameter. When it is not required, pass a null pointer to it. If it is required, the shape must be the same as that of `quantScaleOptional`.[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
  - `biasOptional` (aclTensor *, input for computation): The dimension can be 1 or 2, and the shape can be (n,) or (1, n). The data type can be FLOAT16 or FLOAT. When the data type of `x` is BFLOAT16, the data type of this parameter must be FLOAT. When the data type of `x` is FLOAT16, the data type of this parameter must be FLOAT16.
  - **antiquantGroupSize** (int, compute input): groupSize input for dequantizing the input weight in pergroup or mx [quantization mode](../../../docs/en/context/quant_more_introduction.md) of the fake quantization algorithm. It describes the size of the data to be dequantized corresponding to a group of dequantization parameters in the Reduce direction. If the fake quantization algorithm is not in pergroup or mx [quantization mode](../../../docs/en/context/quant_more_introduction.md), pass **0**. If the fake quantization algorithm is pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md), the value range is [32, k – 1] and the value must be a multiple of 32. In the mx [quantization mode](../../../docs/en/context/quant_more_introduction.md), only 32 is supported.
  - `y` (aclTensor *, output): The dimension supports 2D, and the shape supports (m, n). The data type can be FLOAT16, BFLOAT16, or INT8. If `quantScaleOptional` exists, the data type is INT8. If `quantScaleOptional` does not exist, the data type can be FLOAT16 or BFLOAT16, and must be the same as the data type of the input `x`.

  - Performance optimization suggestions:
    - pertensor [quantization mode](../../../docs/en/context/quant_more_introduction.md): When the  [Data Format](../../../docs/en/context/data_format.md) is ND, the transposed `weight` input is recommended. When the  [Data Format](../../../docs/en/context/data_format.md) is FRACTAL_NZ, the non-transposed `weight` input is recommended.
    - pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md): The non-transposed weight input is recommended.
    - perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md): When the  [Data Format](../../../docs/en/context/data_format.md) is ND, the transposed `weight` input is recommended. When the  [Data Format](../../../docs/en/context/data_format.md) is FRACTAL_NZ, the non-transposed `weight` input is recommended. If the value range of m is [65, 96], **antiquantScale** of the UINT64 or INT64 data type is recommended.

</details>

<a id="atlas-inference-products"></a>

<details>
<summary><term>Atlas inference products</term></summary>

  - **Deterministic description**: The non-deterministic implementation is used by default. You can enable the deterministic implementation by calling aclrtCtxSetSysParamOpt.

  - `x` (aclTensor, input for computation): The data type is FLOAT16. The shape supports 2 to 6 dimensions. The input shape must be (batch, m, k), where batch indicates the batch size of the matrix and supports 0 to 4 dimensions, m indicates the size of the first dimension of the single-batch matrix, and k indicates the size of the second dimension of the single-batch matrix. The batch dimension must meet the broadcast relationship with the batch dimension of `weight`. When the fake-quantization algorithm is in per-tensor mode (../../../docs/en/context/quant_more_introduction.md), the value of *m x k* cannot exceed 512000000.
  - `weight` (aclTensor *, input for computation): The shape supports 2 to 6 dimensions. The batch dimension must meet the broadcast relationship with the batch dimension of `x`. The data type is INT8. Details are as follows:
    - If the  [Data Format](../../../docs/en/context/data_format.md) is ND, the input shape must be (batch, k, n), where **batch** indicates the batch size of the matrix and can be zero- to four-dimensional, **k** indicates the size of the first dimension of the single batch matrix, and **n** indicates the size of the second dimension of the single batch matrix.
    - If the  [Data Format](../../../docs/en/context/data_format.md) is FRACTAL_NZ:
      - The input shape must be (batch, n, k), where **batch** indicates the batch size of the matrix and can be zero- to four-dimensional, **k** indicates the size of the first dimension of the single batch matrix, and **n** indicates the size of the second dimension of the single batch matrix.
      - This API is used together with aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight to convert the input format from ND to FRACTAL_NZ.
  - `antiquantScale` (aclTensor *, input for computation): The data type is FLOAT16. The data type must be the same as that of the input `x`.
    For different fake-quantization algorithm modes, `antiquantScale` supports the following shapes:
    - pertensor [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (1,) or (1, 1).
    - perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (n, 1) or (n, ).[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is related to the data format of `weight` as follows:
      - When the data format of `weight` is ND, the input shape is (⌈k/group_size⌉, n), where **group_size** indicates the size of each group to which **k** is to be grouped.
      - When the data format of `weight` is FRACTAL_NZ, the input shape is (n, ⌈k/group_size⌉), where **group_size** indicates the size of each group to which **k** is to be grouped.
  - `antiquantOffsetOptional` (aclTensor *, input for computation): The data type is FLOAT16. The data type must be the same as that of the input `x`.
  - `quantScaleOptional` (aclTensor *, input for computation): reserved. Currently, this parameter is not used and a null pointer is always passed.
  - `quantOffsetOptional` (aclTensor *, input for computation): reserved. Currently, this parameter is not used and a null pointer is always passed.
  - `biasOptional` (aclTensor *, input for computation): The data type is FLOAT16. One to six dimensions are supported. When **batch** is used, the input shape must be (batch, 1, n), where **batch** must be the same as the batch after the batch dimensions of **x** and **weight** are broadcast. When **batch** is not used, the input shape must be (n,) or (1, n).
  - `antiquantGroupSize` (int, input for computation): The data type is FLOAT16. Two to six dimensions are supported, and the shape can be (batch, m, n), where **batch** is optional. The batch dimensions of **x** and **weight** can be broadcast. The output **batch** is the same as the broadcast **batch**. **m** and **n** are the same as **m** of **x** and **n** of **weight**, respectively.
  - `y` (aclTensor *, output for computation):

</details>

<a id="ascend_950pr_ascend950dt"></a>

<details>
<summary>Ascend 950PR/Ascend 950DT</summary>

  - **Deterministic description**: The default deterministic implementation is used.

  <a id="common-constraints"></a>

  - **Common constraints**
    - The sizes of the `x` and `weight` matrices m, k, and n are in the range of [1, 2^31 – 1]. The dimension k of `weight`Reduce must be the same as that of `x`.
    - The following quantization modes are supported: pertensor [quantization mode](../../../docs/en/context/quant_more_introduction.md), perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md), pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md), and mx [quantization mode](../../../docs/en/context/quant_more_introduction.md).
    - `x` does not support transposition. Therefore, [non-continuous tensors](../../../docs/en/context/non_contiguous_tensor.md) are not supported. The weight supports discontinuous tensors only in the transposition scenario. The anti-quantization scale and anti-quantization offset optional support discontinuous tensors only in the transposition scenario, and the continuity requirements must be the same as those of the weight.
    - The shapes supported by different quantization modes of `antiquantScale` are as follows:
      - pertensor [quantization mode](../../../docs/en/context/quant_more_introduction.md): (1,) or (1, 1).
      - perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (1, n) or (n,).
      - pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (⌈k/group_size⌉, n), where **group_size** indicates the size of each group to which **k** is to be grouped.
      - mx [quantization mode](../../../docs/en/context/quant_more_introduction.md): The input shape is (⌈k/group_size ⌉, n), where group_size indicates the size of each group to be grouped. The value can only be 32.
    - `quantScaleOptional` and `quantOffsetOptional` are reserved parameters and are not used currently. Empty pointers are always passed.

    <a id="constraints-for-a16w8-scenario"></a>
    <details>
    <summary>Constraints for the A16W8 scenario</summary>

    - **Input and Output Data Type Combinations**

    | x        | weight            | weight Format | antiquantScale | antiquantOffsetOptional | quantScaleOptional | quantOffsetOptional | biasOptional | antiquantGroupSize | y    | Scenario Description|
    | ----     | ------------------| --------------| -------------- | ------------------------| ------------------ | ------------------- | ------------ | ------------------ | ---- | ------- |
    | FLOAT16/BFLOAT16 | INT8 | ND | Same as x| Same as x or null| null | null | Same as x or FLOAT (only when x is BFLOAT16) or null| pergroup: [32, k-1], and a multiple of 32<br>Others: 0| Same as x| T & C & G quantization|
    | FLOAT16/BFLOAT16 | HIFLOAT8/FLOAT8_E4M3FN | ND | Same as x| null | null | null | Same as x or null| pergroup: [32, k-1], and a multiple of 32<br>Others: 0| Same as x| C quantization|

    </details>

    <a id="constraints-for-a16w4-scenario"></a>
    <details>
    <summary>Constraints for the A16W4 scenario</summary>

    - **Input and Output Data Type Combinations**

    | x        | weight            | weight Format | antiquantScale | antiquantOffsetOptional | quantScaleOptional | quantOffsetOptional | biasOptional | antiquantGroupSize | y    | Scenario Description|
    | ----     | ------------------| --------------| -------------- | ------------------------| ------------------ | ------------------- | ------------ | ------------------ | ---- | ------- |
    | FLOAT16/BFLOAT16 | INT4/INT32 | ND | Same as x| Same as x or null| null | null | Same as x or FLOAT (only when x is BFLOAT16) or null| 0 | Same as x| T quantization|
    | FLOAT16/BFLOAT16 | INT4/INT32 | ND/FRACTAL_NZ | Same as x| Same as x or null| null | null | Same as x or FLOAT (only when x is BFLOAT16) or null| pergroup: [32, k-1], and a multiple of 32<br>Others: 0| Same as x| C & G quantization|
    | FLOAT16/BFLOAT16 | FLOAT4_E2M1 | FRACTAL_NZ | Consistent with x| null | null | null | Consistent with x or null| [32, k-1], and a multiple of 32| Consistent with x| G quantization|
    | FLOAT16/BFLOAT16 | FLOAT | FRACTAL_NZ | Consistent with x| null | null | null | Consistent with x or null| [32, k-1], and a multiple of 32| Consistent with x| G quantization|
    | FLOAT16/BFLOAT16 | FLOAT4_E2M1 | ND/FRACTAL_NZ | FLOAT8_E8M0 | null | null | null | Consistent with x or null| 32 | Consistent with x| MX quantization|
    | FLOAT16/BFLOAT16 | FLOAT | ND/FRACTAL_NZ | FLOAT8_E8M0 | null | null | null | Consistent with x, FLOAT (only when x is BFLOAT16), or null| 32 | Consistent with x| MX quantization|

    - **Constraints**

      In addition to the common restrictions, the restrictions for the A16W4 scenario are as follows:
      - If the `weight` data type is INT4 or FLOAT4_E2M1, the last dimension of the weight must be 2-aligned. If the `weight` data type is INT32 or FLOAT, the last dimension of the weight must be 8-aligned.
      - If the `weight` data type is INT32 or FLOAT, the `aclnnConvertWeightToINT4Pack` API must be used to convert the data from INT32 or FLOAT to tightly packed INT4 or FLOAT4_E2M1. [For details, see the sample](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack_en.md).
      - The weight  [Data Format](../../../docs/en/context/data_format.md) FRACTAL_NZ is supported only in the following scenarios:
        - perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md): The `weight` data type is INT4 or INT32, `weight` is not transposed, and `x` is not transposed.
        - pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md): The `weight` data type is INT4, INT32, FLOAT4_E2M1, or FLOAT, `weight` is not transposed, and `x` is not transposed.
        - mx [quantization mode](../../../docs/en/context/quant_more_introduction.md): The `weight` data type is FLOAT4_E2M1 or FLOAT, `weight` is not transposed, and `x` is not transposed.

  <a id="ascend_950pr_ascend950dt_Optimization Suggestions"></a>

  - **Performance Optimization Suggestions**

    - pertensor [quantization mode](../../../docs/en/context/quant_more_introduction.md): When the  [Data Format](../../../docs/en/context/data_format.md) is ND, the transposed `weight` input is recommended. When the  [Data Format](../../../docs/en/context/data_format.md) is FRACTAL_NZ, the non-transposed `weight` input is recommended.
    - perchannel [quantization mode](../../../docs/en/context/quant_more_introduction.md): When the  [Data Format](../../../docs/en/context/data_format.md) is ND, the transposed `weight` input is recommended. When the  [Data Format](../../../docs/en/context/data_format.md) is FRACTAL_NZ, the non-transposed `weight` input is recommended.
    - pergroup [quantization mode](../../../docs/en/context/quant_more_introduction.md) and mx [quantization mode](../../../docs/en/context/quant_more_introduction.md): The non-transposed `weight` input is recommended.

    </details>

</details>

## Example

  The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- A16W8 calling example:

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
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

  int AclnnWeightQuantBatchMatmulV2Test(int32_t deviceId, aclrtStream stream)
  {
      int64_t m = 16;
      int64_t k = 32;
      int64_t n = 16;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {k, n};
      std::vector<int64_t> antiquantScaleShape = {n};
      std::vector<int64_t> yShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* antiquantScaleDeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* antiquantScale = nullptr;
      aclTensor* y = nullptr;
      // Fill the FP16 1.0 and BF16 1.0 with 0b0011111110000000.
      std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0b0011110000000000); // fp16 1.0
      std::vector<int8_t> weightHostData(GetShapeSize(weightShape), 1);
      std::vector<uint16_t> antiquantScaleHostData(GetShapeSize(antiquantScaleShape), 0b0011110000000000);
      std::vector<float> yHostData(GetShapeSize(yShape), 0);

      // Create x aclTensor of the ACL_FLOAT16 or ACL_BFLOAT16 type.
      auto ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an other aclTensor of the optional ACL_INT8/ACL_FLOAT8_E8M0/ACL_HIFLOAT8 type.
      ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an aclTensor y to convert the output back to FP32.
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an antiquantScale aclTensor.
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT16,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the yFp16 aclTensor, which is the actual output of the computation. The type is the same as that of x.
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16deviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // Call the first-phase API of aclnnWeightQuantBatchMatmulV2.
      ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
          x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnWeightQuantBatchMatmulV2.
      ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

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
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
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
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      ret = AclnnWeightQuantBatchMatmulV2Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulV2Test failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- The following is an example of invoking the A16W4 interface. The `aclnnConvertWeightToINT4Pack` interface needs to be invoked to assist in the invocation.

  ``` cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_npu_format_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  int GetSize(std::vector<int64_t>& shape)
  {
      int64_t size = 1;
      for (auto i : shape) {
          size *= i;
      }
      return size;
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

  int AclnnWeightQuantBatchMatmulV2Test(int32_t deviceId, aclrtStream& stream)
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
      // Fill the FP16 1.0 and the BF16 1.0 as 0b0011111110000000.
      std::vector<uint16_t> xHostData(GetSize(xShape), 0b0011110000000000);
      std::vector<float> weightHostData(GetSize(weightShape), 1.0); // fp32 1.0, which is converted to fp4_e2m1 1.0 after int4pack
      std::vector<float> yHostData(GetSize(yShape), 0);

      std::vector<uint16_t> antiquantScaleHostData(GetSize(antiquantScaleShape), 0b0011110000000000); // fp16 1.0

      // Create an ACLTensor of type ACL_FLOAT16 or ACL_BFLOAT16.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an other ACL tensor of the FLOAT4_E2M1 type. The pack type is FLOAT.
      ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an aclTensor y to convert the output back to FP32.
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an antiquantScale aclTensor.
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT16,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the yFp16 aclTensor, which is the actual output of the computation. The type is the same as that of x.
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ; // Optional: ACL_FORMAT_ND.
      aclTensor* weightPacked = nullptr;

      std::vector<int8_t> weightB4PackHostData(n * k / 2, 0); // One B8 data stores two B4 data. Therefore, the value is divided by 2.
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

      // Call the first-phase API of aclnnWeightQuantBatchMatmulV2.
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
          x, weightPacked, antiquantScale, nullptr, nullptr, nullptr, nullptr, groupSize, yFp16, &workspaceSize,
          &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnWeightQuantBatchMatmulV2.
      ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

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
      auto ret = AclnnWeightQuantBatchMatmulV2Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulV2Test failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
  