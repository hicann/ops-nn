# aclnnQuantMatmulWeightNz

## Supported Products

| Product                                                        |  Supported|
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                      |    ✓    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>      |    ✓    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>      |    ✓    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                              |    ✓    |
| <term>Atlas training products</term>                              |    ×    |

## Function

- Description: Performs matrix multiplication for quantization. Similar APIs include **aclnnMm** (only two-dimensional tensors can be used as the input of matrix multiplication) and **aclnnBatchMatMul** (only three-dimensional matrix multiplication is supported, whose first dimension is the **batch** dimension). T-C, T-T, K-C, K-T, and mx quantization modes are supported (../../../docs/en/context/quant_more_introduction.md).

- Formula:

    <details>
    <summary><term>Atlas inference products</term>, <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950PR/Ascend 950DT</summary>

    - No x1Scale, no bias:

    $$
    out = x1@x2 * x2Scale + x2Offset
    $$

    - bias INT32:

    $$
    out = (x1@x2 + bias) * x2Scale + x2Offset
    $$

    - With x1Scale and no bias:

    $$
    out = x1@x2 * x2Scale * x1Scale
    $$

    - With x1Scale and bias of INT32 type:

    $$
    out = (x1@x2 + bias) * x2Scale * x1Scale
    $$

    </details>

    <details>
    <summary><term>Atlas A2 training products/Atlas A2 inference products</term> <term>Atlas A3 training products/Atlas A3 inference products</term> Ascend 950PR/Ascend 950DT</summary>

    - bias BFLOAT16/FLOAT32 (no x2Offset in this scenario):

    $$
    out = x1@x2 * x2Scale + bias
    $$

    - With x1Scale, no bias:

    $$
    out = x1@x2 * x2Scale * x1Scale
    $$

    - With x1Scale, bias INT32 (no x2Offset in this scenario):

    $$
    out = (x1@x2 + bias) * x2Scale * x1Scale
    $$

    - With x1Scale, bias BFLOAT16/FLOAT16/FLOAT32 (no x2Offset in this scenario):

    $$
    out = x1@x2 * x2Scale * x1Scale + bias
    $$

    - x1 is INT8, x2 is INT32, x1Scale is FLOAT32, x2Scale is UINT64, and yOffset is FLOAT32:

    $$
    out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale
    $$

    </details>

    <details>
    <summary>Ascend 950PR/Ascend 950DT</summary>
 
    - mx quantization mode:

        $$
        out[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
        $$
        
        gsM, gsN, and gsK represent groupSizeM, groupSizeN, and groupSizeK, respectively. x1Slice represents the vector of groupSizeK length in row m of x1, and x2Slice represents the vector of groupSizeK length in column n of x2. The K axis is sliced from the start of j*groupSizeK. The value range of j is [0, kLoops], where kLoops = ceil(K / groupSizeK). K indicates the length of the K axis. The length of the last slice can be less than groupSizeK.
    </details>

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnQuantMatmulWeightNzGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnQuantMatmulWeightNz** is called to perform computation.

```cpp
aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(
    const aclTensor *x1, 
    const aclTensor *x2, 
    const aclTensor *x1Scale, 
    const aclTensor *x2Scale, 
    const aclTensor *yScale, 
    const aclTensor *x1Offset, 
    const aclTensor *x2Offset, 
    const aclTensor *yOffset, 
    const aclTensor *bias, 
    bool             transposeX1, 
    bool             transposeX2, 
    int64_t          groupSize, 
    aclTensor       *out, 
    uint64_t        *workspaceSize, 
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulWeightNz(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnQuantMatmulWeightNzGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1552px"><colgroup>
  <col style="width: 198px">
  <col style="width: 121px">
  <col style="width: 220px">
  <col style="width: 450px">
  <col style="width: 165px">
  <col style="width: 115px">
  <col style="width: 138px">
  <col style="width: 145px">
  </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
    <tr>
        <td>x1</td>
        <td>Input</td>
        <td>Input x1 in the formula.</td>
        <td>
          <ul>
              <li>Non-contiguous tensors are supported when the last two axes are transposed. In other scenarios, <a href="../../../docs/en/context/non_contiguous_tensor.md">non-contiguous tensors</a> are not supported.</li>
              <li>When transposeX1 is set to false, the dimensions are represented as follows: (batch, m, k). The batch dimension may not exist.</li>
              <li>When transposeX1 is true, the dimensions are represented as (batch, k, m). The batch dimension may not exist.</li>
          </ul>
        </td>
        <td>INT4<sup>1, 3</sup>, INT8, FLOAT8_E4M3FN<sup>1, 2</sup></td>
        <td>ND</td>
        <td>2-6</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Input x2 in the formula.</td>
        <td>
          <ul>
              <li>When transposeX2 is true, the dimensions are represented as (batch, k1, n1, n0, k0). The batch dimension may not exist, k0 = 32, and n0 = 16.</li>
              <li>When transposeX2 is false, the dimensions are represented as (batch, n1, k1, k0, n0). The batch dimension may not exist, k0 = 16, and n0 = 32.</li>
              <li>The k in the x1 shape and the k1 in the x2 shape must meet the following condition: ceil(k/k0) = k1.<br>The n1 in the x2 shape and the n in the output must meet the following condition: ceil(n/n0) = n1.</li>
          </ul>
        </td>
        <td>INT4<sup>1, 3</sup>, INT8, INT32<sup>1, 3</sup>, FLOAT4_E2M1<sup>1, 2</sup>, FLOAT32<sup>1, 2</sup>, FLOAT8_E4M3FN<sup>1, 2</sup></td>
        <td>NZ</td>
        <td>4-8</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x1Scale</td>
        <td>Input</td>
        <td>Quantization parameter, which is the input x1Scale in the formula. This parameter is optional.</td>
        <td>
          <ul>     
              <li>When x1Scale is FLOAT8_E8M0, x2Scale is a 3D tensor. The dimensions are as follows: (m, ceil(k/64), 2) if transposeX1 is false, and (ceil(k/64), m, 2) if transposeX1 is true.</li>
              <li>When x1Scale is FLOAT32, the shape is 1D (t,), where t = m, and m is the same as that of x1.</li>
          </ul>
        </td>
        <td>FLOAT32<sup>1</sup>, FLOAT8_E8M0<sup>1, 2</sup></td>
        <td>ND</td>
        <td>1, 3</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2Scale</td>
        <td>Input</td>
        <td>Quantization parameter, corresponding to the input x2Scale in the formula.</td>
        <td>
          <ul>     
              <li>When x2Scale is FLOAT8_E8M0, x2Scale is a 3D tensor. The dimensions are as follows: (ceil(k/64), n, 2) if transposeX2 is false, and (n, ceil(k/64), 2) if transposeX2 is true.</li>
              <li>When x2Scale is of another dtype, the shape is 1D (t,), where t = 1 or n, and n is the same as that of x2.</li>
              <li>If the original input type does not conform with the data type combinations described in <a href="#constraints">Constraints</a>, call the aclnn API of the TransQuantParamV2 operator to convert scale to the INT64 or UINT64 type in advance.</li>
          </ul>
        </td>
        <td>UINT64, INT64, FLOAT32<sup>1</sup>, BFLOAT16<sup>1</sup>, FLOAT8_E8M0<sup>1, 2</sup></td>
        <td>ND</td>
        <td>1, 3</td>
        <td>-</td>
    </tr>
    <tr>
        <td>yScale</td>
        <td>Input</td>
        <td>Dequantization scale parameter of the output y.</td>
        <td>
            The shape is 2-dimensional (1, n), where n is the same as that of x2.
        </td>
        <td>UINT64<sup>1, 2</sup>, INT64<sup>1, 2</sup></td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x1Offset</td>
        <td>Input</td>
        <td>Reserved parameter.</td>
        <td>
            This parameter is not supported in the current version. You need to pass nullptr or an empty tensor.
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2Offset</td>
        <td>Input</td>
        <td>Optional quantization parameter, which is the input x2Offset in the formula.</td>
        <td>
            The shape is 1-dimensional (t,), where t is 1 or n, and n is the same as that of x2.
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>yOffset</td>
        <td>Input</td>
        <td>Input yOffset in the formula.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Input bias in the formula. This parameter is optional.</td>
        <td>
          <ul>
              <li>The shape can be 1D (n,) or 3D (batch, 1, n). The value of n is the same as that of x2.</li>
              <li>When the shape of out is 2, 4, 5, or 6, the shape of bias supports only 1D (n,).</li>
          </ul>
        </td>
        <td>INT32, BFLOAT16<sup>1</sup>, FLOAT16<sup>1</sup>, FLOAT32<sup>1</sup></td>
        <td>ND</td>
        <td>1, 3</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transposeX1</td>
        <td>Input</td>
        <td>Whether the input shape of x1 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2</td>
        <td>Input</td>
        <td>Whether the input shape of x2 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize</td>
        <td>Input</td>
        <td>Quantization group size in the m, n, and k directions.</td>
        <td>
            The value consists of three group sizes in the m, n, and k directions, respectively. Each group size occupies 16 bits, and the three group sizes together occupy the lower 48 bits of the int64_t group size (the upper 16 bits are invalid). For details about the calculation formula, see formula 1 below. If group size is not supported, pass 0 to this parameter.
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    <tr>
        <td>out</td>
        <td>Output</td>
        <td>`out` in the formula.</td>
        <td>
          <ul>
              <li>The shape supports 2 to 6 dimensions, (batch, m, n). The batch dimension may not exist.</li>
              <li>Broadcasting of the batch dimensions of x1 and x2 is supported. The output batch is the same as the broadcast batch. The value of m is the same as that of x1. The relationship between n and n1 and n0 of x2 is ceil(n / n0) = n1.</li>
          </ul>
        </td>
        <td>FLOAT16, INT8, BFLOAT16<sup>1</sup>, INT32<sup>1</sup>, FLOAT32<sup>1, 2</sup></td>
        <td>ND</td>
        <td>2-6</td>
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
    </tbody></table>

  - Formula 1:

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

    <details>

    <summary><term>Atlas inference products</term></summary>

    - The superscript "1" in the data type column of the preceding table indicates that the data type is not supported by the series.
    - x2 does not support[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md).
    - yScale is not supported.
    - groupSize is not supported. If groupSize is passed, the value is 0.
    </details>

    <details>

    <summary><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></summary>

    - The superscript "2" in the "Data Type" column of the table above indicates data types that are not supported by the products.
    -[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - yScale is not supported.
    - groupSize is not supported. If groupSize is set to 0,
    </details>

    <details>

    <summary>Ascend 950PR/Ascend 950DT</summary>

    - The superscript "3" in the data type column of the preceding table indicates that the data type is not supported by the series.
    - x2 supports[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) when the last two axes are transposed. In other scenarios,[Non-Contiguous Tensor](../../../docs/en/context/non_contiguous_tensor.md) are not supported.
    - groupSize can be set to a non-zero value.
    </details>

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
      <td>The input x1, x2, x2Scale, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or format of x1, x2, bias, x2Scale, x2Offset, or out is not supported.</td>
    </tr>
    <tr>
        <td>The shape of x1, x2, bias, x2Scale, x2Offset, or out does not meet the verification conditions.</td>
    </tr>
    <tr>
        <td>x1, x2, bias, x2Scale, x2Offset, or out is an empty tensor.</td>
    </tr>
    <tr>
        <td>The size of the last dimension of x1 and x2 exceeds 65535. The last dimension of x1 refers to m when transposeX1 is true or k when transposeX1 is false. The last dimension of x2 refers to k when transposeX2 is true or n when transposeX2 is false.</td>
    </tr>
    <tr>
        <td>The input yScale, x1Offset, and yOffset are not nullptr and are not empty tensors.</td>
    </tr>
    <tr>
        <td>The input groupSize does not meet the verification conditions.</td>
    </tr>
    </tbody>
    </table>

## aclnnQuantMatmulWeightNz

- **Parameters:**

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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnQuantMatmulWeightNzGetWorkspaceSize API.</td>
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
  </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic description:
  - The default deterministic implementation of aclnnQuantMatmulWeightNz is used.

<details>
<summary><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></summary>

  - Before calling this API, you can use [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) to process x2 whose format is ND to obtain the AI processor affinity data layout format.

  - The input and output support the following data type combinations:

    | x1   | x2    | x1Scale      | x2Scale          | x2Offset      | yOffset | bias                        | out               |
    | ---- | ----- | ------------ | ---------------- | ------------- | ------- | --------------------------- | ----------------- |
    | INT8 | INT8  | null         | UINT64/INT64     | null          | null    | null/INT32                  | FLOAT16           |
    | INT8 | INT8  | null         | UINT64/INT64     | null/FLOAT32  | null    | null/INT32                  | INT8              |
    | INT8 | INT8  | null/FLOAT32 | FLOAT32/BFLOAT16 | null          | null    | null/INT32/BFLOAT16/FLOAT32 | BFLOAT16          |
    | INT8 | INT8  | FLOAT32      | FLOAT32          | null          | null    | null/INT32/FLOAT16/FLOAT32  | FLOAT16           |
    | INT8 | INT8  | null         | FLOAT32/BFLOAT16 | null          | null    | null/INT32                  | INT32             |
    | INT4 | INT4  | null/FLOAT32 | BFLOAT16         | null/FLOAT32  | null    | null/BFLOAT16               | BFLOAT16          |
    | INT4 | INT4  | null/FLOAT32 | FLOAT32          | null/FLOAT32  | null    | null/BFLOAT16               | BFLOAT16          |
    | INT4 | INT4  | null/FLOAT32 | UINT64           | null/FLOAT32  | null    | null/INT32                  | FLOAT16           |
    | INT4 | INT4  | null/FLOAT32 | FLOAT32          | null/FLOAT32  | null    | null/INT32                  | FLOAT16           |
    | INT8 | INT32 | FLOAT32      | UINT64           | null          | FLOAT32 | null                        | FLOAT16/BFLOAT16  |
    
  - Restrictions on x1: When the data type is INT8 and the data type of x2 is INT32, transposeX1 must be false. The dimension is (m, k). k must be an even number and less than 29576.
  - Restrictions on yOffset: The shape supports one dimension (n). It is the auxiliary result of offline computation during computation. The value must be 8\*x2\*x2Scale and accumulated in the first dimension.

</details>

<details>
<summary><term>Atlas inference products</term></summary>

  - Before calling this API, you can use [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) to process x2 whose format is ND to obtain the AI processor affinity data layout format.

  - The input and output support the following data type combinations:

    | x1   | x2   | x1Scale | x2Scale      | x2Offset      | bias       | out     |
    | ---- | ---- | ------- | ------------ | ------------- | ---------- | ------- |
    | INT8 | INT8 | null    | UINT64/INT64 | null          | null/INT32 | FLOAT16 |
    | INT8 | INT8 | null    | UINT64/INT64 | null/FLOAT32  | null/INT32 | INT8    |
    | INT8 | INT8 | FLOAT   | FLOAT        | null          | null/INT32 | FLOAT16 |

  - When x1Scale is not null, only K-C quantization is supported.

</details>

<details>
<summary>Ascend 950PR/Ascend 950DT</summary>

  - Before calling this API, you can use [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) or [aclnnNpuFormatCast](https://gitcode.com/cann/ops-math/blob/master/conversion/npu_format_cast/docs/aclnnNpuFormatCast.md) to process x2 whose format is ND to obtain the NZ format. When using this API, you must fill 0 to prevent dirty data from being introduced.

  - If the last two dimensions of the original ND are 1, the weightNz feature cannot be used. This API does not support this scenario.

  - **Restrictions on the T-C quantization and T-T quantization scenarios:**
  <a id="T-C quantization and T-T quantization"></a>
    - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .TC/TT"></a>

      | x1              | x2          | x1Scale     |   x2Scale         | x2Offset     | yScale | bias                       |   out                   |
      | --------------- | ----------- | ----------- |   --------------- | ------------ | -------| -------------------------- |   ----------------------|
      | INT8            | INT8        | null        | UINT64/INT64     | null         | null   | null/INT32                 |   FLOAT16/ BFLOAT16     |
      | INT8            | INT8        | null        | UINT64/INT64     | null/FLOAT32 | null   | null/INT32                 |   INT8                  |
      | INT8            | INT8        | null        | FLOAT32/BFLOAT16  | null         | null   | null/INT32/FLOAT32/BFLOAT16|   BFLOAT16              |
      | INT8            | INT8        | null        | FLOAT32/BFLOAT16  | null         | null   | null/INT32                 |   INT32                 |

    - In T-T quantization, the shape of x1Scale is nullptr, and the shape of x2Scale is (1,).
    - In T-C quantization, the shape of x1Scale is nullptr, and the shape of x2Scale is (n,), where n is the same as that of x2.
    - Dynamic T-C or T-T quantization is not supported.

  - **Restrictions on K-C and K-T quantization:**
  <a id="K-C and K-T Quantization"></a>
    - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .KC/KT"></a>

      | x1                   | x2                   | x1Scale | x2Scale         | x2Offset | yScale |   bias                      | out             |
      | -------------------- | -------------------- | ------- | --------------- | -------- | -------|   ------------------------- | --------------- |
      | INT8                 | INT8                 | FLOAT32 | FLOAT32/BFLOAT16| null     | null   | null/INT32/FLOAT32/BFLOAT16 | BFLOAT16        |
      | INT8                 | INT8                 | FLOAT32 | FLOAT32         | null     | null   | null/INT32/FLOAT32/FLOAT16  | FLOAT16         |

    - In K-C quantization, the shape of x1Scale is (m,), and the shape of x2Scale is (n,), where m is the same as that of x1 and n is the same as that of x2.
    - In K-T quantization, the shape of x1Scale is (m,), and the shape of x2Scale is (1,), where m is the same as that of x1.
  
  - **Restrictions on mx quantization:**
  <a id="mx Quantization"></a>
    - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .mx"></a>

      | x1            | x2            | x1Scale     | x2Scale     | x2Offset | yScale | bias         | out                         |
      |---------------| ------------- | ----------- | ----------- | -------- | ------ | -------------| --------------------------- |
      | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null     | null   | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32    |
                               
    - The value relationships between x1 dtype, x2 dtype, x1, x2, x1Scale, x2Scale, and groupSize are as follows:

      |Quantization Type|x1 dtype|x2 dtype|x1 shape|x2 shape|x1Scale shape|x2Scale shape|bias shape|yScale shape|[groupSizeM, groupSizeN, groupSizeK]|groupSize|
      |-------|--------|--------|--------|--------|-------------|-------------|------------|---------------------------------------|--|--|
      |mx full quantization|FLOAT8_E4M3FN|FLOAT8_E4M3FN|<li>Non-transposed: (batch, m, k)</li><li> Transposed: (batch, k, m)</li>|<li>Non-transposed: (batch, k, n)</li><li> Transposed: (batch, n, k)</li>|<li>Non-transposed: (m, ceil(k / 64), 2)</li><li> Transposed: (ceil(k / 64), m, 2)</li>|<li>Non-transposed: (ceil(k / 64), n, 2)</li><li> Transposed: (n, ceil(k / 64), 2)</li>|(n,) or (batch, 1, n)|null|[1, 1, 32]|4295032864|

    - In the mx full quantization scenario, when the x1 and x2 data types are FLOAT8_E4M3FN, the transposition attributes of x1 and x1Scale must be the same, and those of x2 and x2Scale must be the same.

  - **In the fake-quantization scenario, the dtype and shape requirements are as follows:**
  
    |Quantization Type|x1 dtype       |x2 dtype     | x1Scale dtype  |x2Scale dtype |bias dtype   |x1 shape  | x2 shape| x1Scale shape | x2Scale shape     |bias shape | yScale shape| [groupSizeM, groupSizeN, groupSizeK]|
    |---------------| ------------| -------------- |--------------|-------------|--------- | --------| --------------| ------------      |---------- | ------------| ---------------------------------------|-------|
    | mx quantization|FLOAT8_E4M3FN  |FLOAT4_E2M1  |FLOAT8_E8M0     |FLOAT8_E8M0   |null/BFLOAT16|(m, k)  |(n, k)  |(m, k/32)    |(n, k/32)        |(1, n)    | null        | [0, 0, 32] / [1, 1, 32]                |
    | mx quantization|FLOAT8_E4M3FN  |FLOAT32      |FLOAT8_E8M0     |FLOAT8_E8M0   |null/BFLOAT16|(m, k)  |(n, k/8)|(m, k/32)    |(n, k/32)        |(1, n)    | null        | [0, 0, 32] / [1, 1, 32]                |
    | T-CG quantization|FLOAT8_E4M3FN  |FLOAT4_E2M1  |null            |BFLOAT16      |null         |(m, k)  |(k, n)  |null           |(k/32, n)        |null       |(1, n)      | [0, 0, 32] / [1, 1, 32]                |
    | T-CG quantization|FLOAT8_E4M3FN  |FLOAT32      |null            |BFLOAT16      |null         |(m, k)  |(k, n/8)|null           |(k/32, n)        |null       |(1, n)      | [0, 0, 32] / [1, 1, 32]                |
    
    - Constraints:
      - The sizes of k and n must be 64-byte aligned.
      - When x1 is of type FLOAT8_E4M3FN and x2 is of type float32, x2 indicates a format in which eight FLOAT4_E2M1 elements are tightly packed into one float32 element.

</details>  

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

  The following is the sample code for the scenario where x2 is in NZ format (transposeX2=false):.

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
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
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // Call aclrtMalloc to allocate memory on the device.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API.
      std::vector<int64_t> x1Shape = {5, 32};
      std::vector<int64_t> x2Shape = {32, 32};
      std::vector<int64_t> biasShape = {32};
      std::vector<int64_t> offsetShape = {32};
      std::vector<int64_t> scaleShape = {32};
      std::vector<int64_t> outShape = {5, 32};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(5 * 32, 1);
      std::vector<int8_t> x2HostData(32 * 32, 1);
      std::vector<int32_t> biasHostData(32, 1);
      std::vector<float> scaleHostData(32, 1);
      std::vector<float> offsetHostData(32, 1);
      std::vector<uint16_t> outHostData(5 * 32, 1);  // The output data is actually in float16 half-precision mode.
      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor in NZ format.
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // Call the first-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // Call the aclnn API of the TransQuantParamV2 operator in advance for scale of the FLOAT data type.
      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulWeightNz.
      workspaceSize = 0;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, x2, nullptr, quantParam, nullptr, nullptr, nullptr, nullptr,
                                                    bias, transposeX1, transposeX2, 0, out, &workspaceSize,
                                                    &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrNZ(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrNZ.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulWeightNz.
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // The fp16 data cannot be directly printed in the C language. The data needs to be read by using uint16 and converted into fp16 in binary mode.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- <term>Atlas inference products</term>:
  The following is an example of the code in the NZ format (transposeX2=true):

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
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
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // Call aclrtMalloc to allocate memory on the device.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API.
      std::vector<int64_t> x1Shape = {5, 32};
      std::vector<int64_t> x2Shape = {32, 32};
      std::vector<int64_t> x2TransposedShape = {32, 32};
      std::vector<int64_t> biasShape = {32};
      std::vector<int64_t> offsetShape = {32};
      std::vector<int64_t> scaleShape = {32};
      std::vector<int64_t> outShape = {5, 32};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2TransposedDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *x2Transposed = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(5 * 32, 1);
      std::vector<int8_t> x2HostData(32 * 32, 1);
      std::vector<int8_t> x2TransposedHostData(32 * 32, 1);
      std::vector<int32_t> biasHostData(32, 1);
      std::vector<float> scaleHostData(32, 1);
      std::vector<float> offsetHostData(32, 1);
      std::vector<uint16_t> outHostData(5 * 32, 1);  // The output data is actually in float16 half-precision mode.
      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor in NZ format.
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Transposed aclTensor in NZ format.
      ret = CreateAclTensorX2(x2TransposedHostData, x2TransposedShape, &x2TransposedDeviceAddr,
                              aclDataType::ACL_INT8, &x2Transposed);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TransposedHPTensorPtr(x2Transposed,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2TransposedHPDeviceAddrPtr(x2TransposedDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = true;

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // The shape of x2 needs to be transposed to the nk format before TransData.
      std::vector<int64_t> dimsData = {1, 0};
      // Create a dims aclIntArray.
      aclIntArray *dims = aclCreateIntArray(dimsData.data(), dimsData.size());
      // Call the first-phase API of aclnnPermute.
      ret = aclnnPermuteGetWorkspaceSize(x2, dims, x2Transposed, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrPermute(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrPermute.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnPermute.
      ret = aclnnPermute(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      workspaceSize = 0;
      // Call the first-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2Transposed, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // Call the aclnn API of the TransQuantParamV2 operator in advance for scale of the FLOAT data type.
      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulWeightNz.
      workspaceSize = 0;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, x2Transposed, nullptr, quantParam, nullptr, nullptr,
                                                    nullptr, nullptr, bias, transposeX1, transposeX2, 0, out,
                                                    &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrNZ(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrNZ.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulWeightNz.
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // The fp16 data cannot be directly printed in the C language. The data needs to be read by using uint16 and converted into fp16 in binary mode.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- Ascend 950PR/Ascend 950DT:
  The following is an example of the code in the NZ format (transposeX2=true):

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_npu_format_cast.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"

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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  // Convert the uint16_t representation of bfloat16 to the float representation.
  float Bf16ToFloat(uint16_t h)
  {
      uint32_t sign = (h & 0x8000U) ? 0x80000000U : 0x00000000U; // sign bit
      uint32_t exponent = (h >> 7) & 0x00FFU;                    // exponent bits
      uint32_t mantissa = h & 0x007FU;                           // mantissa bits
      // The exponent offset remains unchanged.
      // Shift the mantissa left by 23 - 7 and pad the rest with zeros.
      uint32_t fBits = sign | (exponent << 23) | (mantissa << (23 - 7));
      // Forcibly cast to float.
      return *reinterpret_cast<float*>(&fBits);
  }

  template <typename T>
  int CreateAclTensorWithFormat(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape,
      uint64_t* storageShapeSize, void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format)
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

      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, format, *storageShape, *storageShapeSize, *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API.
      int64_t m = 5;
      int64_t k = 64;
      int64_t n = 128;
      bool transposeX1 = false;
      bool transposeX2 = true;
      int64_t groupSize = 32;
      std::vector<int64_t> x1Shape = {m, k};
      std::vector<int64_t> x2Shape = {n, k};
      std::vector<int64_t> x1ScaleShape = {m, k / groupSize / 2, 2};
      std::vector<int64_t> x2ScaleShape = {n, k / groupSize / 2, 2};
      std::vector<int64_t> outShape = {m, n};
      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x2NzDeviceAddr = nullptr;
      void* x2NzFp4DeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* out = nullptr;
      std::vector<uint8_t> x1HostData(m * k, 0b00111000); // float8_e4m3 1.0
      The input of std::vector<float> x2HostData(n * k, 1); // is fp32, which is converted to Nz and then cast to fp4.
      std::vector<uint8_t> x1ScaleHostData(m * k / groupSize, 0b01111111); // float8_e8m0 1.0
      The value of std::vector<uint8_t> x2ScaleHostData(n * k / groupSize, 0b10000101); // float8_e8m0 is 1.0 x 64. The input in the reference document needs to be multiplied by 64.
      std::vector<uint16_t> outHostData(m * k, 0); // is actually bfloat16.
      std::vector<int32_t> x2NzHostData(k * n, 0);
      std::vector<int32_t> x2NzFp4HostData(k * n, 0);
      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      void* dstDeviceAddr = nullptr;
      aclTensor* x2Nz = nullptr;
      aclTensor* x2NzFp4 = nullptr;
      int actualFormat;

      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // X2-to-NZ conversion
      // Calculate the shape and format of the target tensor.
      aclDataType srcDtype = aclDataType::ACL_FLOAT;
      aclDataType additionalDtype = aclDataType::ACL_FLOAT;
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(x2, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
                return ret);

      ret = CreateAclTensorWithFormat(
          x2NzHostData, x2Shape, &dstShape, &dstShapeSize, &x2NzDeviceAddr, srcDtype, &x2Nz,
          static_cast<aclFormat>(actualFormat));
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2NzTensorPtr(x2Nz, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2NzDeviceAddrPtr(x2NzDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      ret = CreateAclTensorWithFormat(
          x2NzFp4HostData, x2Shape, &dstShape, &dstShapeSize, &x2NzFp4DeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x2NzFp4,
          static_cast<aclFormat>(actualFormat));
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2NzFp4TensorPtr(x2NzFp4, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2NzFp4DeviceAddrPtr(x2NzFp4DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnNpuFormatCastGetWorkspaceSize.
      ret = aclnnNpuFormatCastGetWorkspaceSize(x2, x2Nz, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceNzAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceNzAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceNzAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceNzAddrPtr.reset(workspaceNzAddr);
      }

      // Call the second-phase API of aclnnNpuFormatCastGetWorkspaceSize.
      ret = aclnnNpuFormatCast(workspaceNzAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      //Call cast to convert the fp32 type of x2 to fp4_e2m1.
      ret = aclnnCastGetWorkspaceSize(x2Nz, aclDataType::ACL_FLOAT4_E2M1, x2NzFp4, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceCastAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceCastAddrPtr.reset(workspaceCastAddr);
      }
      ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulWeightNz.
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(
          x1, x2NzFp4, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, bias, transposeX1, transposeX2, groupSize,
          out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulWeightNz.
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0); // In the C language, fp16 data cannot be directly printed. You need to read the data using uint16 and convert the data to fp16 using the binary format.
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %.1f\n", i, Bf16ToFloat(resultData[i]));
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
