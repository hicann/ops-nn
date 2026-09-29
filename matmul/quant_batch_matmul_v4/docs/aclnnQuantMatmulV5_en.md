# aclnnQuantMatmulV5

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/quant_batch_matmul_v4)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                            |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- Description: Performs matrix multiplication for quantization.
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    It is compatible with the **aclnnQuantMatmulV3** and **aclnnQuantMatmulV4** APIs. Performs quantized matrix multiplication. The minimum input dimension is 1 and the maximum input dimension is 2. Similar APIs include **aclnnMm** (only two-dimensional tensors can be used as the input of matrix multiplication).
  - Ascend 950PR/Ascend 950DT:

    Compatible with the aclnnQuantMatmulV3 and aclnnQuantMatmulV4 APIs. In addition to the functions of the two APIs, this API supports the G-B, B-B, T-CG, and mx quantization modes (../../../docs/en/context/quant_more_introduction.md), and the x1 and x2 inputs support the FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8 and FLOAT4_E2M1 data types. Performs quantized matrix multiplication, supporting at least two-dimensional input and at most six-dimensional-dimensional input. Similar APIs include **aclnnMm** (only two-dimensional tensors can be used as the input of matrix multiplication) and **aclnnBatchMatMul** (only three-dimensional matrix multiplication is supported, whose first dimension is the **batch** dimension).

- Formula:
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    K-C && K-T, T-C && T-T, G-B and K-G quantization modes are supported. For details about the input and output data type combinations corresponding to different quantization modes, see [Constraints](#constraints).

    <details>
    
    <summary>K-G quantization mode</summary>
    
      - x1 is of type INT8, x2 is of type INT32, x1Scale is of type FLOAT32, x2Scale is of type UINT64 or INT64, and yOffset is of type FLOAT32:

        $$
        out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale
        $$
        
      - x1 and x2 are of type INT4, x1Scale and x2Scale are of type FLOAT32, x2Offset is of type FLOAT16, and out is of type FLOAT16 or BFLOAT16 (asymmetric quantization per token per group):

        $$
        out = x1Scale * x2Scale * (x1 @ x2 - x1 @ x2Offset)
        $$

    </details>

    <details>

    <summary>K-C and K-T quantization modes</summary>

      - With x1Scale, no bias:

        $$
        out = x1@x2 * x2Scale * x1Scale
        $$

      - x1Scale and bias are of type INT32 (no offset in this scenario):

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

      - x1Scale and bias are of type BFLOAT16, FLOAT16, or FLOAT32 (no offset in this scenario):

        $$
        out = x1@x2 * x2Scale * x1Scale + bias
        $$

    </details>

    <details>

    <summary>T-C and T-T quantization modes</summary>

      - No x1Scale, no bias:

        $$
        out = x1@x2 * x2Scale + x2Offset
        $$
      
      - bias (INT32):

        $$
        out = (x1@x2 + bias) * x2Scale + x2Offset
        $$

      - bias (BFLOAT16/FLOAT32) (no offset in this scenario):
  
        $$
        out = x1@x2 * x2Scale + bias
        $$

    </details>

    <details>

    <summary>G-B quantization mode</summary>

      - x1 and x2 are of type INT8, x1Scale and x2Scale are of type FLOAT32, bias is of type FLOAT32, and out is of type FLOAT16 or BFLOAT16 (pergroup-perblock quantization):

        $$
        out = (x1 @ x2) * x1Scale * x2Scale + bias
        $$

    </details>
  
  - <term>Atlas inference products</term>:

    K-C [quantization mode](../../../docs/en/context/quant_more_introduction.md) is supported. For details about the input and output data type combinations corresponding to different quantization modes, see [Constraints](#constraints).

    <details>

    <summary>K-C quantization mode</summary>

      - With x1Scale, no bias:

        $$
        out = x1@x2 * x2Scale * x1Scale
        $$

      - x1Scale and bias of type INT32 (no offset in this scenario):

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

    </details>

  - Ascend 950PR/Ascend 950DT:

    T-C && T-T, K-C && K-T, G-B , B-B , mx and T-CG quantization modes are supported. For details about the input and output data type combinations corresponding to different quantization modes, see [Constraints](#constraints).

    <details>

    <summary>T-C and T-T quantization modes</summary>

      - x1 and x2 are of type int8. x1Scale is not supported. x2Scale is of type int64 or uint64. The optional parameter x2Offset is of type float32, and the optional parameter bias is of type int32.

        $$
        out = (x1@x2 + bias) * x2Scale + x2Offset
        $$

      - x1 and x2 are of type int8. x1Scale is not supported. x2Scale is of type int64 or uint64. The optional parameter bias is of type int32.
      x1 and x2 are of type FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8. x1Scale is not supported. x2Scale is of type int64 or uint64. The optional parameter bias is of type float32.

        $$
        out = (x1@x2 + bias) * x2Scale
        $$

      - x1 and x2 are of type int8. x1Scale is not supported. x2Scale is of type bfloat16 or float32. The optional parameter bias is of type bfloat16 or float32.

        $$
        out = x1@x2 * x2Scale + bias
        $$

      - x1 and x2 are of type FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8. x1Scale is of type float32. x2Scale is of type float32. The optional parameter bias is of type float32.

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

    </details>

    <details>

    <summary>K-C and K-T quantization modes</summary>

      - x1 and x2 are of type int8. x1Scale is of type float32. x2Scale is of type bfloat16 or float32. The optional parameter bias is of type int32.
      or x1 and x2 are of type FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x1Scale is of type FLOAT32, x2Scale is of type FLOAT32, and the optional bias parameter is of type FLOAT32:

        $$
        out = (x1@x2 + bias) * x2Scale * x1Scale
        $$

      - x1 and x2 are of type INT8, x1Scale is of type FLOAT32, x2Scale is of type BFLOAT16 or FLOAT32, and the optional bias parameter is of type BFLOAT16 or FLOAT32.
      or x1 and x2 are of type INT8, x1Scale is of type FLOAT32, x2Scale is of type FLOAT32, and the optional bias parameter is of type FLOAT16 or FLOAT32:

        $$
        out = x1@x2 * x2Scale * x1Scale + bias
        $$

    </details>

    <details>

    <summary>G-B && B-B && mx: quantization mode </summary>

      $$
      out[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
      $$

      gsM, gsN, and gsK represent groupSizeM, groupSizeN, and groupSizeK, respectively. x1Slice represents the vector of length groupSizeK in the mth row of x1, and x2Slice represents the vector of length groupSizeK in the nth column of x2. The K axis is sliced from the start position of j x groupSizeK. The value range of j is [0, kLoops), where kLoops = ceil(K / groupSizeK). K indicates the length of the K axis. The length of the last slice can be less than groupSizeK. The bias parameter is included only in the mx quantization mode. For the G-B, B-B, and mx quantization modes, the value combinations of [groupSizeM, groupSizeN, groupSizeK] are [1, 128, 128], [128, 128, 128], and [1, 1, 32], respectively.

    </details>

    <details>

    <summary>T-CG quantization mode</summary>

      $$
      out = (x1@(x2 * x2Scale)) * yScale
      $$

    </details>

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnQuantMatmulV5GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnQuantMatmulV5** is called to perform computation.

```c++
aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(
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

```c++
aclnnStatus aclnnQuantMatmulV5(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulV5GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 248px">
  <col style="width: 121px">
  <col style="width: 210px">
  <col style="width: 327px">
  <col style="width: 250px">
  <col style="width: 115px">
  <col style="width: 138px">
  <col style="width: 145px">
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
        <td>x1(aclTensor*)</td>
        <td>Input</td>
        <td>Input x1 in the formula.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
            <li><a href="../../../docs/en/context/non_contiguous_tensor.md">Non-contiguous tensors</a> are supported only when the last m and k axes are transposed. In other scenarios, non-contiguous tensors are not supported.</li>
          </ul>
        </td>
        <td>INT4<sup>1</sup>, INT8, INT32<sup>1</sup>, FLOAT8_E4M3FN<sup>2</sup>, FLOAT8_E5M2<sup>2</sup>, HIFLOAT8<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2(aclTensor*)</td>
        <td>Input</td>
        <td>Input x2 in the formula.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
            <li>In NZ format, empty tensors are not supported.</li>
            <li>In NZ format, the shape supports 4 to 8 dimensions.</li>
            <li>In ND format, non-contiguous tensors are supported when the last two axes are transposed. In other scenarios, <a href="../../../docs/en/context/non_contiguous_tensor.md">non-contiguous tensors</a> are not supported.</li>
          </ul>
        </td>
        <td>INT4<sup>1</sup>, INT8, INT32<sup>1</sup>, FLOAT8_E4M3FN<sup>2</sup>, FLOAT8_E5M2<sup>2</sup>, HIFLOAT8<sup>2</sup>, FLOAT4_E2M1<sup>2</sup></td>
        <td>ND and NZ</td>
        <td>2-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1Scale(aclTensor*)</td>
        <td>Optional input</td>
        <td>Input x1Scale in the formula.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
          </ul>
          </td>
        <td>FLOAT32, FLOAT8_E8M0<sup>2</sup>, FLOAT8_E4M3FN<sup>2</sup>, FLOAT8_E5M2<sup>2</sup>, HIFLOAT8<sup>2</sup></td>
        <td>ND</td>
        <td>1-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2Scale(aclTensor*)</td>
        <td>Input</td>
        <td>Quantization parameter, corresponding to the input x2Scale in the formula.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
          </ul>
        </td>
        <td>UINT64, INT64, FLOAT32, BFLOAT16, FLOAT8_E8M0<sup>2</sup></td>
        <td>ND</td>
        <td>1-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>yScale(aclTensor*)</td>
        <td>Optional input</td>
        <td>Dequantization scale parameter of the output y.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
            <li>If this parameter is not used, pass nullptr.</li>
          </ul>
        </td>
        <td>UINT64<sup>2</sup>, INT64<sup>2</sup></td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>x1Offset(aclTensor*)</td>
        <td>Optional input</td>
        <td>Input x1Offset in the formula.</td>
        <td>
        <ul>
        <li>
        This parameter is reserved and is not supported in the current version. nullptr needs to be passed.
        </li>
        </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Offset(aclTensor*)</td>
        <td>Optional input</td>
        <td>Input x2Offset in the formula.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
          </ul>
        </td>
        <td>FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOffset(aclTensor*)</td>
        <td>Optional input</td>
        <td>Input yOffset in the formula.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bias(aclTensor*)</td>
        <td>Optional input</td>
        <td>Input bias in the formula.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
          </ul>
        </td>
        <td>INT32, FLOAT32, BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>1-3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX1(bool)</td>
        <td>Input</td>
        <td>Whether the input shape of x1 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2(bool)</td>
        <td>Input</td>
        <td>Whether the input shape of x2 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize(int64_t)</td>
        <td>Optional input</td>
        <td>Quantization group size in the m, n, and k directions.</td>
        <td>The value is composed of three values groupSizeM, groupSizeN, and groupSizeK in three directions. Each value occupies 16 bits, and the total value occupies the lower 48 bits of the groupSize of the int64_t type. The upper 16 bits of the groupSize are invalid. For details about the calculation formula, see the following table.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out(aclTensor)</td>
        <td>Output</td>
        <td>out in the formula.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
          </ul>
        </td>
        <td>FLOAT16, INT8, BFLOAT16, INT32, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize(uint64_t)</td>
        <td>Output</td>
        <td>Size of the workspace required to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap">executor(aclOpExecutor)</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

  - Note: Optional inputs refer to optional quantization parameters, and nullptr can be passed.
  
  - Ascend 950PR/Ascend 950DT:

    - The subscript "1" in the data type column of the preceding table indicates the data type that is not supported by this series.
    - The input parameters x1 and x2 do not support the INT4 and INT32 types.
    - When x2 is in ND format, if x1 is an empty tensor with m = 0 or x2 is an empty tensor with n = 0, the output is an empty tensor. When x2 is in FRACTAL_NZ format, if x1 is an empty tensor with m = 0, the output is an empty tensor.

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

    - The superscript "2" in the "Data Type" column of the table above indicates data types that are not supported by the products.

  - Calculation formula: <a name='f1'></a>

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

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
        <td>The passed x1, x2, x1Scale, x2Scale, yOffset, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="5">161002</td>
        <td>x1, x2, bias, x2Scale, x2Offset, or out is an empty tensor.</td>
      </tr>
      <tr>
        <td>The data type or format of x1, x2, bias, x1Scale, x2Scale, x2Offset, or out is not supported.</td>
      </tr>
      <tr>
        <td>The shape of x1, x2, bias, x1Scale, x2Scale, x2Offset, or out does not meet the verification condition.</td>
      </tr>
      <tr>
        <td>The input groupSize does not meet the verification conditions, or when the input groupSize is 0, the shape relationship between x1, x2, x1Scale, and x2Scale cannot be used to infer groupSize.</td>
      </tr>
    </tbody></table>

## aclnnQuantMatmulV5

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
    <td>Size of the workspace to be allocated on the device, which is obtained by first-phase API aclnnQuantMatmulV5GetWorkspaceSize.</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The default deterministic implementation of aclnnQuantMatmulV5 is used.

<details>

<summary><term>Atlas inference products</term></summary>

- **Common constraints:**
  <a id="common-constraints"></a>
  - The current version does not support yScale, x1Offset, x2Offset, and yOffset. You need to pass nullptr.

  <details>

  <summary>Restrictions in the K-C quantization scenario:</summary>
  <a id="K-Quantization with"></a>

  - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .K-C && K-T"></a>

    | x1         | x2           | x1Scale     | x2Scale      | x1Offset    | x2Offset    | yScale   | yOffset    | bias         | out                |
    | -----------| ------------ | ----------- | -----------  | ----------- |-----------  | -------  | -----------| ------------ | -------------------|
    | INT8       | INT8         | FLOAT32     | FLOAT32      | null        | null        | null     | null       | null/INT32   | FLOAT16            |

  - Restrictions on x1:
    - The size of the last dimension of x1 cannot exceed 65535. transposeX1 can only be false.
  - Restrictions on x2:
    - The size of the last dimension of x2 cannot exceed 65535. transposeX2 can only be true.
    - The shape of the input is (batch, k1, n1, n0, k0), where batch is optional, k0 = 32, n0 = 16, and k in x1 and k1 in x2 must meet the following relationship: ceil(k/32) = k1.
    - x2 needs to be processed by [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) to obtain the AI processor affinity data layout format from x2 in ND format.
  - Restrictions on x1Scale: The data format is ND, the shape is 1D (t,), and t = m, where m is the same as that of x1.
  - Restrictions on x2Scale: The data format is ND, the shape is 1D (t,), and t = n, where n is the same as that of x2.
  - Restrictions on bias: The data format is ND, and the shape can be 1D (n,) or 3D (batch, 1, n), where n is the same as that of x2.
      
  </details>

</details>

<details>

<summary><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></summary>

- **Common constraints:**
  <a id="Common restrictions.1"></a>
  - When transposeX1 is false, the shape of x1 is (batch, m, k). When transposeX1 is true, the shape of X1 is (batch, k, m), where batch may not exist.
  - The constraints for **transposeX2** are as follows:
    - In ND format, if transposeX2 is false, the shape of X2 is (batch, k, n); if transposeX2 is true, the shape of X2 is (batch, n, k), where batch may not exist. The value of k is the same as that of k in the shape of x1.
    - In NZ format:
      - If transposeX2 is true, the shape of X2 is (batch, k1, n1, n0, k0), where batch may not exist. The value of k0 is 32, and the value of n0 is 16. The value of k in the shape of x1 and the value of k1 in the shape of x2 must meet the following relationship: ceil(k/32) = k1.
      - If transposeX2 is false, the shape of X2 is (batch, n1, k1, k0, n0), where batch may not exist. The value of k0 is 16, and the value of n0 is 32. The value of k in the shape of x1 and the value of k1 in the shape of x2 must meet the following relationship: ceil(k/16) = k1.
      - You can use the aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight APIs to convert the input format from ND to NZ.
  - If the original input type of x2Scale does not meet the constraints in the quantization scenario, call the aclnnTransQuantParamV2 API to convert the scale to the INT64 or UINT64 data type.
  - **yScale** is not supported in the current version. Pass **nullptr**.
  - The shape of out supports 2 to 6 dimensions, (batch, m, n), where batch may not exist. The data type can be FLOAT16, INT8, BFLOAT16, or INT32.
  - When x1 and x2 are of type INT8, out is of type INT32, and bias is of type INT32 or nullptr, the actual scale is not involved in the computation. The computation formula is as follows:
    - bias INT32

        $$
        out = x1@x2 + bias
        $$

    - No bias

        $$
        out = x1@x2
        $$

  <details>

  <summary>Restrictions on G-B quantization:</summary>
  <a id="G-B quantization"></a>

  - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .GB"></a>
  
    | x1                        | x2                        | x1Scale     | x2Scale         | x2Offset    | yScale   | bias         | yOffset    | out                                    |
    | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
    | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | FLOAT32      | null       | BFLOAT16                               |

  - The value relationships of x1 shape, x2 shape, x1Scale shape, x2Scale shape, bias shape, and groupSize are as follows:

    |Quantization Type|x1 shape|x2 shape|x1Scale shape|x2Scale shape|bias shape|[gsM, gsN, gsK]|
    | ----- | ------ | ------ | ----------- | ----------- | ----------- | ----------- |
    | G-B quantization| (m, k) |(n, k)|(m, ceil(k / 128))|(ceil(n / 128),ceil(k / 128))| (n, ) | [1, 128, 128]|

  - Note: In the preceding table, gsM, gsK, and gsN indicate groupSizeM, groupSizeK, and groupSizeN, respectively.
  - Restrictions on x1: Currently, k must be 128-pixel aligned and be a multiple of 4 x 128, and transposeX1 must be false.
  - Restrictions on x2: Currently, n must be 256-pixel aligned, k must be 128-pixel aligned and be a multiple of 4 x 128, and transposeX2 must be true.

  </details>

  <details>

  <summary>Restrictions on the T-C && T-T && K-C && K-T quantization scenarios:</summary>
  <a id="T-C && T-T && K-C && K-T quantization"></a>;

  - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .K-C && K-T"></a>

    | x1                        | x2                        | x1Scale     | x2Scale         | x2Offset    | yScale   | bias         | yOffset    | out                                    |
    | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
    | INT8                      | INT32                     | FLOAT32     | UINT64/INT64    | null        | null     | null         | FLOAT32    | FLOAT16/BFLOAT16                       |
    | INT8                      | INT8                      | null        | UINT64/INT64    | null        | null     | null/INT32   | null       | FLOAT16                                |
    | INT8                      | INT8                      | null        | UINT64/INT64    | null/FLOAT32| null     | null/INT32   | null       | INT8                                   |
    | INT8                      | INT8                      | null/FLOAT32| FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
    | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
    | INT4/INT32                | INT4/INT32                | null        | UINT64/INT64    | null        | null     | null/INT32   | null       | FLOAT16                                |
    | INT8                      | INT8                      | null        | FLOAT32/BFLOAT16| null        | null     | null/INT32   | null       | INT32                                  |
    | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | FLOAT32      | null       | BFLOAT16                               |
    | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
    | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |

  - Restrictions on x1:
    - When the data type is INT4, transposeX1 is false. The dimension is (m, k), and k must be an even number.
    - When the data type is INT32, transposeX1 is false. Eight INT4 data elements are stored in each INT32 data element. The corresponding dimension is (m, ceil(k / 8)). k must be a multiple of 8.
    - When the data type is INT8 and the data type of x2 is INT32, transposeX1 is false. The dimension is (m, k), and k must be an even number.
  - Restrictions on x2:
    - When the data type is INT4:
      - Currently, only the 2D ND format is supported.
      - If **transposeX2** is set to **true**, the shape is (n, k), where **k** must be an even number.
      - If **transposeX2** is set to **false**, the shape is (k, n), where **n** must be an even number.
    - When the data type is INT32, each INT32 data entry stores eight INT4 data entries.
       - Currently, only the 2D ND format is supported.
       - When transposeX2 is true, the dimension is (n, ceil(k / 8)). k must be a multiple of 8.
       - When transposeX2 is false, the dimension is (k, ceil(n / 8)). n must be a multiple of 8.
       - The **aclnnConvertWeightToINT4Pack** API can be used to convert **x2** from INT32 (one int32 space stores one int4 data entry in bits 0–3) to INT32 (one int32 space stores eight int4 data entries) or INT4 (one int4 space stores one int4 data entry). For details, see [aclnnConvertWeightToINT4Pack](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack_en.md).

  - Restrictions on x1Scale: The data format is ND, the shape is 1D (t,), and t = m, where m is the same as that of x1.
  - Restrictions on x2Scale: The data format is ND, the shape is 1D (t,), and t = 1 or n, where n is the same as that of x2.
  - Restrictions on x2Offset: The data format is ND, the shape is 1D (t,), and t = 1 or n, where n is the same as that of x2.
  - Restrictions on bias:
      - The data layout can be ND. The shape can be one-dimensional (n,) or three-dimensional (batch, 1, n), where **n** is the same as **n** of **x2**.
      - When **x1** and **x2** are INT32 or INT4, the shape of **bias** can only be one-dimensional (n,).
      - When the shape of **out** is two-, four-, five-, or six-dimensional, the shape of **bias** can only be one-dimensional (n,).
  - Restrictions on yOffset: The shape can be 1D (n). It is an auxiliary result calculated offline during computation. The value must be 8 *x2* x2Scale and accumulated in the first dimension.

  </details>

  <details>

  <summary>Restrictions on the K-G quantization scenario:</summary>
  <a id="K-G quantization"></a>

  - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .K-G"></a>

      | x1                        | x2                        | x1Scale     | x2Scale         | x2Offset    | yScale   | bias         | yOffset    | out                                    |
      | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
      | INT8                      | INT32                     | FLOAT32     | UINT64/INT64    | null        | null     | null         | FLOAT32    | FLOAT16/BFLOAT16                       |
      | INT4                      | INT4                      | FLOAT32     | FLOAT32         | FLOAT16     | null     | null         | null       | BFLOAT16                               |

  - The value relationships between x1, x2, x1Scale, x2Scale, and groupSize are as follows:

    |Quantization Type| x1 Data Type                | x2 Data Type                | x1Scale Data Type| x2Scale Data Type| x1 shape | x2 shape| x1Scale shape| x2Scale shape|x2Offset shape| yOffset shape| [gsM, gsN, gsK]|
    | ----- | ------------------------- | ------------------------- | -------------- | ------------- | -------- | ------- | ------------ | ------ |------------ | ------------ | ------------ |
    | K-G quantization| INT8                    |INT32                   |FLOAT32              |UINT64/INT64 |(m, k) |(k, ceil(n / 8))|(m, 1)|(ceil(k / 256), n)|null| (n) | [0, 0, 256]|
    | K-G quantization| INT4                    |INT4                    |FLOAT32              |FLOAT32      |(m, k)|(n, k)|(m, 1)|(ceil(k / 256), n)|(ceil(k / 256), n)| null | [0, 0, 256]|
    
  - Restrictions on x1:
    - When the data type is INT8, k must be aligned with 256 and be less than 29576. transposeX1 is false.
    - When the data type is INT4, k must be aligned with 1024. transposeX1 is false.
  - Restrictions on x2:
    - When the data type is INT32, k must be aligned with 256. transposeX2 is false.
    - When the data type is INT4, k must be aligned with 1024 and n must be aligned with 256. transposeX2 is true.
  - Restrictions on x2Scale:
    - When the data type is UINT64 or INT64, the TransQuantParamV2 operator supports only one dimension. Therefore, you need to reshape x2Scale to a one-dimensional view (k / groupSize * n), call the aclnn API of the TransQuantParamV2 operator to convert x2Scale to the UINT64 or INT64 data type, and then reshape the output to a two-dimensional view (k / groupSize, n). The groupSize value is 256.
    - When x1 and x2 are of the INT4 type, the shape of x2Scale is (ceil(k / 256), n).

  </details>

</details>

<details>

<summary>Ascend 950PR/Ascend 950DT</summary>

- **Common constraints:**
  <a id="Common Restrictions 2"></a>
  
  - Shape of x1 when transposeX1 is false: (batch, m, k). Shape of x1 when transposeX1 is true: (batch, k, m). The first 0 to 4 dimensions represent batch. Dimension 0 indicates that the batch does not exist.
  - Shape of x2 when transposeX2 is false: (batch, k, n). Shape of x2 when transposeX2 is true: (batch, n, k). The first 0 to 4 dimensions represent batch. Dimension 0 indicates that the batch does not exist. k is the same as k in the shape of x1.
  - If the original input type of x2Scale does not meet the combination requirements in the quantization scenario, call the aclnnTransQuantParamV2 API to convert the scale to the INT64 or UINT64 data type.
  - yScale is supported only when x1 is FLOAT8_E4M3FN and x2 is FLOAT4_E2M1. The shape is 2-dimensional (1, n), where n is the same as that of x2.
  - x2Offset is supported only when the data types of x1 and x2 are both INT8 and the data type of out is INT8. For other input types, nullptr needs to be passed. The shape is 1-dimensional (t,). t can be 1 or n, where n is the same as that of x2.
  - yOffset is a reserved parameter and is not supported in the current version. nullptr or an empty tensor needs to be passed.
  - Restrictions on bias:
    - This parameter is optional. nullptr can be passed.
    - When the shape of out is 2, 4, 5, or 6, the shape of bias can be 1-dimensional (n,) or 2-dimensional (1, n).
    - When the shape of out is 3-dimensional, the shape of bias can be 1-dimensional (n,) or 3-dimensional (batch, 1, n).
  - Restrictions on groupSize:
    - It is valid only in the mx, G-B, B-B, and T-CG quantization modes (../../../docs/en/context/quant_more_introduction.md).
    - The value of groupSize is valid only when the input of x1Scale and x2Scale is 2-dimensional or higher. In other scenarios, 0 needs to be passed.
    - The input groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formulas. If one or more of them are 0, groupSizeM, groupSizeN, and groupSizeK will be reset based on the input shape of x1, x2, x1Scale, and x2Scale for computation. Principle: Assume that groupSizeM is 0, indicating that the quantization group size in the m direction is inferred by the API. The inference formula is groupSizeM = m/scaleM. (Ensure that m can be exactly divided by scaleM.) m is the same as that in the shape of x1, and scaleM is the same as that in the shape of x1Scale.

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

  - The output shape supports 2 to 6 dimensions, (batch, m, n). The batch dimension may not exist. The batch dimensions of x1 and x2 can be broadcast. The output batch is the same as the broadcast batch. m is the same as that of x1, and n is the same as that of x2.
  - When x1 and x2 are of type int8, out is of type int32, and bias is of type int32 or nullptr, the actual scale is not involved in the calculation. The calculation formula is as follows:
    - bias INT32

        $$
        out = x1@x2 + bias
        $$

    - No bias

        $$
        out = x1@x2
        $$

  <details>

  <summary>Restrictions on T-C and T-T quantization:</summary>
  <a id="T-C and T-T quantization"></a>

  - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .TC/TT"></a>

      | x1                        | x2                        | x1Scale     |   x2Scale     | x2Offset | yScale | bias    |   out                                    |
      | ------------------------- | ------------------------- | ----------- |   ----------- | -------- | -------| ------- |   -------------------------------------- |
      | INT8                      | INT8                      | null        | UINT64/INT64      | null     | null     | null/INT32   | FLOAT16/BFLOAT16                       |
      | INT8                      | INT8                      | null        | UINT64/INT64      | null/FLOAT32  | null     | null/INT32   |   INT8                              |
      | INT8                      | INT8                      | null        | FLOAT32/BFLOAT16  | null     | null     | null/INT32/FLOAT32/BFLOAT16   |   BFLOAT16              |
      | INT8                      | INT8                      | null        | FLOAT32/BFLOAT16  | null        | null     | null/INT32   |   INT32                                |
      | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | null        | UINT64/INT64      | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32 |
      | HIFLOAT8                  | HIFLOAT8                  | null        | UINT64/INT64      | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32      |
      | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT32     | FLOAT32           | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      | HIFLOAT8                  | HIFLOAT8                  | FLOAT32     | FLOAT32           | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |

    - In T-T quantization, the shape of x1Scale is (1,) or nullptr, and the shape of x2Scale is (1,).
    - In T-C quantization, the shape of x1Scale is (1,) or nullptr, and the shape of x2Scale is (n,), where n is the same as that of x2.
    - When the data type of x1/x2 is FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, static quantization and dynamic quantization are distinguished. In static quantization, the data type of x2Scale is UINT64 or INT64. In dynamic quantization, the data type of x2Scale is FLOAT32. When the data type of x1/x2 is INT8, dynamic T-C or dynamic T-T quantization is not supported.
    - In dynamic T-C quantization, bias is not supported.

  </details>

  <details>

  <summary>Restrictions on K-C and K-T quantization:</summary>
  <a id="K-C and K-T quantization"></a>

  - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .KC/KT"></a>

      | x1                        | x2                        | x1Scale     | x2Scale     | x2Offset | yScale |   bias    | out                                    |
      | ------------------------- | ------------------------- | ----------- | ----------- | -------- | -------|   ------- | -------------------------------------- |
      | INT8                      | INT8                      | FLOAT32| FLOAT32/BFLOAT16  | null     | null     |  null/INT32/FLOAT32/BFLOAT16   | BFLOAT16              |
      | INT8                      | INT8                      | FLOAT32     | FLOAT32           | null     |  null     | null/INT32/FLOAT32/FLOAT16  | FLOAT16                 |
      | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT32     | FLOAT32           | null     |  null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      | HIFLOAT8                  | HIFLOAT8                  | FLOAT32     | FLOAT32           | null     |  null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |

    - In the K-C quantization scenario, the shape of x1Scale is (m,), and the shape of x2Scale is (n,), where m is the same as that of x1, and n is the same as that of x2.
    - In the K-T quantization scenario, the shape of x1Scale is (m,), and the shape of x2Scale is (1,), where m is the same as that of x1.

  </details>

  <details>

  <summary>Restrictions on G-B quantization and B-B quantization:</summary>
  <a id="G-B quantization and B-B quantization"></a>

  - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .GB/BB"></a>

      | x1                        | x2                        | x1Scale     | x2Scale     |   x2Offset | yScale | bias    | out                                    |
      | ------------------------- | ------------------------- | ----------- | ----------- |   -------- | -------| ------- | -------------------------------------- |
      | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT32     |   FLOAT32           | null     | null     | null | FLOAT16/BFLOAT16/  FLOAT32               |
      | HIFLOAT8                  | HIFLOAT8                  | FLOAT32     |   FLOAT32           | null     | null     | null | FLOAT16/BFLOAT16/  FLOAT32               |
      | INT8                      |INT8                       | FLOAT32     |   FLOAT32           | null     | null     | FLOAT32 |BFLOAT16       |

  - The value relationships between x1, x2, x1Scale, x2Scale, and groupSize are as follows:

    |Quantization Type|x1 shape|x2 shape|x1Scale shape|x2Scale shape|yScale shape|[gsM, gsN, gsK]|groupSize|
    |-------|--------|--------|-------------|-------------|------------|---|---|
    |B-B quantization|<li>Non-transposed: (batch, m, k)</li><li> Transposed: (batch, k, m)</li>|<li>Non-transposed: (batch, k, n)</li><li> Transposed: (batch, n, k)</li>|<li>Non-transposed: (batch, ceil(m / 128), ceil(k / 128))</li><li> Transposed: (batch, ceil(k / 128), ceil(m / 128))</li>|<li>Non-transposed: (batch, ceil(k / 128), ceil(n / 128))</li><li> Transposed: (batch, ceil(n / 128), ceil(k / 128))</li>|null|[128, 128, 128]|549764202624|
    |G-B quantization|<li>Non-transposed: (batch, m, k)</li><li> Transposed: (batch, k, m)</li>|<li>Non-transposed: (batch, k, n)</li><li> Transposed: (batch, n, k)</li>|<li>Non-transposed: (batch, m, ceil(k / 128))</li><li> Transposed: (batch, ceil(k / 128), m)</li>|<li>Non-transposed: (batch, ceil(k / 128), ceil(n / 128))</li><li> Transposed: (batch, ceil(n / 128), ceil(k / 128))</li>|null|[1, 128, 128]|4303356032|

  - Note: In the preceding table, gsM, gsK, and gsN indicate groupSizeM, groupSizeK, and groupSizeN, respectively.
  - In G-B and B-B quantization scenarios, the transpose attributes of x1 and x1Scale must be the same, and the transpose attributes of x2 and x2Scale must be the same.
  - In G-B quantization scenarios, bias is supported only when the input is of type int8.
  - In B-B quantization scenarios, the input is of type int8 and bias is not supported.

  </details>

  <details>

  <summary>Restrictions on the mx quantization scenario:</summary>
  <a id="mx quantization"></a>

  - The input and output support the following data type combinations:
  <a id="The following data type combinations are supported for the input and output .mx"></a>

      |Quantization Type| x1                        | x2                        | x1Scale     | x2Scale     |   x2Offset | yScale | bias    | out                                    |
      |-------| ------------------------- | ------------------------- | ----------- | ----------- |   -------- | -------| ------- | -------------------------------------- |
      |mx full quantization| FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E8M0 |   FLOAT8_E8M0       | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      |mx full quantization| FLOAT4_E2M1                | FLOAT4_E2M1                | FLOAT8_E8M0 |   FLOAT8_E8M0       | null     | null     | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
      |mx fake-quantization| FLOAT8_E4M3FN             | FLOAT4_E2M1               | FLOAT8_E8M0 |   FLOAT8_E8M0       | null     | null     | null/BFLOAT16|  BFLOAT16                               |

  - The value relationships between x1 data type, x2 data type, x1, x2, x1Scale, x2Scale, and groupSize are as follows:

    |Quantization Type|x1 Data Type|x2 Data Type|x1 shape|x2 shape|x1Scale shape|x2Scale shape|bias shape|yScale shape|[gsM, gsN, gsK]|groupSize|
    |-------|--------|--------|--------|--------|-------------|-------------|------------|---------------------------------------|--|--|
    |mx full quantization|FLOAT8_E4M3FN/FLOAT8_E5M2|FLOAT8_E4M3FN/FLOAT8_E5M2|<li>Non-transposed: (batch, m, k)</li><li> Transposed: (batch, k, m)</li>|<li>Non-transposed: (batch, k, n)</li><li> Transposed: (batch, n, k)</li>|<li>Non-transposed: (m, ceil(k / 64), 2)</li><li> Transposed: (ceil(k / 64), m, 2)</li>|<li>Non-transposed: (ceil(k / 64), n, 2)</li><li> Transposed: (n, ceil(k / 64), 2)</li>|(n,) or (batch, 1, n)|null|[1, 1, 32]|4295032864|
    |mx full quantization|FLOAT4_E2M1|FLOAT4_E2M1|(batch, m, k)|(batch, n, k)|(m, ceil(k / 64), 2)|(n, ceil(k / 64), 2)|(n,) or (batch, 1, n)|null|[1, 1, 32]|4295032864|
    |mx fake-quantization|FLOAT8_E4M3FN|FLOAT4_E2M1|(m, k)|(n, k)|(m, ceil(k / 64), 2)|(n, ceil(k / 64), 2)|(1, n)|null|[0, 0, 32]/[1, 1, 32]|32/4295032864|

  - In the mx full quantization scenario, when the data type of x2 is FLOAT8_E4M3FN/FLOAT8_E5M2, the transpose attributes of x1 and x1Scale must be the same, and the transpose attributes of x2 and x2Scale must be the same.
  - In the mx full quantization scenario, when the data type of x2 is FLOAT4_E2M1, only transposeX1 = false and transposeX2 = true are supported. In addition, k must be an even number and ceil(k/32) must be an even number.
  - In the mx fake-quantization scenario, when the data type of x2 is FLOAT4_E2M1, transposeX1 is false and transposeX2 is true, and the batch axis is not supported. The data format can be ND or AI processor affinity format. When the data format is ND, k must be a multiple of 64. When the data format is the AI processor affinity format, both k and n must be multiples of 64.
  - In the mx fake-quantization scenario, bias is an optional parameter. The data type must be BFLOAT16, the data format must be ND, and the shape must be 2D (1, n). If this parameter is not required, pass nullptr.
  - In the mx fake-quantization scenario, the value combinations of [groupSizeM, groupSizeN, groupSizeK] can be [0, 0, 32] and [1, 1, 32], and the corresponding groupSize values are 32 and 4295032864, respectively.

  </details>

  <details>

  <summary>Restrictions on T-CG quantization:</summary>
  <a id="T-CG quantization"></a>

  - The input and output support the following data type combinations:
  <a id="Supported input and output data type combinations:TCG"></a>

      | x1                        | x2                        | x1Scale     | x2Scale     |     x2Offset | yScale | bias    | out                                    |
      | ------------------------- | ------------------------- | ----------- | ----------- |     -------- | -------| ------- | -------------------------------------- |
      | FLOAT8_E4M3FN             | FLOAT4_E2M1               | null        |     BFLOAT16          | null     | INT64/UINT64    | null         |     BFLOAT16                               |

  - The relationship between x1, x2, x1Scale, x2Scale, and groupSize is as follows:

    |Quantization Type|x1 shape|x2 shape|x1Scale shape|x2Scale shape|yScale shape|[gsM, gsN, gsK]|groupSize|
    |-------|--------|--------|-------------|-------------|------------|---------------------------------------|--|
    |T-CG quantization|(m, k)|(n, k)/(k, n)|null|(n, ceil(k / 32))/(ceil(k / 32), n)|(1, n)|[0, 0, 32]/[1, 1, 32]|32/4295032864|

  - In T-CG quantization mode, the yScale data type supports INT64 and UINT64, the data format supports ND, and the shape supports 2D (1, n). If the original input data type does not meet the data type combination in the restrictions, you need to call the aclnn API of the TransQuantParamV2 operator to convert the data type to UINT64 in advance. When the input data type is INT64, the INT64 data is processed as UINT64 internally.
  - In T-CG quantization mode, bias is a reserved parameter and is not supported in the current version. Therefore, nullptr needs to be passed.
  - In T-CG quantization mode, transposeX1 is set to false. The data format can be ND or the AI processor affinity data layout format. When the data format is ND, k must be a multiple of 64 and transposeX2 must be true. When the data format is the AI processor affinity data format, k and n must be multiples of 64 and transposeX2 must be false.
  - In T-CG quantization mode, the value combination of [groupSizeM, groupSizeN, groupSizeK] can be [0, 0, 32] or [1, 1, 32], and the corresponding groupSize values are 32 and 4295032864, respectively.

  </details>

</details>

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- Ascend 950PR/Ascend 950DT:
x1 is of type FLOAT8_E4M3FN, x2 is of type FLOAT4_E2M1, x2Scale is of type BFLOAT16, and yScale is of type UINT64.

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  
  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"
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
      // The mantissa is shifted left by 23 - 7, and the rest is padded with 0s.
      uint32_t fBits = sign | (exponent << 23) | (mantissa << (23 - 7));
      // Forcibly convert to float.
      return *reinterpret_cast<float*>(&fBits);
  }
  
  int AclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> x1Shape = {5, 64};
      std::vector<int64_t> x2Shape = {8, 64};
      std::vector<int64_t> x2ScaleShape = {8, 2};
      std::vector<int64_t> yScaleShape = {1, 8};
      std::vector<int64_t> outShape = {5, 8};
      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* yScaleDeviceAddr = nullptr;
      void* quantParamDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* yScale = nullptr;
      aclTensor* yOffset = nullptr;
      aclTensor* quantParam = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* out = nullptr;
      std::vector<uint8_t> x1HostData(5 * 64, 0b00111000); // 0b00111000 is 1.0 of fp8_e4m3fn.
      std::vector<uint8_t> x2HostData(8 * 64 / 2, 0b0010 + (0b0010 << 4)); // 0b0010 is 1.0 of fp4_e2m1. Here, uint8 is used to represent two fp4s.
      1.0 of std::vector<uint16_t> x2ScaleHostData(8 * 2, 0b0011111110000000); // bf16
      1.0 of std::vector<float> yScaleHostData(8, 1.0); // fp32
      std::vector<uint64_t> quantParamHostData(8, 0);
      std::vector<uint16_t> outHostData(40, 0); // is actually bfloat16.
      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_BF16, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create yScale aclTensor.
      ret = CreateAclTensor(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yScaleTensorPtr(yScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yScaleDeviceAddrPtr(yScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a quantParam aclTensor.
      ret = CreateAclTensor(quantParamHostData, yScaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> quantParamTensorPtr(quantParam, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = true;
      int64_t groupSize = 32;
  
      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
  
      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(yScale, yOffset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceQuantParamAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceQuantParamAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceQuantParamAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceQuantParamAddrPtr.reset(workspaceQuantParamAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceQuantParamAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);
  
      workspaceSize = 0;
      executor = nullptr;
      // Call the first-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5GetWorkspaceSize(
          x1, x2, nullptr, x2Scale, quantParam, nullptr, nullptr, nullptr, bias, transposeX1, transposeX2, groupSize, out,
          &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);
  
      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0); // In C language, fp16 data cannot be directly printed. You need to read the data using uint16 and convert it to fp16 using the binary format.
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
      auto ret = AclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);
  
      Finalize(deviceId, stream);
      return 0;
  }
  ```

- Ascend 950PR/Ascend 950DT:
x1 is of type INT8, x2 is of type INT8, x1Scale is of type FLOAT32, x2Scale is of type FLOAT32, and bias is of type INT32.

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

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

      int64_t
      GetShapeSize(const std::vector<int64_t> &shape)
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

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> x1Shape = {5, 16};
      std::vector<int64_t> x2Shape = {16, 8};
      std::vector<int64_t> biasShape = {8};
      std::vector<int64_t> x2OffsetShape = {8};
      std::vector<int64_t> x1ScaleShape = {5};
      std::vector<int64_t> x2ScaleShape = {8};
      std::vector<int64_t> outShape = {5, 8};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x2OffsetDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(80, 1);
      std::vector<int8_t> x2HostData(128, 1);
      std::vector<int32_t> biasHostData(8, 1);
      std::vector<float> x2ScaleHostData(8, 1);
      std::vector<float> x2OffsetHostData(8, 1);
      std::vector<float> x1ScaleHostData(5, 1);
      std::vector<uint16_t> outHostData(40, 1); // is actually in float16 half-precision mode.
      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
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
      aclOpExecutor *executor = nullptr;
      // Call the first-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, bias,
                                               transposeX1, transposeX2, 0, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

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
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- Ascend 950PR/Ascend 950DT:
**x1** and **x2** are FLOAT8_E4M3FN, **x1Scale** and **x2Scale** are FLOAT32, **x2Offset** is not available, and **bias** is FLOAT32.

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

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

      int64_t
      GetShapeSize(const std::vector<int64_t> &shape)
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

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> x1Shape = {16, 16};
      std::vector<int64_t> x2Shape = {16, 16};
      std::vector<int64_t> biasShape = {16};
      std::vector<int64_t> x2OffsetShape = {16};
      std::vector<int64_t> x1ScaleShape = {1};
      std::vector<int64_t> x2ScaleShape = {1};
      std::vector<int64_t> outShape = {16, 16};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x2OffsetDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(256, 1);
      std::vector<int8_t> x2HostData(256, 1);
      std::vector<int32_t> biasHostData(16, 1);
      std::vector<float> x2ScaleHostData(1, 1);
      std::vector<float> x2OffsetHostData(16, 1);
      std::vector<float> x1ScaleHostData(1, 1);
      std::vector<uint16_t> outHostData(256, 1);  // Half-precision float16

      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
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
      aclOpExecutor *executor = nullptr;
      // Call the first-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, x2Offset, nullptr, bias,
                                               transposeX1, transposeX2, 0, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

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
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
**x1** is INT8, **x2** is INT32, **x1Scale** is FLOAT32, and **x2Scale** is UINT64

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

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

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> x1Shape = {1, 8192};     // (m,k)
      std::vector<int64_t> x2Shape = {8192, 128};  // (k,n)
      std::vector<int64_t> yoffsetShape = {1024};

      std::vector<int64_t> x1ScaleShape = {1,1};
      std::vector<int64_t> x2ScaleShape = {32, 1024}; // x2ScaleShape = [KShape / groupsize, N]
      std::vector<int64_t> outShape = {1, 1024};

      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *yoffsetDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *yoffset = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 1);
      std::vector<int32_t> x2HostData(GetShapeSize(x2Shape), 1);
      std::vector<int32_t> yoffsetHostData(GetShapeSize(yoffsetShape), 1);
      std::vector<int32_t> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
      float tmp = 1;
      uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
      std::vector<int64_t> x2ScaleHostData(GetShapeSize(x2ScaleShape), ans);
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 1);  // Half-precision float16

      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_UINT64, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a yoffset aclTensor.
      ret = CreateAclTensor(yoffsetHostData, yoffsetShape, &yoffsetDeviceAddr, aclDataType::ACL_FLOAT, &yoffset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yoffsetTensorPtr(yoffset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> yoffsetDeviceAddrPtr(yoffsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;
      int64_t groupSize = 256;

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, yoffset, nullptr,
                                              transposeX1, transposeX2, groupSize, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size, 0); // The fp16 data cannot be directly printed in the C language. The data needs to be read by using uint16 and converted into fp16 in binary mode.
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
      int32_t deviceId = 1;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);
      Finalize(deviceId, stream);
      return 0;
  }
  ```
  