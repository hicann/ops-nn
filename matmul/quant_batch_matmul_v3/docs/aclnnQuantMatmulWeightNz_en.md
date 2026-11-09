# aclnnQuantMatmulWeightNz

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs quantized matrix multiplication computation. Similar APIs include aclnnMm (which supports only 2D tensors as inputs for matrix multiplication) and aclnnBatchMatMul (which supports only 3D matrix multiplication, where the first dimension is the batch dimension). Supports T-C, T-T, K-C, K-T, G-B, B-B, and MX [quantization mode](../../../docs/en/context/quant_mode_introduction.md).

- Formula:

  <!-- npu="950,A3,910b,310p" id7 -->
  <details>
  <summary><term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term></summary>

  - Without x1Scale and bias:

    $$
    out = x1@x2 * x2Scale + x2Offset
    $$

  - bias INT32:

    $$
    out = (x1@x2 + bias) * x2Scale + x2Offset
    $$

  - x1Scale without bias

    $$
    out = x1@x2 * x2Scale * x1Scale
    $$

  - With x1Scale, bias INT32.

    $$
    out = (x1@x2 + bias) * x2Scale * x1Scale
    $$

  - With x1Scale, bias FLOAT32

    $$
    out = x1@x2 * x2Scale * x1Scale + bias
    $$

  </details>
  <!-- end id7 -->

  <!-- npu="950,A3,910b" id8 -->
  <details>
  <summary><term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term></summary>

  - bias BFLOAT16/FLOAT32 (no x2Offset in this scenario):

    $$
    out = x1@x2 * x2Scale + bias
    $$

  - x1Scale, no bias:

    $$
    out = x1@x2 * x2Scale * x1Scale
    $$

  - x1Scale, bias INT32 (this scenario has no x2Offset):

    $$
    out = (x1@x2 + bias) * x2Scale * x1Scale
    $$

  - x1Scale, bias BFLOAT16/FLOAT16/FLOAT32 (this scenario has no x2Offset):

    $$
    out = x1@x2 * x2Scale * x1Scale + bias
    $$

  - x1 is INT8, x2 is INT32, x1Scale is FLOAT32, x2Scale is UINT64, and yOffset is FLOAT32:

    $$
    out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale
    $$

  </details>
  <!-- end id8 -->

  <!-- npu="950" id9 -->
  <details>
  <summary><term>Ascend 950PR&950DT products</term></summary>

  - G-B && B-B && MX quantization mode:

    $$
    out[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
    $$

    Here, gsM, gsN, and gsK denote groupSizeM, groupSizeN, and groupSizeK. x1Slice is a vector of length groupSizeK from row m of x1, and x2Slice is a vector of length groupSizeK from column n of x2. Both slices start at j*groupSizeK along K, where j is in [0, kLoops), kLoops = ceil(K / groupSizeK), and K is the reduction dimension. The last slice may be shorter than groupSizeK. Bias is accumulated only when supported by the selected mode and provided; otherwise, that term is 0.
  </details>
  <!-- end id9 -->

In the formulas, `x1` and `x2` are the matrices after applying the transpose attributes. Their last two dimensions are (M,K) and (K,N), and those of the output are (M,N). `@` denotes matrix multiplication, and `*` denotes multiplication after broadcasting along the corresponding dimensions. For UINT64/INT64 quantization parameters, the formulas use the encoded scale values. An omitted optional bias or offset is treated as 0. For supported combinations, see [Constraints](#constraints).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call "aclnnQuantMatmulWeightNzGetWorkspaceSize" first to obtain the workspace size required for the calculation and the executor that includes the calculation process of the operator, and then call "aclnnQuantMatmulWeightNz" to perform the calculation.

```cpp
aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(
  const aclTensor* x1,
  const aclTensor* x2,
  const aclTensor* x1Scale,
  const aclTensor* x2Scale,
  const aclTensor* yScale,
  const aclTensor* x1Offset,
  const aclTensor* x2Offset,
  const aclTensor* yOffset,
  const aclTensor* bias,
  bool             transposeX1,
  bool             transposeX2,
  int64_t          groupSize,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnQuantMatmulWeightNz(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulWeightNzGetWorkspaceSize

- **Parameters**
  <table style="table-layout: fixed; width: 1500px">
    <colgroup>
      <col style="width: 210px">
      <col style="width: 100px">
      <col style="width: 240px">
      <col style="width: 390px">
      <col style="width: 230px">
      <col style="width: 120px">
      <col style="width: 110px">
      <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimensions (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x1 (const aclTensor*)</td>
        <td>Input</td>
        <td>Input x1 in the Formula.</td>
        <td><ul><li>Supports <a href="../../../docs/en/context/non_contiguous_tensor.md">non-contiguous tensors</a>.</li><li>When transposeX1 is false, the shape is (batch, m, k). The batch dimensions are optional.</li><li>When transposeX1 is true, the shape is (batch, k, m). The batch dimensions are optional.</li></ul></td>
        <td>INT4<sup>1, 3</sup>, INT8, HIFLOAT8<sup>1, 2</sup>, FLOAT8_E4M3FN<sup>1, 2</sup>, FLOAT4_E2M1<sup>1, 2</sup></td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2 (const aclTensor*)</td>
        <td>Input</td>
        <td>Input x2 in the Formula.</td>
        <td><ul><li>The logical shape is (batch,K,N) when transposeX2=false and (batch,N,K) when it is true. The batch dimensions are optional.</li><li>Storage must use the NZ layout. For storage shapes and packing requirements, see the product-specific and quantization-mode descriptions below. Physical block axes must not be treated as logical matrix axes.</li></ul></td>
        <td>INT4<sup>1, 3</sup>, INT8, INT32<sup>1, 3</sup>, HIFLOAT8<sup>1, 2</sup>, FLOAT4_E2M1<sup>1, 2</sup>, FLOAT32<sup>1, 2</sup>, FLOAT8_E4M3FN<sup>1, 2</sup></td>
        <td>FRACTAL_NZ</td>
        <td>Logical: 2–6; storage: 4–8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1Scale (const aclTensor*)</td>
        <td>Optional input</td>
        <td>Optional quantization parameter, which is the input x1Scale in the formula.</td>
        <td><ul><li>For FLOAT8_E8M0 x1Scale, the shape is (batch, m, ceil(k / 64), 2) when transposeX1 is false, and (batch, ceil(k / 64), m, 2) when it is true. The batch dimensions must match those of x1.</li><li>For non-grouped quantization, FLOAT32 x1Scale has shape (m,), where m matches x1. For grouped modes such as G-B and B-B, see <a href="#constraints">Constraints</a>.</li><li>Pass nullptr when a left-matrix scale is not used. This parameter is required for modes such as grouped quantization that need it. See <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32<sup>1</sup>, FLOAT8_E8M0<sup>1, 2</sup></td>
        <td>ND</td>
        <td>1–7</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2Scale (const aclTensor*)</td>
        <td>Input</td>
        <td>Represents the quantization parameter, which is the input x2Scale in the formula.</td>
        <td><ul><li>For FLOAT8_E8M0 x2Scale in ND format, the shape is (batch, ceil(k / 64), n, 2) when transposeX2 is false, and (batch, n, ceil(k / 64), 2) when it is true. The batch dimensions must match those of x2.</li><li>For FLOAT8_E8M0 x2Scale in NZ format (ScaleNZ), the logical shape is fixed at (batch, ceil(k / 64), n, 2), and the storage shape is (batch, ceil(n / 16), ceil(k / 64), 16, 2), independently of transposeX2. See <a href="#scalenz-constraints">ScaleNZ Constraints</a>.</li><li>For non-grouped quantization, other x2Scale data types use shape (1,) or (n,), where n matches x2. For grouped modes and T-CG, see <a href="#constraints">Constraints</a>.</li><li>When x1 and x2 are HIFLOAT8 and x2 uses NZ format, x2Scale supports UINT64 and INT64 for static quantization; x1Scale and x2Scale both support FLOAT32 for dynamic quantization.</li><li>If the original input type does not match the combinations in <a href="#constraints">Constraints</a>, first call the ACLNN interface of TransQuantParamV2 to convert scale to INT64 or UINT64.</li></ul></td>
        <td>UINT64, INT64, FLOAT32<sup>1</sup>, BFLOAT16<sup>1</sup>, FLOAT8_E8M0<sup>1, 2</sup>, FLOAT16<sup>1, 2</sup></td>
        <td>ND, FRACTAL_NZ</td>
        <td>1–7 (ND)<br/>3–7 (NZ logical shape)<br/>4–8 (NZ storage shape)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>yScale (const aclTensor*)</td>
        <td>Optional input</td>
        <td>Dequantization scale parameter of output y.</td>
        <td>The shape is two-dimensional (1, n), where n matches x2.<br/>Required for T-CG. For support in other modes, see their constraints. Pass nullptr when unused.</td>
        <td>UINT64<sup>1, 2</sup>, INT64<sup>1, 2</sup></td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>x1Offset (const aclTensor*)</td>
        <td>Optional input</td>
        <td>Reserved parameter.</td>
        <td>Not supported in the current version. nullptr or an empty tensor must be passed in.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Offset (const aclTensor*)</td>
        <td>Optional input</td>
        <td>Optional quantization parameter, the input x2Offset in the formula.</td>
        <td>The shape is one-dimensional (t,), where t = 1 or n and n matches x2.<br/>Pass nullptr when no output offset is used.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>yOffset (const aclTensor*)</td>
        <td>Optional input</td>
        <td>The input yOffset in the formula.</td>
        <td>For A8W4 with INT8 input and INT32 packed weights, provide the offline auxiliary result with shape (N,). For its values, see <a href="#constraints">Constraints</a>. Pass nullptr in other cases.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
      </tr>
      <tr>
        <td>bias (const aclTensor*)</td>
        <td>Optional input</td>
        <td>Optional parameter, the input bias in the formula.</td>
        <td><ul><li>Supports shape (n,) or (batch, 1, n), where n matches x2.</li><li>When out has 2, 4, 5, or 6 dimensions, bias must have shape (n,).</li><li>Pass nullptr when no bias is used.</li></ul></td>
        <td>INT32, BFLOAT16<sup>1</sup>, FLOAT16<sup>1</sup>, FLOAT32<sup>1</sup></td>
        <td>ND</td>
        <td>1, 3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>transposeX1 (bool)</td>
        <td>Input</td>
        <td>Indicates whether the input shape of x1 includes transpose.</td>
        <td>false disables transposition and true enables it. For supported combinations, see <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2 (bool)</td>
        <td>Input</td>
        <td>Indicates whether the input shape of x2 contains transpose.</td>
        <td>false disables transposition and true enables it. For supported combinations, see <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize (int64_t)</td>
        <td>Input</td>
        <td>Specifies the quantization group sizes along the m, n, and k dimensions.</td>
        <td>Composed of the three directional values groupSizeM, groupSizeN, and groupSizeK concatenated, each occupying 16 bits, for a total of the lower 48 bits of the int64_t type groupSize (the upper 16 bits are invalid). For the calculation formula, see Formula 1 below the table. If groupSize is not supported, pass 0.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>The output out in the formula.</td>
        <td><ul> <li>The shape supports 2-6 dimensions, (batch, m, n), and batch may not exist.</li> <li>The batch dimensions of x1 and x2 support broadcasting. The output batch is the same as the batch after broadcasting, m is the same as the m of x1, and n, together with n1 and n0 of x2, satisfies the relationship ceil(n / n0) = n1.</li> </ul></td>
        <td>FLOAT16, INT8, BFLOAT16<sup>1</sup>, INT32<sup>1</sup>, FLOAT32<sup>1, 2</sup></td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to allocate on the device, in bytes.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which includes the calculation process of the operator.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

  - Formula 1:

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
    <!-- npu="310p" id16 -->
    <details>

    <summary><term>Atlas inference products</term></summary>

    - The superscript "1" in the data type column of the table above indicates a data type not supported by this series.
    - x2 does not support [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md).
    - yScale is not supported.
    - groupSize is not supported. Pass 0 for groupSize.
    </details>
    <!-- end id16 -->
    <!-- npu="A3,910b" id17 -->
    <details>

    <summary><term>Atlas A2 products</term>, <term>Atlas A3 products</term></summary>

    - The superscript "2" in the data type column of the table above indicates a data type not supported by this series.
    - x2 does not support [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md).
    - yScale is not supported.
    - groupSize is not supported. Pass 0 for groupSize.
    </details>
    <!-- end id17 -->
    <!-- npu="950" id18 -->
    <details>

    <summary><term>Ascend 950PR&950DT products</term></summary>

    - The superscript "3" in the data type column of the table above indicates a data type not supported by this series.
    - x2 supports [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) with the last two axes transposed; [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) in other use cases are not supported.
    - Passing a non-zero value for groupSize is supported.
    - In the full quantization scenario, when m of input x1 or n of x2 is 0, the output is an empty tensor.
    </details>
    <!-- end id18 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>Return/Status Code</th>
        <th>Value</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x1, x2, x2Scale, or out is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td><ul><li>The data types or formats of x1, x2, bias, x1Scale, x2Scale, x2Offset, or out are outside the supported ranges.</li><li>The shapes of x1, x2, bias, x1Scale, x2Scale, x2Offset, or out fail validation.</li><li>x1, x2, bias, x1Scale, x2Scale, x2Offset, or out is an empty tensor outside the supported empty-tensor cases described above.</li><li>A nonempty tensor is provided for yScale, x1Offset, or yOffset in a quantization mode that does not support it.</li><li>groupSize fails validation.</li></ul></td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>Failed to create the operator executor.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>An internal tensor operation returned a null pointer during tensor creation, preprocessing, operator computation, or result copying.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>The current platform does not support the selected quantized matrix multiplication path.</td>
      </tr>
    </tbody>
  </table>

## aclnnQuantMatmulWeightNz

- **Parameters**

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 220px">
      <col style="width: 100px">
      <col style="width: 680px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace (void*)</td>
        <td>Input</td>
        <td>Workspace memory address applied on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t)</td>
        <td>Input</td>
        <td>Workspace size in bytes to allocate on the device, obtained from the first-stage aclnnQuantMatmulWeightNzGetWorkspaceSize interface.</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor*)</td>
        <td>Input</td>
        <td>op executor, which includes the calculation process of the operator.</td>
      </tr>
      <tr>
        <td>stream (aclrtStream)</td>
        <td>Input</td>
        <td>Specifies the stream for executing tasks.</td>
      </tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation description:
  - aclnnQuantMatmulWeightNz defaults to a deterministic implementation.

<!-- npu="A3,910b" id10 -->
<details>
<summary><term>Atlas A2 products</term>, <term>Atlas A3 products</term></summary>

- The last dimension size of x1 cannot exceed 65535. However, if the dtype of x1 and x2 is INT8, the dtype of x2Scale is FLOAT32 or BFLOAT16, and both x2Scale and x1Scale have a dimension of 1, a size greater than 65535 is supported. The last dimension of x1 refers to m when transposeX1 is true, or k when transposeX1 is false.
- The last dimension size of x2 cannot exceed 65535. However, if the dtype of x1 and x2 is INT8, the dtype of x2Scale is FLOAT32 or BFLOAT16, and both x2Scale and x1Scale have a dimension of 1, a size greater than 65535 is supported. The last dimension of x2 refers to k when transposeX2 is true, or n when transposeX2 is false.
- This interface supports calling [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) before the call to process x2 with the ND format to obtain the AI processor-affinity data layout format.

- When either of the last two dimensions of the original ND is 1, the weightNz feature cannot be used, and this API does not support such a scenario.

- The input and output support the following data type combinations:

    | x1 | x2 | x1Scale | x2Scale | x2Offset | yOffset | bias | out |
    | ---- | ----- | ------------ | ---------------- | ------------- | ------- | --------------------------- | ----------------- |
    | INT8 | INT8 | null | UINT64/INT64 | null | null | null/INT32 | FLOAT16 |
    | INT8 | INT8 | null | UINT64/INT64 | null/FLOAT32 | null | null/INT32 | INT8 |
    | INT8 | INT8 | null/FLOAT32 | FLOAT32/BFLOAT16 | null | null | null/INT32/BFLOAT16/FLOAT32 | BFLOAT16 |
    | INT8 | INT8 | FLOAT32 | FLOAT32 | null | null | null/INT32/FLOAT16/FLOAT32 | FLOAT16 |
    | INT8 | INT8 | null | FLOAT32/BFLOAT16 | null | null | null/INT32 | INT32 |
    | INT4 | INT4 | null/FLOAT32 | BFLOAT16 | null/FLOAT32 | null | null/BFLOAT16 | BFLOAT16 |
    | INT4 | INT4 | null/FLOAT32 | FLOAT32 | null/FLOAT32 | null | null/BFLOAT16 | BFLOAT16 |
    | INT4 | INT4 | null/FLOAT32 | UINT64 | null/FLOAT32 | null | null/INT32 | FLOAT16 |
    | INT4 | INT4 | null/FLOAT32 | FLOAT32 | null/FLOAT32 | null | null/INT32 | FLOAT16 |
    | INT8 | INT32 | FLOAT32 | UINT64 | null | FLOAT32 | null | FLOAT16/BFLOAT16 |

- Constraint on x1: When the data type is INT8 and the data type of x2 is INT32, transposeX1 must be false. Dimensions: (m, k), where k must be an even number and less than 29576.
- Constraint on yOffset: The shape supports 1 dimension (n). It is an auxiliary result computed offline during the computation process, with the value required to be 8\*x2\*x2Scale, and accumulated along the first dimension.

</details>
<!-- end id10 -->

<!-- npu="310p" id11 -->
<details>
<summary><term>Atlas inference products</term></summary>

- The size of the last dimension of x1 cannot exceed 65535. The last dimension of x1 refers to m when transposeX1 is true, or k when transposeX1 is false.
- The size of the last dimension of x2 cannot exceed 65535. The last dimension of x2 refers to k when transposeX2 is true or n when transposeX2 is false.
- This interface supports calling [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) before the call to process x2 with the ND format to obtain the AI processor-affinity data layout format.

- When either of the last two dimensions of the original ND is 1, the weightNz feature cannot be used, and this API does not support such a scenario.

- The input and output support the following data type combinations:

    | x1 | x2 | x1Scale | x2Scale | x2Offset | bias | out |
    | ---- | ---- | ------- | ------------ | ------------- | ------------------ | ------- |
    | INT8 | INT8 | null | UINT64/INT64 | null | null/INT32 | FLOAT16 |
    | INT8 | INT8 | null | UINT64/INT64 | null/FLOAT32 | null/INT32 | INT8 |
    | INT8 | INT8 | FLOAT | FLOAT | null | null/INT32/FLOAT32 | FLOAT16 |

- When x1Scale is not null, only K-C quantization is supported.

</details>
<!-- end id11 -->

<!-- npu="950" id12 -->
<details>
<summary><term>Ascend 950PR&950DT products</term></summary>

- This interface supports calling [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) or [aclnnNpuFormatCast](https://gitcode.com/cann/ops-math/blob/master/conversion/npu_format_cast/docs/aclnnNpuFormatCast.md) before the call to process x2 with the ND format into the NZ format. When used, it must be padded with 0s to prevent dirty data.

- When either of the last two dimensions of the original ND is 1, the weightNz feature cannot be used, and this API does not support such a scenario.

- **T-C Quantization && T-T Quantization Scenario Constraints:**
  <a id="t-c-quantization--t-t-quantization"></a>
  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-tc-tt"></a>

      | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | bias | out |
      | --------------- | ----------- | ----------- | --------------- | ------------ | ------- | -------------------------- | ---------------------- |
      | INT8 | INT8 | null | UINT64/ INT64 | null | null | null/INT32 | FLOAT16/ BFLOAT16 |
      | INT8 | INT8 | null | UINT64/ INT64 | null/FLOAT32 | null | null/INT32 | INT8 |
      | INT8 | INT8 | null | FLOAT32/BFLOAT16 | null | null | null/INT32/FLOAT32/BFLOAT16 | BFLOAT16 |
      | INT8 | INT8 | null | FLOAT32/BFLOAT16 | null | null | null/INT32 | INT32 |
      | HIFLOAT8 | HIFLOAT8 | null | UINT64/INT64 | null | null | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32 |
      | FLOAT8_E4M3FN | FLOAT8_E4M3FN | null | UINT64/INT64 | null | null | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32 |

  - In the T-T quantization scenario, null is passed to x1Scale, and the shape of x2Scale is (1,).
  - In the T-C quantization scenario, x1Scale is passed as null, and the shape of x2Scale is (n,), where n is consistent with n of x2.
  - Dynamic T-C or dynamic T-T quantization is not supported.

- **K-C Quantization && K-T Quantization Scenario Constraints:**
  <a id="k-c-quantization--k-t-quantization"></a>
  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-kc-kt"></a>

      | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | bias | out |
      | -------------------- | -------------------- | ------- | --------------- | -------- | ------- | ------------------------- | ------------------------- |
      | INT8 | INT8 | FLOAT32 | FLOAT32/BFLOAT16 | null | null | null/INT32/FLOAT32/BFLOAT16 | BFLOAT16 |
      | INT8 | INT8 | FLOAT32 | FLOAT32 | null | null | null/INT32/FLOAT32/FLOAT16 | FLOAT16 |
      | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | null | null | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32 |
      | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | null | null | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32 |

  - In the K-C quantization scenario, the shape of x1Scale is (m,), and the shape of x2Scale is (n,), where m matches the m of x1, and n matches the n of x2.
  - In the K-T quantization scenario, the shape of x1Scale is (m,), and the shape of x2Scale is (1,), where m matches the m of x1.
  - When x1 and x2 use FLOAT8_E4M3FN and HIFLOAT8, only transposeX1=false is supported.

- **Constraints for G-B and B-B quantization:**
  <a id="g-b-quantization--b-b-quantization"></a>
  - The input and output support the following data type combinations:
  <a id="input-and-output-support-the-following-data-type-combinations-gbbb"></a>

      | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | bias | out |
      | -------------------- | -------------------- | ------- | --------------- | -------- | ------- | ------------------------- | ------------------------- |
      | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | null | null | null | FLOAT16/BFLOAT16/  FLOAT32 |
      | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | null | null | null | FLOAT16/BFLOAT16/  FLOAT32 |

  - Relationship between x1, x2, x1Scale, x2Scale, and groupSize values:

      | Quantization mode | x1 shape | x2 shape | x1Scale shape | x2Scale shape | yScale shape | [gsM, gsN, gsK] | groupSize |
      | ------- | -------- | -------- | ------------- | ------------- | ------------ | --- | --- |
      | B-B quantization | <ul><li>Not transposed: (batch, m, k)</li></ul> | <ul><li>Not transposed: (batch, k, n)</li><li>Transposed: (batch, n, k)</li></ul> | <ul><li>Not transposed: (batch, ceil(m / 128), ceil(k / 128))</li></ul> | <ul><li>Not transposed: (batch, ceil(k / 128), ceil(n / 128))</li><li>Transposed: (batch, ceil(n / 128), ceil(k / 128))</li></ul> | null | [128, 128, 128] | 549764202624 |
      | G-B quantization | <ul><li>Not transposed: (batch, m, k)</li></ul> | <ul><li>Not transposed: (batch, k, n)</li><li>Transposed: (batch, n, k)</li></ul> | <ul><li>Not transposed: (batch, m, ceil(k / 128))</li></ul> | <ul><li>Not transposed: (batch, ceil(k / 128), ceil(n / 128))</li><li>Transposed: (batch, ceil(n / 128), ceil(k / 128))</li></ul> | null | [1, 128, 128] | 4303356032 |

  - Note: In the table above, gsM, gsK, and gsN denote groupSizeM, groupSizeK, and groupSizeN respectively. Dimensions with gsM, gsK, or gsN set to 0 are automatically inferred. The table above lists the case where automatic inference is not used.
  - In the G-B quantization and B-B quantization scenarios, x1 supports only non-transpose, and the transpose attribute of x2 must be consistent with that of x2Scale (when an axis in the shape has a size of 1 and the case is not a dynamic graph NZ scenario, the transpose attributes of x and scale can be inconsistent).
  - Batch consistency: The B-B quantization scenario does not support batch consistency. Even if the batch consistency switch is enabled, the output cannot be guaranteed to meet batch consistency requirements.

- **Constraints for MX quantization:**
    <a id="mx-quantization"></a>

  - The input and output support the following data type combinations:
      <a id="input-and-output-support-the-following-data-type-combinations-mx"></a>

      | x1 | x2 | x1Scale | x2Scale | x2Offset | yScale | bias | out |
      | --------------- | ------------- | ----------- | ----------- | -------- | ------ | ------------- | --------------------------- |
      | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null | null | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32 |
      | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | null | null | null/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32 |

  - Relationship among the x1 data type, x2 data type, x1, x2, x1Scale, x2Scale, and groupSize:

      | Quantization mode | x1 data type | x2 data type | x1 shape | x2 shape | x1Scale shape | x2Scale shape | bias shape | yScale shape | [groupSizeM, groupSizeN, groupSizeK] | groupSize |
      | ------- | -------- | -------- | -------- | -------- | ------------- | ------------- | ------------ | --------------------------------------- | -- | -- |
      | Full MX quantization | FLOAT8_E4M3FN | FLOAT8_E4M3FN | <ul><li>Not transposed: (batch, m, k)</li><li>Transposed: (batch, k, m)</li></ul> | <ul><li>Not transposed: (batch, k, n)</li><li>Transposed: (batch, n, k)</li></ul> | <ul><li>Not transposed: (batch, m, ceil(k / 64), 2)</li><li>Transposed: (batch, ceil(k / 64), m, 2)</li></ul> | <ul><li>Not transposed: (batch, ceil(k / 64), n, 2)</li><li>Transposed: (batch, n, ceil(k / 64), 2)</li></ul> | (n,) or (batch, 1, n) | null | [1, 1, 32] | 4295032864 |
      | Full MX quantization | FLOAT4_E2M1 | FLOAT4_E2M1 | <ul><li>Not transposed: (batch, m, k)</li></ul> | <ul><li>Not transposed: (batch, k, n)</li><li>Transposed: (batch, n, k)</li></ul> | <ul><li>Not transposed: (batch, m, ceil(k / 64), 2)</li></ul> | <ul><li>Not transposed: (batch, ceil(k / 64), n, 2)</li><li>Transposed: (batch, n, ceil(k / 64), 2)</li></ul> | (n,) or (batch, 1, n) | null | [1, 1, 32] | 4295032864 |

  - The x2Scale shapes in the table above apply to ND format. For full MX quantization with FLOAT8_E4M3FN/FLOAT4_E2M1 inputs, x1 and x1Scale must have matching transpose settings. When x2Scale uses ND format, x2 and x2Scale must also have matching transpose settings. ScaleNZ uses the fixed shape constraints below.
  - In the MX full quantization scenario, when the data types of x1 and x2 are FLOAT4_E2M1, the following conditions must be met:
    - The inner axes of x1 and x2 must be even, and k must be greater than 2.
    - transposeX1 must be false.
    - When transposeX2 is false, n must be greater than 2.
    - When transposeX2 is true, n must be greater than 1.

  - **ScaleNZ constraints:**
      <a id="scalenz-constraints"></a>

    - Before calling this interface, you can use [aclnnNpuFormatCast](https://gitcode.com/cann/ops-math/blob/master/conversion/npu_format_cast/docs/aclnnNpuFormatCast.md) to convert x2Scale from ND to NZ format (ScaleNZ), reducing scale format conversions during matrix multiplication. x2 must use NZ format (WeightNZ). Combining ND x2 with ScaleNZ is not supported.
    - Supports only full MX quantization where x1 and x2 are both FLOAT8_E4M3FN or both FLOAT4_E2M1. x1Scale and x2Scale must be FLOAT8_E8M0, and x1Scale must remain in ND format. Other parameters follow the full MX quantization constraints above.
    - Before format conversion, arrange x2Scale data contiguously as (batch, ceil(k / 64), n, 2), where k and n are the logical matrix dimensions and FLOAT4_E2M1 k counts elements. If the original scale layout is (batch, n, ceil(k / 64), 2), first swap the n and ceil(k / 64) axes and reorder the data; changing only the shape is insufficient. After conversion to ScaleNZ, do not transpose the scale according to transposeX2.
    - The ScaleNZ logical shape is (batch, ceil(k / 64), n, 2), and its storage shape is (batch, ceil(n / 16), ceil(k / 64), 16, 2). batch denotes 0–4 leading dimensions and must match the rank and sizes of the batch dimensions of x2. aclnnNpuFormatCast currently supports only three- or four-dimensional FLOAT8_E8M0 inputs; its constraints also apply when using it for conversion.
    - When constructing x2Scale with aclCreateTensor, set viewDims to the logical shape above, storageDims to the storage shape above, format to ACL_FORMAT_FRACTAL_NZ, and offset to 0. Compute strides by cumulative multiplication of the logical dimensions from the last axis. For example, without batch dimensions, the strides are (2 * n, 2, 1). data must point to data already converted to NZ format; changing the format flag of ND data to NZ is insufficient.
    - The interface checks the ScaleNZ logical shape, storage shape, batch dimensions, strides, and offset. It does not infer data semantics from transpose operations performed before format conversion. The caller must ensure that the scale k and n axes and values correspond to x2.
    - For direct ACLNN invocation, use the two-stage process described here. For ACLNN Graph, complete WeightNZ and ScaleNZ conversion before capture, then capture this interface's computation. Both invocation methods must satisfy the input constraints above.

- **Data type and shape requirements for weight-only quantization:**

    | Quantization mode | x1 dtype | x2 dtype | x1Scale dtype | x2Scale dtype | bias dtype | yScale dtype | out dtype | x1 shape | x2 shape | x1Scale shape | x2Scale shape | bias shape | yScale shape | [groupSizeM, groupSizeN, groupSizeK] |
    | --------------- | ------------ | -------------- | -------------- | ------------- | --------- | -------- | -------- | ---------- | -------------- | ------------ | ---------- | ------------ | --------------------------------------- | ------- |
    | MX quantization | FLOAT8_E4M3FN | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | null/BFLOAT16/FLOAT16 | null | BFLOAT16/FLOAT16 | (m, k) | (n, k) | (m, ceil(k / 64), 2) | (n, ceil(k / 64), 2) | (1, n) | null | [0, 0, 32] / [1, 1, 32] |
    | MX quantization | FLOAT8_E4M3FN | FLOAT32 | FLOAT8_E8M0 | FLOAT8_E8M0 | null/BFLOAT16/FLOAT16 | null | BFLOAT16/FLOAT16 | (m, k) | (n, k/8) | (m, ceil(k / 64), 2) | (n, ceil(k / 64), 2) | (1, n) | null | [0, 0, 32] / [1, 1, 32] |
    | T-CG quantization | FLOAT8_E4M3FN | FLOAT4_E2M1 | null | BFLOAT16/FLOAT16 | null | UINT64/INT64 | BFLOAT16/FLOAT16 | (m, k) | (k, n) | null | (k/32, n) | null | (1, n) | [0, 0, 32] / [1, 1, 32] |
    | T-CG quantization | FLOAT8_E4M3FN | FLOAT32 | null | BFLOAT16/FLOAT16 | null | UINT64/INT64 | BFLOAT16/FLOAT16 | (m, k) | (k, n/8) | null | (k/32, n) | null | (1, n) | [0, 0, 32] / [1, 1, 32] |

  - Constraints:
    - For MX quantization, k and n must be multiples of 8.
    - For T-CG quantization, k must be a multiple of 32 and at least 64, and n must be a multiple of 8.
    - When x1 is FLOAT8_E4M3FN and x2 is FLOAT32, each FLOAT32 storage element of x2 contains eight densely packed FLOAT4_E2M1 elements.
    - For MX quantization, bias must have the same data type as out. For T-CG quantization, x2Scale must have the same data type as out.

</details>
<!-- end id12 -->

## Example

The example code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950" id19 -->
- <term>Ascend 950PR&950DT products</term>:
  In the MM_MX_A8W4 scenario, call `aclnnWeightQuantPreprocess` first to synchronously preprocess weight and weightScale, and then call this interface to perform matrix multiplication.

  ```cpp
  #include <cstdint>
  #include <cstdio>
  #include <cstring>
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnn/opdev/float4_e2m1.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"
  #include "aclnnop/aclnn_weight_quant_preprocess.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  #define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto dim : shape) {
          shapeSize *= dim;
      }
      return shapeSize;
  }

  class AclRuntimeGuard {
  public:
      explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

      ~AclRuntimeGuard()
      {
          if (stream_ != nullptr) {
              aclrtDestroyStream(stream_);
          }
          if (deviceSet_) {
              aclrtResetDevice(deviceId_);
          }
          if (aclInited_) {
              aclFinalize();
          }
      }

      int Init(aclrtStream* stream)
      {
          auto ret = aclInit(nullptr);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
          aclInited_ = true;

          ret = aclrtSetDevice(deviceId_);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
          deviceSet_ = true;

          ret = aclrtCreateStream(stream);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
          stream_ = *stream;
          return ACL_SUCCESS;
      }

  private:
      int32_t deviceId_;
      aclrtStream stream_ = nullptr;
      bool aclInited_ = false;
      bool deviceSet_ = false;
  };

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& viewShape,
                      const std::vector<int64_t>& viewStrides, const std::vector<int64_t>& storageShape,
                      aclDataType dataType, aclFormat format, void** deviceAddr, aclTensor** tensor)
  {
      auto size = hostData.size() * sizeof(T);
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      const int64_t* strides = viewStrides.empty() ? nullptr : viewStrides.data();
      *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides, 0, format, storageShape.data(),
                                storageShape.size(), *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return ACL_SUCCESS;
  }

  std::vector<uint8_t> PackFp4(const std::vector<float>& data)
  {
      std::vector<uint8_t> packedData((data.size() + 1) / 2, 0);
      for (size_t i = 0; i < data.size(); i += 2) {
          uint8_t low = op::Float4E2M1(data[i]).value;
          uint8_t high = i + 1 < data.size() ? op::Float4E2M1(data[i + 1]).value : 0;
          packedData[i / 2] = static_cast<uint8_t>((high << 4) | low);
      }
      return packedData;
  }

  float Bf16ToFloat(uint16_t value)
  {
      uint32_t bits = static_cast<uint32_t>(value) << 16;
      float result;
      std::memcpy(&result, &bits, sizeof(result));
      return result;
  }

  int AclnnQuantMatmulWeightNzA8W4PreprocessTest(int32_t deviceId)
  {
      aclrtStream stream = nullptr;
      AclRuntimeGuard runtime(deviceId);
      auto ret = runtime.Init(&stream);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      const int64_t m = 5;
      const int64_t k = 64;
      const int64_t n = 128;
      const int64_t kGroupSize = 32;
      const int64_t scaleK = CEIL_DIV(k, 64);
      const bool transposeX1 = false;
      const bool transposeX2 = false;

      std::vector<int64_t> x1Shape = {m, k};
      std::vector<int64_t> x1Strides = {k, 1};
      std::vector<int64_t> weightViewShape = {k, n};
      std::vector<int64_t> weightStorageShape = {n, k};
      std::vector<int64_t> weightStrides = {1, k};
      std::vector<int64_t> x1ScaleShape = {m, scaleK, 2};
      std::vector<int64_t> x1ScaleStrides = {scaleK * 2, 2, 1};
      std::vector<int64_t> weightScaleViewShape = {scaleK, n, 2};
      std::vector<int64_t> weightScaleStorageShape = {n, scaleK, 2};
      std::vector<int64_t> weightScaleStrides = {2, scaleK * 2, 1};
      std::vector<int64_t> outWeightStorageShape = {CEIL_DIV(k, 32), CEIL_DIV(n, 16), 16, 32};
      std::vector<int64_t> outShape = {m, n};
      std::vector<int64_t> outStrides = {n, 1};

      std::vector<uint8_t> x1HostData(GetShapeSize(x1Shape), 0b00111000); // FLOAT8_E4M3FN 1.0
      std::vector<float> weightFloatData(GetShapeSize(weightStorageShape), 1.0f);
      std::vector<uint8_t> weightHostData = PackFp4(weightFloatData);
      std::vector<uint8_t> x1ScaleHostData(GetShapeSize(x1ScaleShape), 0b01111111); // FLOAT8_E8M0 1.0
      std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleStorageShape),
                                               0b10000101); // FLOAT8_E8M0 64.0
      std::vector<uint8_t> outWeightHostData(GetShapeSize(outWeightStorageShape) / 2, 0);
      std::vector<uint8_t> outWeightScaleHostData(GetShapeSize(weightScaleStorageShape), 0);
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);

      void* x1DeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* weightScaleDeviceAddr = nullptr;
      void* outWeightDeviceAddr = nullptr;
      void* outWeightScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* weightScale = nullptr;
      aclTensor* outWeight = nullptr;
      aclTensor* outWeightScale = nullptr;
      aclTensor* out = nullptr;

      ret = CreateAclTensor(x1HostData, x1Shape, x1Strides, x1Shape, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, &x1DeviceAddr,
                            &x1);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1Ptr(x1, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(weightHostData, weightViewShape, weightStrides, weightStorageShape, ACL_FLOAT4_E2M1,
                            ACL_FORMAT_ND, &weightDeviceAddr, &weight);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPtr(weight, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, x1ScaleStrides, x1ScaleShape, ACL_FLOAT8_E8M0, ACL_FORMAT_ND,
                            &x1ScaleDeviceAddr, &x1Scale);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScalePtr(x1Scale, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(weightScaleHostData, weightScaleViewShape, weightScaleStrides, weightScaleStorageShape,
                            ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &weightScaleDeviceAddr, &weightScale);
      std::unique_ptr<void, aclError (*)(void*)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightScalePtr(weightScale, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Retain the transpose stride of weight so that QuantMatmul can identify the transposed layout from tensor metadata when transposeX2=false.
      ret = CreateAclTensor(outWeightHostData, weightViewShape, weightStrides, outWeightStorageShape, ACL_FLOAT4_E2M1,
                            ACL_FORMAT_FRACTAL_NZ_C0_32, &outWeightDeviceAddr, &outWeight);
      std::unique_ptr<void, aclError (*)(void*)> outWeightDeviceAddrPtr(outWeightDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightPtr(outWeight, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(outWeightScaleHostData, weightScaleViewShape, weightScaleStrides, weightScaleStorageShape,
                            ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &outWeightScaleDeviceAddr, &outWeightScale);
      std::unique_ptr<void, aclError (*)(void*)> outWeightScaleDeviceAddrPtr(outWeightScaleDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightScalePtr(outWeightScale, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(outHostData, outShape, outStrides, outShape, ACL_BF16, ACL_FORMAT_ND, &outDeviceAddr, &out);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outPtr(out, aclDestroyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      ret = aclnnWeightQuantPreprocessGetWorkspaceSize(weight, weightScale, nullptr, nullptr, ACL_FLOAT8_E4M3FN,
                                                       ACL_FLOAT8_E8M0, kGroupSize, outWeight, outWeightScale, nullptr,
                                                       nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantPreprocessGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);

      void* preprocessWorkspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> preprocessWorkspacePtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&preprocessWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate preprocess workspace failed. ERROR: %d\n", ret); return ret);
          preprocessWorkspacePtr.reset(preprocessWorkspaceAddr);
      }

      ret = aclnnWeightQuantPreprocess(preprocessWorkspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantPreprocess failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after preprocess failed. ERROR: %d\n", ret);
                return ret);

      workspaceSize = 0;
      executor = nullptr;
      // The output of WeightQuantPreprocess already meets the input requirements of QuantMatmul and can be directly used as x2 and x2Scale.
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, outWeight, x1Scale, outWeightScale, nullptr, nullptr, nullptr,
                                                     nullptr, nullptr, transposeX1, transposeX2, kGroupSize, out,
                                                     &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);

      void* matmulWorkspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> matmulWorkspacePtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&matmulWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate matmul workspace failed. ERROR: %d\n", ret); return ret);
          matmulWorkspacePtr.reset(matmulWorkspaceAddr);
      }

      ret = aclnnQuantMatmulWeightNz(matmulWorkspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after matmul failed. ERROR: %d\n", ret);
                return ret);

      std::vector<uint16_t> resultData(GetShapeSize(outShape), 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (size_t i = 0; i < resultData.size(); ++i) {
          LOG_PRINT("result[%zu] is: %.1f\n", i, Bf16ToFloat(resultData[i]));
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // WeightQuantPreprocess currently supports only Ascend 950PR&950DT products.
      int32_t deviceId = 0;
      auto ret = AclnnQuantMatmulWeightNzA8W4PreprocessTest(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantMatmulWeightNzA8W4PreprocessTest failed. ERROR: %d\n", ret);
                return ret);
      return 0;
  }
  ```

<!-- end id19 -->

<!-- npu="A3,910b" id13 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

  The following is sample code for the scenario where x2 is in NZ format (transposeX2=false).

  ```cpp
  #include <cstdint>
  #include <cstdio>
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
      // Boilerplate: initialize resources.
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
      // Call aclrtMemcpy to copy data from the host to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
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
      // Call aclrtMemcpy to copy data from the host to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. The construction must be customized based on the API.
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
      std::vector<uint16_t> outHostData(5 * 32, 1);  // Actually float16 half-precision mode.
      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor in NZ format
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
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
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. Call the CANN operator library API, which needs to be changed to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // Call the first-phase API of aclnnTransMatmulWeight
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransMatmulWeight
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // For scale of the FLOAT data type, the aclnn API of the TransQuantParamV2 operator must be called in advance
      // Call the first-phase API of aclnnTransQuantParamV2
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulWeightNz.
      workspaceSize = 0;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, x2, nullptr, quantParam, nullptr, nullptr, nullptr, nullptr,
                                                    bias, transposeX1, transposeX2, 0, out, &workspaceSize,
                                                    &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrNZ(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrNZ.reset(workspaceAddr);
      }
      // Call the second-phase API aclnnQuantMatmulWeightNz.
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, copy the result from device memory to the host, and modify it based on the definition of the specific API.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // uint16_t stores raw FP16 bits. The code below prints the integer bit patterns; decode them to float to print floating-point values.
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
      // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream = nullptr;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- end id13 -->
<!-- npu="310p" id14 -->
- <term>Atlas inference products</term>:
  The following is the sample code for the scenario where x2 is in NZ format (transposeX2=true).

  ```cpp
  #include <cstdint>
  #include <cstdio>
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
      // Boilerplate: initialize resources.
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
      // Call aclrtMemcpy to copy data from the host to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
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
      // Call aclrtMemcpy to copy data from the host to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. The construction must be customized based on the API.
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
      std::vector<uint16_t> outHostData(5 * 32, 1);  // Actually float16 half-precision mode.
      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor in NZ format
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2Transposed aclTensor in NZ format.
      ret = CreateAclTensorX2(x2TransposedHostData, x2TransposedShape, &x2TransposedDeviceAddr,
                              aclDataType::ACL_INT8, &x2Transposed);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TransposedHPTensorPtr(x2Transposed,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2TransposedHPDeviceAddrPtr(x2TransposedDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
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
      bool transposeX1 = false;
      bool transposeX2 = true;

      // 3. Call the CANN operator library API, which needs to be changed to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // The shape of x2 needs to be transposed into the nk format before transdata is performed.
      std::vector<int64_t> dimsData = {1, 0};
      // Create the dims aclIntArray.
      aclIntArray *dims = aclCreateIntArray(dimsData.data(), dimsData.size());
      // Call the first-phase API of aclnnPermute.
      ret = aclnnPermuteGetWorkspaceSize(x2, dims, x2Transposed, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrPermute(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrPermute.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnPermute
      ret = aclnnPermute(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermute failed. ERROR: %d\n", ret); return ret);

      workspaceSize = 0;
      // Call the first-phase API of aclnnTransMatmulWeight
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2Transposed, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransMatmulWeight
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // For scale of the FLOAT data type, the aclnn API of the TransQuantParamV2 operator must be called in advance
      // Call the first-phase API of aclnnTransQuantParamV2
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulWeightNz.
      workspaceSize = 0;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, x2Transposed, nullptr, quantParam, nullptr, nullptr,
                                                    nullptr, nullptr, bias, transposeX1, transposeX2, 0, out,
                                                    &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrNZ(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrNZ.reset(workspaceAddr);
      }
      // Call the second-phase API aclnnQuantMatmulWeightNz.
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, copy the result from device memory to the host, and modify it based on the definition of the specific API.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // uint16_t stores raw FP16 bits. The code below prints the integer bit patterns; decode them to float to print floating-point values.
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
      // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream = nullptr;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950PR&950DT products</term>:
  The following is the sample code for the scenario where x2 is in NZ format (transposeX2=true).

  ```cpp
  #include <cstdint>
  #include <cstdio>
  #include <cstring>
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_npu_format_cast.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"
  #include "aclnn/opdev/float4_e2m1.h"

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
      // Boilerplate: initialize resources.
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
                      aclDataType dataType, aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from the host to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
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
      // Shift the mantissa to the left by 23 - 7, and pad the remaining bits with 0s.
      uint32_t fBits = sign | (exponent << 23) | (mantissa << (23 - 7));
      // Use memcpy to avoid strict aliasing violations.
      float result;
      std::memcpy(&result, &fBits, sizeof(float));
      return result;
  }

  uint64_t GetStorageTensorSize(const int64_t* storageShape, uint64_t storageShapeSize, aclDataType dataType)
  {
      uint64_t elementCount = 1;
      for (uint64_t i = 0; i < storageShapeSize; ++i) {
          elementCount *= static_cast<uint64_t>(storageShape[i]);
      }
      if (dataType == aclDataType::ACL_FLOAT) {
          return elementCount * sizeof(float);
      }
      if (dataType == aclDataType::ACL_FLOAT4_E2M1) {
          return (elementCount + 1) / 2;
      }
      return 0;
  }

  int CreateAclTensorFp4(const std::vector<float>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                         aclTensor** tensor)
  {
      auto elementCount = GetShapeSize(shape);
      auto size = (elementCount + 1) / 2;
      std::vector<uint8_t> packedData(size, 0);
      for (int64_t i = 0; i < elementCount; i += 2) {
          uint8_t low = op::Float4E2M1(hostData[i]).value;
          uint8_t high = (i + 1 < elementCount) ? op::Float4E2M1(hostData[i + 1]).value : 0;
          packedData[i / 2] = (high << 4) | low;
      }
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      ret = aclrtMemcpy(*deviceAddr, size, packedData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT4_E2M1, strides.data(), 0,
                                aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  int CreateAclTensorWithFormat(const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize,
                                void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format)
  {
      auto size = GetStorageTensorSize(*storageShape, *storageShapeSize, dataType);
      CHECK_RET(size > 0, LOG_PRINT("unsupported data type. ERROR: %d\n", static_cast<int>(dataType));
                return ACL_ERROR_INVALID_PARAM);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      std::vector<uint8_t> hostData(size, 0);
      // Call aclrtMemcpy to copy data from the host to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, *storageShape,
                                *storageShapeSize, *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. The construction must be customized based on the API.
      int64_t m = 5;
      int64_t k = 64;
      int64_t n = 128;
      bool transposeX1 = false;
      bool transposeX2 = true;
      int64_t groupSize = 32;
      int64_t numGroups = (k + groupSize - 1) / groupSize;
      int64_t scaleMidDim = (numGroups + 1) / 2;
      std::vector<int64_t> x1Shape = {m, k};
      std::vector<int64_t> x2Shape = {n, k};
      std::vector<int64_t> x1ScaleShape = {m, scaleMidDim, 2};
      std::vector<int64_t> x2ScaleShape = {n, scaleMidDim, 2};
      std::vector<int64_t> outShape = {m, n};
      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
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
      std::vector<uint8_t> x1HostData(m * k, 0b00111000);                    // 1.0 of float8_e4m3
      std::vector<float> x2HostData(n * k, 1);                               // The input is fp32, which is packed into fp4 and then converted to Nz
      std::vector<uint8_t> x1ScaleHostData(m * scaleMidDim * 2, 0b01111111); // 1.0 of float8_e8m0
      std::vector<uint8_t> x2ScaleHostData(n * scaleMidDim * 2,
                                           0b10000101); // 1.0 * 64 of float8_e8m0. According to the reference document, the input needs to be scaled up by 64 times.
      std::vector<uint16_t> outHostData(m * n, 0);      // It is actually bfloat16.
      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      aclTensor* x2NzFp4 = nullptr;
      int actualFormat;

      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2 aclTensor (directly constructed in fp4_e2m1 ND format).
      ret = CreateAclTensorFp4(x2HostData, x2Shape, &x2DeviceAddr, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, which needs to be changed to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // x2 is already in fp4_e2m1 ND format. Use NpuFormatCast to convert it to fp4 NZ format.
      // Use ACL_FLOAT8_E4M3FN to infer C0=32, which is consistent with the Ascend-affine format of fp4.
      aclDataType additionalDtype = aclDataType::ACL_FLOAT8_E4M3FN;
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(x2, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
                return ret);

      // Create the output tensor in fp4 NZ format.
      ret = CreateAclTensorWithFormat(x2Shape, &dstShape, &dstShapeSize, &x2NzFp4DeviceAddr, aclDataType::ACL_FLOAT4_E2M1,
                                      &x2NzFp4, static_cast<aclFormat>(actualFormat));
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2NzFp4TensorPtr(x2NzFp4, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2NzFp4DeviceAddrPtr(x2NzFp4DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // Call aclnnNpuFormatCast to directly convert x2 from fp32 ND to fp4 NZ.
      ret = aclnnNpuFormatCastGetWorkspaceSize(x2, x2NzFp4, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceNzAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceNzAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceNzAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceNzAddrPtr.reset(workspaceNzAddr);
      }

      // Call the second-phase API of aclnnNpuFormatCast.
      ret = aclnnNpuFormatCast(workspaceNzAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulWeightNz.
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, x2NzFp4, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr,
                                                     bias, transposeX1, transposeX2, groupSize, out, &workspaceSize,
                                                     &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API aclnnQuantMatmulWeightNz.
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, copy the result from device memory to the host, and modify it based on the definition of the specific API.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size,
                                       0); // uint16_t stores raw BF16 bits. The code below converts them to float with Bf16ToFloat before printing.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %.1f\n", i, Bf16ToFloat(resultData[i]));
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream = nullptr;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- end id15 -->
