# QuantBatchMatmulV3

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|x|
|Atlas inference products|√|
|Atlas training products|x|
|Kirin X90 processor series products|√|
|Kirin 9030 processor series products|√|

## Function

- Operator function: Performs quantized matrix multiplication. The input supports a minimum of 2 dimensions and a maximum of 6 dimensions.
- Formula:

  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
    - No pertoken and no bias:

      $$
      out = x1@x2 * scale + offset
      $$

    - bias INT32:

      $$
      out = (x1@x2 + bias) * scale + offset
      $$

    - bias BFLOAT16/FLOAT32 (no offset in this scenario):

      $$
      out = x1@x2 * scale + bias
      $$

    - pertoken and no bias:

      $$
      out = x1@x2 * scale * pertokenScaleOptional
      $$

    - pertoken and bias INT32 (no offset in this scenario):

      $$
      out = (x1@x2 + bias) * scale * pertokenScaleOptional
      $$

    - pertoken and bias BFLOAT16/FLOAT16/FLOAT32 (no offset in this scenario):

      $$
      out = x1@x2 * scale * pertokenScaleOptional + bias
      $$

  - <term>Atlas inference products</term>:
    - No pertoken and no bias:

      $$
      out = x1@x2 * scale + offset
      $$

    - No pertoken and bias INT32:

      $$
      out = (x1@x2 + bias) * scale + offset
      $$

    - pertoken and no bias

      $$
      out = x1@x2 * scale * pertokenScaleOptional
      $$

    - pertoken and bias INT32

      $$
      out = (x1@x2 + bias) * scale * pertokenScaleOptional
      $$

    - pertoken and bias FLOAT32

      $$
      out = x1@x2 * scale * pertokenScaleOptional + bias
      $$

## Parameters

<table class="tg"><thead>
  <tr>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Parameter name</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Input/Output/Attribute</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Description</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data type</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data format</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x1</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Left matrix in the matrix multiplication operation.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND, FRACTAL_NZ</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x2</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Right matrix in the matrix multiplication operation.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND, FRACTAL_NZ</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">scale</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Scale factor of the quantization parameter, corresponding to scale in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">UINT64, FLOAT32, INT64, BF16, FLOAT8_E8M0</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">offset</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Offset factor of the quantization parameter, corresponding to offset in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">bias</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Bias accumulated after the matrix multiplication operation, corresponding to bias in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">INT32, BF16, FLOAT16, FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">pertoken_scale</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Scale factor of the quantization parameter, corresponding to pertokenScaleOptional in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT32, FLOAT8_E8M0</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">y</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Output</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Computation result of the matrix multiplication operation.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT16, INT8, BF16, INT32, FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
</tbody></table>

- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - x1 supports only the INT8 and INT4 data types.
  - x2 supports only the INT8 and INT4 data types.
  - scale supports only the UINT64, FLOAT32, INT64, and BF16 data types.
  - bias supports only the INT32, BFLOAT16, FLOAT16, and FLOAT32 data types.
  - offset supports only the FLOAT32 data type.
  - pertoken_scale supports only the FLOAT32 data type.
  - y supports only the FLOAT16 and BFLOAT16 data types.
- <term>Atlas inference products</term>:
  - x1 supports only the INT8 data type.
  - x2 supports only the INT8 data type.
  - scale supports only the UINT64 and FLOAT32 data types.
  - bias supports only the INT32 data type, and the pertoken scenario additionally supports the FLOAT32 data type.
  - pertoken_scale supports only the FLOAT32 data type.
  - y supports only the FLOAT16 and INT32 data types.
- Kirin X90/Kirin 9030 processor series products:
  - x1 and x2 support only the INT8 data type.
  - scale supports only the UINT64 and INT64 data types.
  - offset supports only the FLOAT32 data type.
  - bias supports only the INT32 data type.
  - pertoken_scale supports only the FLOAT32 data type.
  - y supports only the FLOAT16 and INT8 data types.

## Constraints

- Empty tensors are not supported.
- Contiguous tensors are supported. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_quant_matmul_v3](examples/test_aclnn_quant_matmul_v3_at.cpp) | Calls the QuantBatchMatmulV3 operator through <br>[aclnnQuantMatmulV3](docs/aclnnQuantMatmulV3_en.md)<br>[aclnnQuantMatmulV4](docs/aclnnQuantMatmulV4_en.md)<br>[aclnnQuantMatmulWeightNz](docs/aclnnQuantMatmulWeightNz_en.md)<br>[aclnnQuantMatmulV5](../quant_batch_matmul_v4/docs/aclnnQuantMatmulV5_en.md)<br>and other interfaces. |
| GEIR interface | [test_geir_quant_batch_matmul_v3](examples/test_geir_quant_batch_matmul_v3.cpp) | Builds the graph and invokes the operator through the [QuantBatchMatmulV3 operator IR](docs/ascendir_QuantBatchMatmulV3.md). |
