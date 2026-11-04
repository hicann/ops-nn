# WeightQuantBatchMatmulV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|

## Function

- **Operator function**: Performs a matrix multiplication computation where the input is in the pseudo-quantization scenario, and can perform quantization computation on the output.
- **Formula**:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  $weight$ in the formula is the input in the pseudo-quantization scenario, and its dequantization formula $ANTIQUANT(weight)$ is

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  When the output needs to be quantized, the quantization formula is

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  When the output does not need to be quantized, the computation formula is

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

## Parameters

<table class="tg"><thead>
  <tr>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Parameter name</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Input/Output/Attribute</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Description</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data type</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data format</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Left matrix in the matrix multiplication operation.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT16, BF16</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">weight</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Right matrix in the matrix multiplication operation.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">INT8, INT4, INT32, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND, FRACTAL_NZ</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">antiquant_scale</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Scale factor in the dequantization parameters, corresponding to antiquantScale in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT16, BF16, UINT64, INT64, FLOAT8_E8M0</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">antiquant_offset</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Offset factor in the dequantization parameters, corresponding to antiquantOffset in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT16, BF16, INT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">quant_scale</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Scale factor in the quantization parameters, corresponding to quantScale in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT32, UINT64</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">quant_offset</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Offset factor in the quantization parameters, corresponding to quantOffset in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">bias</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Bias accumulated after the matrix multiplication operation, corresponding to bias in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT16, FLOAT32, BF16</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">y</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Output</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Computation result of the matrix multiplication operation.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT16, BF16, INT8</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
</tbody></table>

- Atlas A2 products:
  - weight supports only INT8, INT4, and INT32.
  - antiquant_scale supports only FLOAT16, BF16, UINT64, and INT64.
- Atlas A3 products:
  - weight supports only INT8, INT4, and INT32.
  - antiquant_scale supports only FLOAT16, BF16, UINT64, and INT64.
- Ascend 950PR&950DT products: quant_scale and quant_offset are not supported for now.

## Constraints

- Empty tensors are not supported.
- Contiguous tensors are supported, and [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_weight_quant_batch_matmul_v2](examples/test_aclnn_weight_quant_batch_matmul_v2.cpp) | Calls the WeightQuantBatchMatmulV2 operator through APIs such as<br>[aclnnWeightQuantBatchMatmulV2](docs/aclnnWeightQuantBatchMatmulV2_en.md)<br>[aclnnWeightQuantBatchMatmulV3](docs/aclnnWeightQuantBatchMatmulV3_en.md)<br>[aclnnWeightQuantBatchMatmulNz](docs/aclnnWeightQuantBatchMatmulNz_en.md)<br>and so on.|
