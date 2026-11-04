# Sparse4to2QuantMatmul

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Function

- Operator function: Perform sparse 4:2 quantized matrix multiplication.
- Formula:

  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
      $$
      out = x@sparseWeight * sparseWeightScale * xScale + bias
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
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Left matrix in the matrix multiplication, corresponding to x in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">INT8</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">sparseWeight</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Sparse right matrix in the matrix multiplication. It is compressed and corresponds to sparseWeight in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">INT8</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FRACTAL_NZ</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">index</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Index matrix generated during the compression of the right matrix in the matrix multiplication.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">UINT8</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">xScale</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Dequantization parameter corresponding to input x, corresponding to xScale in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">sparseWeightScale</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Dequantization parameter corresponding to input sparseWeight, corresponding to sparseWeightScale in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">bias</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Bias accumulated after the matrix multiplication, corresponding to bias in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">BFLOAT16</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">y</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Output</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Computation result of the matrix multiplication.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">BFLOAT16</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
</tbody></table>

## Constraints

- Empty tensors are not supported.
- Non-contiguous tensors are not supported.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_sparse4to2quant_matmul](examples/test_aclnn_sparse4to2quant_matmul.cpp) | Invoke the Sparse4to2QuantMatmul operator through<br>[aclnnSparse4to2QuantMatmulWeightNz](docs/aclnnSparse4to2QuantMatmulWeightNz_en.md)<br>and other APIs. |
