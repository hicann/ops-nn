# BatchMatMulV3

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|√|
|Atlas training products|√|
|Kirin X90 processor series products|√|
|Kirin 9030 processor series products|√|

## Function

- Operator function: performs matrix multiplication with a batch.
- Formula:

  $$
  out = self@mat2 + bias
  $$

  Here, the $self$ and $mat2$ input dimensions support 2 to 6 dimensions, and matrix multiplication is computed on the last two dimensions. For example, when the input dimensions of $self$ and $mat2$ are $(B, M, K)$ and $(B, K, N)$ respectively, the dimension of $out$ is $(B, M, N)$. $bias$ is a vector with the dimension $(B, 1, N)$. Here, B is the batch count of the matrix multiplication, M is the number of rows of the first tensor $self$ and of the output tensor $out$, N is the number of columns of the second tensor $mat2$ and of the output tensor $out$, and K is the number of columns of the first tensor $self$ and the number of rows of the second tensor $mat2$.

## Parameters

<table class="tg" style="undefined;table-layout: fixed; width: 885px"><colgroup>
<col style="width: 83px">
<col style="width: 126px">
<col style="width: 302px">
<col style="width: 226px">
<col style="width: 148px">
</colgroup>
<thead>
  <tr>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Parameter name</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Input/Output/Attribute</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Description</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data type</span></th>
    <th class="tg-85j1"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data format</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">self</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Left matrix in the matrix multiplication.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT16, BFLOAT16, FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND, FRACTAL_NZ</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">mat2</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Right matrix in the matrix multiplication.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT16, BFLOAT16, FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND, FRACTAL_NZ</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">bias</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Bias accumulated after the matrix multiplication, corresponding to bias in the formula.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT16, BFLOAT16, FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">out</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Output</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Computation result of the general matrix multiplication.</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT16, BFLOAT16, FLOAT32</span></td>
    <td class="tg-22a9"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND, FRACTAL_NZ</span></td>
  </tr>
</tbody></table>

- Atlas A2 products and Atlas A3 products: bias does not support the BFLOAT16 data format.
- Atlas inference products: only the FLOAT16 data format is supported, and the input mat2 supports only the FRACTAL_NZ format.
- Ascend 950PR&950DT products: only the input mat2 supports the FRACTAL_NZ format.
- Kirin X90/Kirin 9030 processor series products: BFLOAT16 and FLOAT32 are not supported.

## Constraints

- Empty tensors are supported, and bias is not supported in the empty tensor scenario.
- Contiguous tensors are supported, and [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_batchmatmul](examples/test_aclnn_batchmatmul.cpp) | Invokes the BatchMatMulV3 operator through<br>[aclnnAddbmm&aclnnInplaceAddbmm](./docs/aclnnAddbmm&aclnnInplaceAddbmm_en.md)<br>[aclnnBaddbmm&aclnnInplaceBaddbmm](./docs/aclnnBaddbmm&aclnnInplaceBaddbmm_en.md)<br>[aclnnBatchMatMul](docs/aclnnBatchMatMul_en.md)<br>[aclnnBatchMatMulWeightNz](docs/aclnnBatchMatMulWeightNz_en.md)<br>and other methods. |
