# DualLevelQuantBatchMatmul

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: completes the two-level quantized mxfp4 matrix multiplication computation.
- Formula:
    - x1 and x2 are FLOAT4_E2M1, x1Level0Scale and x2Level0Scale are FLOAT32, x1Level1Scale and x2Level1Scale are FLOAT8_E8M0, out is FLOAT16/BFLOAT16, and bias is an optional parameter of type FLOAT32:

      $$
      out =\sum_{i}^{level0GroupSize} x1Level0Scale @ x2Level0Scale \sum_{ij}^{level1GroupSize} ((x1Level1Scale @ x1_{ij})@ (x2Level1Scale @ x2_{ij})) + bias
      $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
<col style="width: 150px">
<col style="width: 150px">
<col style="width: 430px">
<col style="width: 212px">
<col style="width: 144px">
</colgroup>
<thead>
  <tr>
    <th>Parameter name</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x1</td>
    <td>Input</td>
    <td>The left matrix in the matrix multiplication operation.</td>
    <td>FLOAT4_E2M1</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>Input</td>
    <td>The right matrix in the matrix multiplication operation.</td>
    <td>FLOAT4_E2M1</td>
    <td>FRACTAL_NZ</td>
  </tr>
  <tr>
    <td>x1_level0_scale</td>
    <td>Input</td>
    <td>In the matrix multiplication computation, the scaling factor of the level-1 quantization parameter of x1, corresponding to x1Level0Scale in the formula.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x1_level1_scale</td>
    <td>Input</td>
    <td>In the matrix multiplication computation, the scaling factor of the level-2 quantization parameter of x1, corresponding to x1Level1Scale in the formula.</td>
    <td>FLOAT8_E8M0</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2_level0_scale</td>
    <td>Input</td>
    <td>In the matrix multiplication computation, the scaling factor of the level-1 quantization parameter of x2, corresponding to x2Level0Scale in the formula.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2_level1_scale</td>
    <td>Input</td>
    <td>In the matrix multiplication computation, the scaling factor of the level-2 quantization parameter of x2, corresponding to x2Level1Scale in the formula.</td>
    <td>FLOAT8_E8M0</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Input</td>
    <td>The bias accumulated after the matrix multiplication operation, corresponding to bias in the formula.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>The computation result of the matrix multiplication operation.</td>
    <td>BFLOAT16, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dtype</td>
    <td>Attribute</td>
    <td>The output type of the matrix multiplication computation result.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transpose_x1</td>
    <td>Attribute</td>
    <td>Whether the left matrix x1 in the matrix multiplication operation is transposed</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transpose_x2</td>
    <td>Attribute</td>
    <td>Whether the right matrix x2 in the matrix multiplication operation is transposed.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>level0_group_size</td>
    <td>Attribute</td>
    <td>The size of the level-1 quantization groupsize, corresponding to level0GroupSize in the formula</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>level1_group_size</td>
    <td>Attribute</td>
    <td>The size of the level-2 quantization groupsize, corresponding to level1GroupSize in the formula</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- Empty tensors are not supported.
- Contiguous tensors are supported; [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_dual_level_quant_matmul_weight_nz](examples/arch35/test_aclnn_dual_level_quant_matmul_weight_nz.cpp) | Calls the DualLevelQuantBatchMatmul operator through the [aclnnDualLevelQuantMatmulWeightNz](docs/aclnnDualLevelQuantMatmulWeightNz_en.md) API. |
