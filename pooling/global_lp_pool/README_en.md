# GlobalLpPool

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|

## Function

- Operator function: performs the global Lp pooling computation on the spatial dimensions of the input tensor, with the formula `y = (sum(|x|^p))^(1/p)`, keeping the spatial dimensions as 1 (keepdims=True). Supports 4D(NCHW) and 5D(NCD0D1D2) inputs.

## Parameters

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor for the global_lp_pool computation, 4D(NCHW) or 5D(NCD0D1D2).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>Attribute</td>
      <td>Exponent p of the Lp norm. The default value is 2.0 (that is, the L2 norm).</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Global Lp pooling result. The spatial dimensions are reduced to 1, with the same dtype as input x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The input tensor supports only the 4D(NCHW) or 5D(NCD0D1D2) dimensions.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode interface | [test_geir_global_lp_pool](examples/arch35/test_geir_global_lp_pool.cpp) | Invokes the GlobalLpPool operator through graph construction with the [operator IR](./op_graph/global_lp_pool_proto.h). |
