# GroupedQuantMax

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

- Operator function: Perform grouped quantization on input x based on the input scale, and compute the maximum absolute value amax of input x group by group.

- Formula:

  $group\_list$ splits $x$ along dim-0 into num_groups groups, and $group\_list[g]$ is the cumulative size of the first $g+1$ groups along dim-0 (monotonically increasing, $group\_list[-1]=x.shape[0]$). For each group (g = 0, 1, ..., num_groups - 1):

  $$
  x\_g = x[group\_list[g-1] : group\_list[g], ...]
  $$

  $$
  y = \text{Cast}(x\_g \times scale[g], \text{dst\_type}, \text{round\_mode})
  $$

  $$
  amax[g] = \max(|x\_g|)
  $$

  Here $x$ denotes the input tensor, and $x\_g$ denotes the g-th group of data of the input tensor (the lower bound is 0 when g=0). $scale$ denotes the quantization scaling factor, $amax[g]$ denotes the maximum absolute value of the group of the input tensor, $\text{dst\_type}$ specifies the output data type, and $\text{round\_mode}$ specifies the rounding mode.

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>Input tensor to be quantized, corresponding to x in the formula. Any shape of 2D to 8D is supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Quantization scaling factor, corresponding to scale in the formula. The shape is fixed to [num_groups].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>Input</td>
      <td>Offset of each group of input x along axis 0 (cumsum mode), corresponding to group_list in the formula. The shape is fixed to [num_groups].</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Quantized output tensor, corresponding to y in the formula. The shape is the same as that of x, and the data type is determined by dst_type.</td>
      <td>HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>amax</td>
      <td>Output</td>
      <td>Maximum absolute value of x, corresponding to amax in the formula. The shape is fixed to [num_groups], and the data type is the same as that of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Attribute</td>
      <td>The value must be a valid rounding mode supported by the Cast API.</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Specifies the output quantization data type. 34=HIFLOAT8, 35=FLOAT8_E5M2, and 36=FLOAT8_E4M3.</td>
      <td>int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Matching relationship between `round_mode` and `dst_type`:
    - When `dst_type` is `35` or `36`, `round_mode` must be `"rint"`.
    - When `dst_type` is `34`, `round_mode` must be `"round"` or `"hybrid"`.
- The range of num_groups currently supports only 1-384.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_grouped_quant_max](./examples/arch35/test_aclnn_grouped_quant_max.cpp) | Invoke the GroupedQuantMax operator through the [aclnnGroupedQuantMax](./docs/aclnnGroupedQuantMax_en.md) API.    |
| Graph-mode invocation | -   | Invoke the GroupedQuantMax operator through graph construction using the [operator IR](./op_graph/grouped_quant_max_proto.h). |
