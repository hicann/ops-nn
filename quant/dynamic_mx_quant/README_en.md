# DynamicMxQuant

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: along the given axis, the operator computes the quantization scale mxscale corresponding to each group of blocksize elements, divides each element of the group by mxscale, converts the result to the corresponding dst_type according to round_mode, and obtains the quantized result y. When dst_type is FLOAT8_E4M3FN or FLOAT8_E5M2, the value of scale_alg specifies the different algorithms for computing mxscale.

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
      <td>Data to be quantized.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>Axis where the quantization occurs</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional attribute</td>
      <td>Mode of the data conversion</td>
      <td>STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Input</td>
      <td>Specifies the type of y after the data conversion</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blocksize</td>
      <td>Input</td>
      <td>Number of elements per quantization</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>Input</td>
      <td>Computation method of mxscale</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Corresponding result after the input x is quantized</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>Output</td>
      <td>Quantization scale corresponding to each group</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The shape constraints of x and mxscale are as follows:
  - rank(mxscale) = rank(x) + 1.
  - axis_change = axis if axis >= 0 else axis + rank(x).
  - mxscale.shape[axis_change] = (ceil(x.shape[axis] / blocksize) + 2 - 1) / 2.
  - mxscale.shape[-1] = 2.
  - The other dimensions are consistent with the input x.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_dynamic_mx_quant](./examples/arch35/test_aclnn_dynamic_mx_quant.cpp) | Invokes the DynamicMxQuant operator through the [aclnnDynamicMxQuant](./docs/aclnnDynamicMxQuant_en.md) API. |
