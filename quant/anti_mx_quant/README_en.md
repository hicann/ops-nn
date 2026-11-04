# AntiMxQuant

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

- Operator function: Dequantizes the FLOAT4/FLOAT8 tensor obtained by quantization through `DynamicMxQuant` into the FLOAT16/BFLOAT16/FLOAT32 format.

- Dequantization formula:

  $$
  X_{dq} = X_q \times 2^{sf - bias}
  $$

  - Here, $sf$ is the scaling factor provided by the input mxscale. $bias$ is the offset of the exponent bits. For the FLOAT8_E8M0 format, $bias=127$. $X_q$ is the quantized FLOAT4/FLOAT8 tensor, and $X_{dq}$ is the dequantized FLOAT16/BFLOAT16/FLOAT32 tensor.

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
      <td>Data to be dequantized. It is the FLOAT4/FLOAT8 tensor obtained through DynamicMxQuant quantization.</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>Input</td>
      <td>Quantization scale obtained through DynamicMxQuant computation.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Specifies the dequantization axis.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Specifies the type of the output y after dequantization.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Corresponding result of the input x after dequantization. Its shape is consistent with x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The shape constraints of x and mxscale are described as follows:
  - If the data type of the input x is float4_e2m1 or float4_e1m2, x.shape[-1] must be an even number.
  - axis_change = axis if axis >= 0 else axis + rank(x).
  - mxscale.shape[axis_change] = (ceil(x.shape[axis] / 32) + 2 - 1) / 2.
  - mxscale.shape[-1] = 2.
  - rank(mxscale) = rank(x) + 1.
  - The other dimensions are consistent with the input x.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_anti_mx_quant](./examples/arch35/test_aclnn_anti_mx_quant.cpp) | Calls the AntiMxQuant operator through the [aclnnAntiMxQuant](./docs/aclnnAntiMxQuant_en.md) interface. |
