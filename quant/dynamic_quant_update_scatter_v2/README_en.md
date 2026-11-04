# DynamicQuantUpdateScatterV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | √ |
| <term>kirin X90 processor series products</term> | √ |
| <term>kirin 9030 processor series products</term> | √ |

## Function

- Description: Automatically fuses the DynamicQuantV2 and ScatterUpdate single operators into the DynamicQuantUpdateScatterV2 fused operator to implement INT4-type asymmetric quantization.
- Process description: the raw data x is first processed by the DynamicQuantV2 operator, which converts it to the INT4 type and outputs the corresponding scaling factor scale and offset. Then, three different ScatterUpdate operations take the insertion index indices, three initial data items (var, var_scale, and var_offset), and the three outputs of the DynamicQuantV2 operator (that is, the quantized data, scale, and offset) as inputs, and perform write-update operations at specified positions. Finally, three in-place updated outputs var, var_scale, and var_offset are obtained, corresponding to the data after quantized update and the corresponding quantization parameters respectively.

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>Quantization input, corresponding to "x" in the process description. Empty tensors are not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Update index of the quantized data, corresponding to "indices" in the process description. Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Input/Output</td>
      <td>Quantization result to be updated, corresponding to "var" in the process description. Empty tensors are not supported.</td>
      <td>INT4</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var_scale</td>
      <td>Input/Output</td>
      <td>Quantization scale factor to be updated, corresponding to "var_scale" in the process description. Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var_offset</td>
      <td>Input/Output</td>
      <td>Quantization offset factor to be updated, corresponding to "var_offset" in the process description. Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

- Asymmetric quantization is supported for the quantization method, and the INT4 quantization data type is supported.
- Quantization does not support the smooth_scale input.
- In the INT4 quantization case, the tail axis of the input x must be divisible by 2.
- The shape of the input x is (B, 1, H), the shape of var is (B, S, 1, H), and the shapes of var_scale and var_offset are (B, S).
- The automatic fusion scenario requires H to be divisible by 8 and H to not exceed 14960.
- output0 of DynamicQuantV2 is of the INT4 type, output1 is of the FLOAT type, and output2 is of the FLOAT type.
- The input dtype of DynamicQuantV2 must be FLOAT16 or BFLOAT16. If input1 exists and input2 does not exist, the shape of input1 must be 1D and equal the last dimension of input0; if input2 exists, input1 is 2D, where the size of the first dimension is the number of experts and does not exceed 1024, and the size of the second dimension equals the last dimension of input0.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_dynamic_quant_update_scatter_v2](./examples/test_geir_dynamic_quant_update_scatter_v2.cpp)   | Invokes the DynamicQuantUpdateScatterV2 operator by building a graph through the [operator IR](./op_graph/dynamic_quant_update_scatter_v2_proto.h). |
