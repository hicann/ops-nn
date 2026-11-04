# DequantSituQuant

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    ×     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: Adds dequant and quant operations before and after the Situ activation function to implement the DequantSituQuant computation of x.

- Formula:

  1. The dequantization path differs depending on the data type of the input x:

     - INT8 path:

       $$
       dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i + bias_i
       $$

     - INT32 path:

       $$
       dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i \times activation\_scale_i + bias_i
       $$

     - BF16/FLOAT16 path (pre-dequantization):

       $$
       dequantOut_i = cast\_to\_float(x_i)
       $$

  2. Situ activation:

     $$
     situ_a = \beta \times \tanh(gate / \beta) \times sigmoid(gate)
     $$

     When linear_beta > 0:

     $$
     up = linear\_beta \times \tanh(up / linear\_beta)
     $$

     $$
     situOut = situ_a \times up
     $$

     When activate_left is true, gate takes the first half of dequantOut and up takes the second half; when activate_left is false, gate takes the second half of dequantOut and up takes the first half.

  3. Quantization:

     - static mode:

       $$
       out_i = trunc(situOut_i / quant\_scale_i + quant\_offset_i)
       $$

     - dynamic mode:

       $$
       scale_i = absmax(situOut_i) / 127
       $$

       $$
       out_i = trunc(situOut_i / scale_i)
       $$

## Parameters

<table style="undefined;table-layout: fixed; width: 951px"><colgroup>
  <col style="width: 121px">
  <col style="width: 144px">
  <col style="width: 313px">
  <col style="width: 257px">
  <col style="width: 116px">
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
      <td>Data to be processed, which is x in the formula. The input cannot contain ±inf or nan.</td>
      <td>INT8, INT32, BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_scale</td>
      <td>Input</td>
      <td>Weight scale for dequantization, which is weight_scale in the formula. The input cannot contain ±inf or nan.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activation_scale</td>
      <td>Input</td>
      <td>Activation scale for dequantization, which is activation_scale in the formula. The input cannot contain ±inf or nan.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Bias for dequantization, which is bias in the formula. The input cannot contain ±inf or nan.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_scale</td>
      <td>Input</td>
      <td>Scale for quantization, which is quant_scale in the formula. The input cannot contain ±inf or nan.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_offset</td>
      <td>Input</td>
      <td>Offset for quantization, which is quant_offset in the formula. The input cannot contain ±inf or nan.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Input</td>
      <td>group_index required for MoE grouping. The input cannot contain ±inf or nan.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Attribute</td>
      <td>beta parameter of the Situ activation, which is β in the formula. It cannot be 0. Default: 4.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>Attribute</td>
      <td>linear_beta parameter of the Situ activation, which is linear_beta in the formula. It is not enabled when the value is ≤ 0. Default: 25.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>Attribute</td>
      <td>Specifies whether gate takes the first half or the second half of dequantOut, which is activate_left in the formula. Default: true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_type</td>
      <td>Attribute</td>
      <td>Specifies the quantization mode, corresponding to the static/dynamic mode in the formula. "static" and "dynamic" are supported. Default: "dynamic".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Quantized output, which is out in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>Output</td>
      <td>Scale of dynamic quantization, which is scale in the formula (meaningful only in dynamic mode).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The last dimension of x must be a multiple of 2.
- When the data type of x is INT8, x must have 2 or more dimensions; when the data type of x is INT32/BF16/FLOAT16, x must have 2 dimensions.
- The beta parameter cannot be 0.
- INT8 path: weight_scale must be provided, and activation_scale and group_index are prohibited.
- INT32 path: weight_scale and activation_scale must be provided, and quant_scale and quant_offset are prohibited.
- BFLOAT16/FLOAT16 path: none of the optional inputs are used (pre-dequantization mode).
- When quant_type is static, quant_scale must be provided.
- When quant_type is dynamic, quant_scale is optional (used as smoothScale).
- The memory size of the input tensors supported by the operator has an upper limit. The validation formula is: memory size of the weight_scale tensor + memory size of the bias tensor + memory size of the quant_scale tensor + memory size of the quant_offset tensor + (memory size of the activation_scale tensor + memory size of the y_scale tensor)/40 + memory size of the last dimension H of the x tensor * 10 < 192KB.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_dequant_situ_quant](./examples/test_aclnn_dequant_situ_quant.cpp) | Invokes the DequantSituQuant operator through the [aclnnDequantSituQuant](./docs/aclnnDequantSituQuant_en.md) interface.    |
| Graph-mode invocation | - | Invokes the DequantSituQuant operator through graph construction with the [operator IR](./op_graph/dequant_situ_quant_proto.h). |
