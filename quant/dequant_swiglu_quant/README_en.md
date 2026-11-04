# DequantSwigluQuant

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>kirin X90 processor series products</term> | √ |
|  <term>kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Adds dequant and quant operations before and after the Swish gated linear unit activation function to implement the DequantSwigluQuant computation of x.
- Formula when swiglu_mode is 0:

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  swigluOut_i = Swiglu(dequantOut_i)=Swish(A_i)*B_i
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  Here, A<sub>i</sub> represents the first half of dequantOut<sub>i</sub>, and B<sub>i</sub> represents the second half of dequantOut<sub>i</sub>.

- Formula when swiglu_mode is 1:

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  x\_glu = x\_glu.clamp(min=None, max=clamp\_limit)
  $$

  $$
  x\_linear = x\_linear.clamp(min=-clamp\_limit, max=clamp\_limit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(glu\_alpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + glu\_bias)
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  Here, x\_glu represents the even-index part of dequantOut<sub>i</sub>, and x\_linear represents the odd-index part of dequantOut<sub>i</sub>.

- Formula when swiglu_mode is 2:

  The computation logic is the same as when swiglu_mode is 1 (both are the GPT-OSS variant SwiGLU, using clamp_limit, glu_alpha, and glu_bias):

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  x\_glu = x\_glu.clamp(min=None, max=clamp\_limit)
  $$

  $$
  x\_linear = x\_linear.clamp(min=-clamp\_limit, max=clamp\_limit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(glu\_alpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + glu\_bias)
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  The difference from swiglu_mode 1 lies in how x\_glu and x\_linear are split: when swiglu_mode is 2, x\_glu represents the first half of dequantOut<sub>i</sub>, and x\_linear represents the second half of dequantOut<sub>i</sub> (the same split method as when swiglu_mode is 0); when swiglu_mode is 1, the split interleaves even and odd indices.

- Formula when swiglu_mode is 3:

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  gate = dequantOut_i[:, :H/2]
  $$

  $$
  up = dequantOut_i[:, H/2:]
  $$

  $$
  gate = SiLU(gate) = gate * sigmoid(gate)
  $$

  $$
  gate = gate.clamp(max=clamp\_limit)
  $$

  $$
  up = up.clamp(min=-clamp\_limit, max=clamp\_limit)
  $$

  $$
  swigluOut_i = gate * up
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  Here, gate represents the first half of dequantOut<sub>i</sub>, and up represents the second half of dequantOut<sub>i</sub> (the same split method as when swiglu_mode is 0). swiglu_mode 3 uses the SiLU activation function (consistent with swiglu_mode 0) and the clamp limiting mechanism (using clamp_limit), but does not use the glu_alpha and glu_bias parameters.

## Parameters

<table style="undefined;table-layout: fixed; width: 951px"><colgroup>
  <col style="width: 221px">
  <col style="width: 144px">
  <col style="width: 213px">
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
      <td>Input data to be processed, x in the formula. The input cannot contain ±inf or nan.</td>
      <td>FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>weight_scale</td>
      <td>Input</td>
      <td>The input cannot contain ±inf or nan.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>activation_scale</td>
      <td>Input</td>
      <td>Dequantization scale of the activation function. The input cannot contain ±inf or nan.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>bias</td>
      <td>Input</td>
      <td>The bias of Matmul, bias in the formula. The input cannot contain ±inf or nan.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_scale</td>
      <td>Input</td>
      <td>Quantization scale, quant_scale in the formula. The input cannot contain ±inf or nan.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>quant_offset</td>
      <td>Input</td>
      <td>Quantization offset. The input cannot contain ±inf or nan.</td>
      <td>FLOAT</td>
      <td>ND</td>
     </tr>
      <tr>
      <td>group_index</td>
      <td>Input</td>
      <td>The group_index required for MoE grouping. The input cannot contain ±inf or nan.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>activate_left</td>
      <td>Attribute</td>
      <td>Specifies whether to perform the swiglu activation on the left half of the input.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td>Indicates that dynamic quantization is used.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
     <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Specifies the data type of output y.</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
     <tr>
      <td>round_mode</td>
      <td>Attribute</td>
      <td>Specifies the rounding mode for the output y result.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
      <tr>
      <td>activate_dim</td>
      <td>Attribute</td>
      <td>Specifies the selected split axis for the swish computation.</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
       <tr>
      <td>swiglu_mode</td>
      <td>Attribute</td>
      <td>Specifies the swiglu computation mode. Values are 0/1/2/3: 0 is traditional swiglu; 1 is variant swiglu (even-odd chunking); 2 is variant swiglu (contiguous first-half/second-half chunking); 3 is variant swiglu (contiguous first-half/second-half chunking, with clamp after silu)</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
      <tr>
      <td>clamp_limit</td>
      <td>Attribute</td>
      <td>Specifies the threshold value used by the variant swiglu.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>glu_alpha</td>
      <td>Attribute</td>
      <td>Specifies the parameter used by the variant swiglu.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>glu_bias</td>
      <td>Attribute</td>
      <td>Specifies the bias parameter used by the variant swiglu.</td>
      <td>FLOAT</td>
      <td>-</td>
     </tr>
       <tr>
      <td>y</td>
      <td>Output</td>
      <td>-</td>
      <td>INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Output</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/kirin 9030 processor series products:
  - Input `x`: The data type does not support BFLOAT16.
  - Input `bias`: The data type does not support BFLOAT16.
  - Input `quant_scale`: The data type does not support FLOAT16.
  - Output `y`: The data type does not support FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, or FLOAT4_E1M2.

## Constraints

- <term>Ascend 950PR&950DT products</term>:
  - The dimension of input x corresponding to activate_dim must be a multiple of 2, and the number of dimensions of x must be greater than 1.
  - When the data type of input x is INT32, weight_scale cannot be empty; when the data type of input x is not INT32, weight_scale cannot be provided as an input and a null pointer is passed in.
  - When the data type of input x is not INT32, activation_scale cannot be provided as an input and the parameter is set to a null pointer.
  - When the data type of input x is not INT32, bias cannot be provided as an input and the parameter is set to a null pointer.
  - When the data type of output y is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of y must be a multiple of 2.
  - The tail axis of output y does not exceed 5120.

- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - The four parameters swiglu_mode, clamp_limit, glu_alpha, and glu_bias are used for the GPT-OSS variant SwiGLU.
  - The last dimension of x must be a multiple of 2, and the number of dimensions of x must be greater than 1.
  - When quant_mode is static, quant_scale and quant_offset are 1-dimensional with a value of 1; when quant_mode is dynamic, quant_scale and quant_offset
  - The memory size of input tensors supported by the operator has an upper limit. Validation formula: weight_scale tensor memory size + bias tensor memory size + quant_scale tensor memory size + quant_offset tensor memory size + (activation_scale tensor memory size + scale tensor memory size) / 40 + x tensor last-dimension H memory size * 10 < 192KB.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_dequant_swiglu_quant](./examples/test_aclnn_dequant_swiglu_quant.cpp) | Calls the DequantSwigluQuant operator through the [aclnnDequantSwigluQuant](./docs/aclnnDequantSwigluQuant_en.md) interface.    |
| aclnn invocation | [test_aclnn_dequant_swiglu_quant_v2](./examples/test_aclnn_dequant_swiglu_quant_v2.cpp) | Calls the DequantSwigluQuant operator through the [aclnnDequantSwigluQuantV2](./docs/aclnnDequantSwigluQuantV2_en.md) interface.    |
| Graph-mode invocation | -   | Calls the DequantSwigluQuant operator by building a graph through the [operator IR](./op_graph/dequant_swiglu_quant_proto.h). |
