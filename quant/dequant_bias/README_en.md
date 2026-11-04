# DequantBias

## Product Support

| Product | Supported |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT series products</term> | √ |
| <term>Atlas A3 series products</term> | √ |
| <term>Atlas A2 series products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference series products</term> | × |
| <term>Atlas training series products</term> | × |
| <term>kirin X90 processor series products</term> | √ |
| <term>kirin 9030 processor series products</term> | √ |

## Function Description

- Operator function: dequantizes the input x, converting int32 input data to FLOAT16/BFLOAT16 output.
- Usage scenarios: backward dequantization for quantization-aware training (QAT) and quantized inference. `weight_scale` carries per-channel weight quantization scales (broadcast along the M dimension), `activate_scale` carries per-token activation quantization scales (broadcast along the N dimension), and `bias` is the bias added back before quantization; the three combined restore int32 quantized values to the floating-point domain.
- Computation formulas (selected by bias presence and dtype; the three are not mathematically equivalent):

  - When bias is absent:

  $$
  y = A \times \text{weight\_scale} \times \text{activate\_scale}
  $$

  - When bias is present and INT32 (integer-domain addition first, then unified scaling):

  $$
    y = (A + \text{bias}) \times \text{weight\_scale} \times \text{activate\_scale}
  $$

  - When bias is present and floating-point (FLOAT/FLOAT16/BFLOAT16) (add bias in the floating-point domain after dequantization):

  $$
    y = A \times \text{weight\_scale} \times \text{activate\_scale} + \text{bias}
  $$

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
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input A in the formulas.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_scale</td>
      <td>Input</td>
      <td>Input weight_scale in the formulas (named weightScale at the aclnn interface layer). Per-channel weight scaling factor, shape [N], length equal to dimension N of x.</td>
      <td>BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activate_scale</td>
      <td>Input (optional)</td>
      <td>Input activate_scale in the formulas (named activateScaleOptional at the aclnn interface layer). Per-token activation scaling factor, shape [M], length equal to dimension M of x. Optional input: omit when not participating in computation (at the aclnn layer, passing a null pointer or empty Tensor is treated as absent).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>bias</td>
      <td>Input (optional)</td>
      <td>Input bias in the formulas (named biasOptional at the aclnn interface layer). Shape [N], length equal to dimension N of x. Optional input: omit when not participating in computation (at the aclnn layer, passing a null pointer or empty Tensor is treated as absent). When present and INT32, formula 2 (integer-domain addition) applies; floating-point types follow formula 3.</td>
      <td>BFLOAT16, FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_dtype</td>
      <td>Attribute</td>
      <td><ul><li>Data type of output y (named outputDtype at the aclnn interface layer).</li><li>Legal values are only 1 and 27: 1 for FLOAT16 (used when weight_scale is FLOAT), 27 for BFLOAT16 (used when weight_scale is BFLOAT16), bound to the weight_scale data type.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output y in the formulas (named out at the aclnn interface layer). Shape [M, N], identical to input x.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- kirin X90/kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

- N and M in the shapes of input and output parameters must be positive integers, and M must be no greater than 25000.
- x must be a 2D tensor [M, N]; empty tensors are supported (empty output returned when M or N is 0).
- The length of weight_scale/bias must equal N; the length of activate_scale must equal M; for optional inputs (activate_scale/bias), passing an empty tensor is treated as absent.

## Invocation

| Invocation Method | Sample Code | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn API | [test_aclnn_dequant_bias](./examples/test_aclnn_dequant_bias.cpp) | Invoke the DequantBias operator via the [aclnnDequantBias](./docs/aclnnDequantBias.md) interface. |
| GE graph mode | - | Invoke the DequantBias operator by constructing a graph with the [operator IR](./op_graph/dequant_bias_proto.h). |
| PyTorch API | - | Invoked via the `torch.ops.npu.npu_dequant_bias` interface registered by the torch_npu plugin (implemented in op-plugin; ge.dequant_bias graph conversion registered in torchair); for sample construction, refer to the aclnn sample. |
