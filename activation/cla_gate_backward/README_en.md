# ClaGateBackward

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×    |
| <term>Atlas A2 products</term> |    ×    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×    |
| <term>Atlas training products</term>                              |    ×    |

## Function

- Operator function: This is a fused operator that implements backward gradient computation of CLA (Cross-Layer Attention) gate merge. It takes the upstream gradient G = ∂L/∂O, the two original Attention outputs O_g/O_l, and the two gate logits z_g/z_l, and computes and returns four gradients in one pass: the gradients of the two Attention outputs (dO_g, dO_l) and the gradients of the two gate logits (dz_g, dz_l). This interface does not quantize gradients: outputs keep the input precision (BF16/FP16) and no FP8 gradients are generated. The backward operator does not receive the intermediate sigmoid results of the forward pass; s_g/s_l are recomputed inside the operator.

- Formula:

  Let T be the number of tokens on this rank, N the number of attention heads, and D the value head dim.

  **Stage 1: Sigmoid recomputation (recomputed inside the operator, FP32 domain)**

  - The two head-wise Sigmoid gates and their derivative terms (z_g and z_l are the two gate logits):

    $$
    s\_g = \sigma(z\_g), \qquad s\_l = \sigma(z\_l), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
    $$

  **Stage 2: Attention output gradients (element-wise chain rule, with s_g/s_l broadcast along the D dimension)**

  $$
  dO\_g = G \odot s\_g, \qquad dO\_l = G \odot s\_l
  $$

  **Stage 3: Gate logits gradients (chain rule + Sigmoid derivative; sum_d reduces only along head_dim D and does not cross tokens or heads)**

  $$
  dz\_g = \left( \sum_{d=0}^{D-1} G \odot O\_g \right) \odot s\_g \odot (1 - s\_g), \qquad
  dz\_l = \left( \sum_{d=0}^{D-1} G \odot O\_l \right) \odot s\_l \odot (1 - s\_l)
  $$

  - The products G⊙O_g/G⊙O_l and the reduction along D are completed in the FP32 domain (within one pass, with no extra memory round trip) to reduce the reduction accumulation error.
  - No division/exponent overflow path exists; NaN/Inf inputs propagate according to IEEE semantics (no special protection is required).

## Parameters

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 640px">
  <col style="width: 180px">
  <col style="width: 90px">
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
      <td>grad_merged</td>
      <td>Input</td>
      <td>Gradient back-propagated from the output projection to gate merge, that is, G in the formula. The shape is [T, N, D]. Non-contiguous tensors are supported, and empty tensors are not supported.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>global_attn</td>
      <td>Input</td>
      <td>Attention output of the forward Global/CLA branch, that is, O_g in the formula. The shape is the same as grad_merged, that is, [T, N, D]. The data type is the same as grad_merged. Non-contiguous tensors are supported, and empty tensors are not supported.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>local_attn</td>
      <td>Input</td>
      <td>Attention output of the forward Local/SWA branch, that is, O_l in the formula. The shape is the same as grad_merged, that is, [T, N, D]. The data type is the same as grad_merged. Non-contiguous tensors are supported, and empty tensors are not supported.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>global_gate_logits</td>
      <td>Input</td>
      <td>Logits of the forward Global gate, that is, z_g in the formula. The shape is [T, N], where T and N are the same as those of grad_merged. The data type is the same as grad_merged. Non-contiguous tensors are supported, and empty tensors are not supported.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>local_gate_logits</td>
      <td>Input</td>
      <td>Logits of the forward Local gate, that is, z_l in the formula. The shape is the same as global_gate_logits, that is, [T, N]. The data type is the same as grad_merged. Non-contiguous tensors are supported, and empty tensors are not supported.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_global_attn_out</td>
      <td>Output</td>
      <td>Gradient of the Global Attention branch, that is, dO_g in the formula. The shape is [T, N, D]. The data type is the same as global_attn.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_local_attn_out</td>
      <td>Output</td>
      <td>Gradient of the Local Attention branch, that is, dO_l in the formula. The shape is [T, N, D]. The data type is the same as local_attn.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_global_gate_logits_out</td>
      <td>Output</td>
      <td>Gradient of the Global gate logits, that is, dz_g in the formula. The shape is [T, N]. The data type is the same as global_gate_logits.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_local_gate_logits_out</td>
      <td>Output</td>
      <td>Gradient of the Local gate logits, that is, dz_l in the formula. The shape is [T, N]. The data type is the same as local_gate_logits.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_attn_layout</td>
      <td>Optional attribute</td>
      <td>Layout format string of the input tensor. Currently, only "TND" is supported. The default value is "TND".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- `grad_merged`, `global_attn`, and `local_attn` support three dimensions `[T, N, D]`, where `D` supports only 128 or 256.

## Invocation Description

- <term>Ascend 950PR&950DT products</term>:

  | Invocation method | Invocation sample | Description |
  |---------|---------|------|
  | aclnn API | [test_aclnn_cla_gate_backward](./examples/test_aclnn_cla_gate_backward.cpp) | Invokes the ClaGateBackward operator through the [aclnnClaGateBackward](./docs/aclnnClaGateBackward.md) interface. |
  | PyTorch API | - | Invokes the ClaGateBackward operator through the [cla_gate_backward](./docs/torchapi_cla_gate_backward.md) interface. |
