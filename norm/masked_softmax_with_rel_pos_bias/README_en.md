# MaskedSoftmaxWithRelPosBias

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Replaces the part that uses window attention to compute softmax in swinTransformer.
- Formula:

  $$
  out=\operatorname{softmax}(scaleValue*x+attenMaskOptional+relativePosBias)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
  <col style="width: 213px">
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
      <td>Corresponds to x in the formula. The shapes 4D (B*W, N, S1, S2) and 5D (B, W, N, S1, S2) are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Optional input</td>
      <td>Corresponds to attenMaskOptional in the formula. It is a null pointer when not needed. The shapes 3D (W, S1, S2), 4D (W, 1, S1, S2), and 5D (1, W, 1, S1, S2) are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>relativePosBias</td>
      <td>Input</td>
      <td>Corresponds to relativePosBias in the formula. The shapes 3D (N, S1, S2), 4D (1, N, S1, S2), and 5D (1, 1, N, S1, S2) are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>Optional attribute</td>
      <td>Corresponds to scaleValue in the formula. The IR attribute defaults to 1.0; ACLNN callers must pass it explicitly. It must be a finite value representable by float32; NaN/±Inf is rejected.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecisionMode</td>
      <td>Optional attribute</td>
      <td>Precision mode parameter. Default: 0; reserved field; only 0 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Corresponds to out in the formula. The shapes 4D (B*W, N, S1, S2) and 5D (B, W, N, S1, S2) are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas inference products</term>: BFLOAT16 is not supported, and the scenario where S2 of the input is not aligned to 32 bytes is not supported.
- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: The data types of all inputs and outputs do not support BFLOAT16.

## Constraints

> In the parameter table, `attenMaskOptional`, `relativePosBias`, `scaleValue`, and `innerPrecisionMode` are ACLNN C-header parameter names; the corresponding IR/OpDef names are `atten_mask`, `relative_pos_bias`, `scale_value`, and `inner_precision_mode`.

- The dtype of attenMaskOptional (when not null), relativePosBias, and out must be the same as that of x.
- scaleValue defaults to 1.0 and must be finite and representable by float32; NaN/±Inf is rejected. innerPrecisionMode is a reserved field; its default is 0 and only 0 is supported.
- For 4D x (B*W, N, S1, S2) with attenMaskOptional, the W axis of the mask divides the D=B*W axis (D%W==0); this defines the merged-axis decomposition rule when B>1.
- Empty tensors are not supported; callers must not rely on their behavior. An empty output returns success without a guaranteed result; if attenMaskOptional or relativePosBias is empty while the output is non-empty, an error is reported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_masked_softmax_with_rel_pos_bias](examples/test_aclnn_masked_softmax_with_rel_pos_bias.cpp) | Calls the MaskedSoftmaxWithRelPosBias operator through the [aclnnMaskedSoftmaxWithRelPosBias](docs/aclnnMaskedSoftmaxWithRelPosBias_en.md) interface. |
| Graph mode | [test_geir_masked_softmax_with_rel_pos_bias](examples/test_geir_masked_softmax_with_rel_pos_bias.cpp)  | Calls the MaskedSoftmaxWithRelPosBias operator by building a graph through [operator IR](op_graph/masked_softmax_with_rel_pos_bias_proto.h).         |
