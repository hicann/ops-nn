# SigmoidCrossEntropyWithLogitsGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>   |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Operator function: Backward gradient computation of the fused loss of Sigmoid and binary cross entropy (BCELoss). The operator passes the input `predict` through `sigmoid`, subtracts the label `target`, and then multiplies by the upstream gradient `d_out` to obtain the gradient with respect to `predict`.
- Formula:

  $$
  gradient = (sigmoid(predict) - target) \cdot d_out
  $$

## Parameters

<table class="tg" style="undefined;table-layout: fixed; width: 940px"><colgroup>
<col style="width: 138px">
<col style="width: 121px">
<col style="width: 280px">
<col style="width: 280px">
<col style="width: 121px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0lax">Parameter name</th>
    <th class="tg-0lax">Input/Output</th>
    <th class="tg-0lax">Description</th>
    <th class="tg-0lax">Data type</th>
    <th class="tg-0lax">Data format</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0lax">predict</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">Computation result of the previous layer in the forward pass of the network (logits before sigmoid), which determines the shape and dtype of the output.</td>
    <td class="tg-0lax">FLOAT16, FLOAT, BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">target</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">Label value of the sample; the shape is consistent with that of predict.</td>
    <td class="tg-0lax">Same as predict</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">d_out</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">Gradient value from the previous step of network backward propagation; the shape is consistent with that of predict (the total number of elements equals that of predict; broadcasting is not supported).</td>
    <td class="tg-0lax">Same as predict</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">gradient</td>
    <td class="tg-0lax">Output</td>
    <td class="tg-0lax">Stores the gradient computation result; the shape and dtype are consistent with those of predict.</td>
    <td class="tg-0lax">Same as predict</td>
    <td class="tg-0lax">ND</td>
  </tr>
</tbody></table>

## Constraints

- The data types of predict, target, and dout must be the same; only FLOAT16, FLOAT, and BFLOAT16 are supported.
- The number of dimensions (rank) of predict, target, and dout must not exceed 8; if it exceeds 8, operator compilation reports an error.
- The shape of target must be exactly the same as that of predict; the total number of elements of dout must equal that of predict (verified by equal total element counts, not by broadcast deduction; broadcast semantics are not supported).
- Empty tensors are supported (any dimension is 0): with zero elements, the operator directly executes successfully with zero iterations and does not produce out-of-bounds access.
- Dynamic shapes are supported (including dynamic range partitioning); scenarios where the shape contains -1 are deduced according to the actual input.
- Only the ND data format is supported.
- Deterministic computation: SigmoidCrossEntropyWithLogitsGrad is deterministic by default.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode (GE IR)  | [test_geir_sigmoid_cross_entropy_with_logits_grad.cpp](examples/test_geir_sigmoid_cross_entropy_with_logits_grad.cpp) | Invokes the SigmoidCrossEntropyWithLogitsGrad operator in GE graph mode; for the operator prototype definition, see [sigmoid_cross_entropy_with_logits_grad_proto.h](op_graph/sigmoid_cross_entropy_with_logits_grad_proto.h). |
