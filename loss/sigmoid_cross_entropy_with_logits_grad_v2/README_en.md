# SigmoidCrossEntropyWithLogitsGradV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

Performs the `logits` computation on input `self`, and performs the backward propagation computation of `BCELoss` on the obtained value together with the label value `target`.

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
    <th class="tg-0lax">Input/Output/Attribute</th>
    <th class="tg-0lax">Description</th>
    <th class="tg-0lax">Data type</th>
    <th class="tg-0lax">Data format</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0lax">gradOutput</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">Gradient value from the previous step of network backpropagation.</td>
    <td class="tg-0lax">FLOAT16, FLOAT, BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">self</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">Computation result of the previous layer in the forward network.</td>
    <td class="tg-0lax">FLOAT16, FLOAT, BFLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">target</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">Label value of the sample, with the value range 0~1.</td>
    <td class="tg-0lax">Same as self</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">weightOptional</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">Weight of the binary cross entropy, whose shape can be broadcast to the shape of self</td>
    <td class="tg-0lax">Same as self</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">posWeightOptional</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">Weight of the positive class.</td>
    <td class="tg-0lax">Same as self</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0lax">reduction</td>
    <td class="tg-0lax">Input</td>
    <td class="tg-0lax">Denotes the reduce operation performed on the backward gradient computation result of the binary cross entropy.</td>
    <td class="tg-0lax">INT64</td>
    <td class="tg-0lax">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">out</td>
    <td class="tg-0lax">Output</td>
    <td class="tg-0lax">Stores the gradient computation result.</td>
    <td class="tg-0lax">Same as self</td>
    <td class="tg-0lax">Same as self</td>
  </tr>
</tbody></table>

## Constraints

- Deterministic computation:

    - aclnnBinaryCrossEntropyWithLogitsBackward uses the default deterministic implementation.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_sigmoid_cross_entropy_with_logits_grad_v2.cpp](examples/test_aclnn_sigmoid_cross_entropy_with_logits_grad_v2.cpp) | Invokes the SigmoidCrossEntropyWithLogitsGradV2 operator through the [aclnnBinaryCrossEntropyWithLogitsBackward](docs/aclnnBinaryCrossEntropyWithLogitsBackward_en.md) interface. |
