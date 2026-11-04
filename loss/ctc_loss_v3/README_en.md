# CtcLossV3

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: computes the loss between a continuous (unsegmented) time series and a target sequence.
- Formula:

  $$
  O^{ML}(S,N_w) = - \sum_{x,z\in S} {\ln{p(z|x)}}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
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
      <td>log_probs</td>
      <td>Input</td>
      <td>Probability of each token at each moment of the input.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>targets</td>
      <td>Input</td>
      <td>Target sequence.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_lengths</td>
      <td>Input</td>
      <td>Length of the input sequence.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_lengths</td>
      <td>Input</td>
      <td>Length of the target sequence.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>blank</td>
      <td>Attribute</td>
      <td>Position of the blank character.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Attribute</td>
      <td>Specifies the reduction method for computing the output.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>zero_infinity</td>
      <td>Attribute</td>
      <td>Whether to zero infinite losses and the associated gradients.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>neg_log_likelihood</td>
      <td>Output</td>
      <td>Loss value differentiable with respect to each node.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_alpha</td>
      <td>Output</td>
      <td>Probability that the input sequence traces to the target sequence.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

* The shape of log_probs is (T, N, C), where T>=max(input_lengths).
* The shape of targets is (N, S) or (sum(target_lengths),), where S>=max(target_lengths).
* The value range of reduction is {'none', 'mean', 'sum'}.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_ctc_loss_v3](examples/test_aclnn_ctc_loss_v3.cpp) | Calls the CtclossV3 operator through the [aclnnCtcLoss](../ctc_loss_v2/docs/aclnnCtcLoss_en.md) interface. |
