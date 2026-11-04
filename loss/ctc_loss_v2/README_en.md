# CtcLossV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: computes the connectionist temporal classification loss value.

- Computation formula:
  Define $y_{k}^{t}$ as the probability that the true character is $k$ at time $t$. (Generally, $y_{k}^{t}$ is an element of the output matrix after softmax.) The set of all sequences that the character set $L^{'}$ can form is called $L^{'T}$, any sequence in $L^{'T}$ is called a path, and is denoted as $π$. The distribution of $π$ is formula (1):

  $$
  p(π|x)=\prod_{t=1}^{T}y^{t}_{π_{t}} , \forall π \in L'^{T}. \tag{1}
  $$

  Define the many-to-one mapping B: $L^{'T} \to L^{\leq T}$. The conditional probability of $l \in L^{\leq T}$ computed through mapping B equals the sum of the probabilities of all possible paths corresponding to $l$, formula (2):

  $$
  p(l|x)=\sum_{π \in B^{-1}(l)}p(π|x).\tag{2}
  $$

  The task of finding the path of the $l$ that maximizes the value of $p(l|x)$ is called decoding, formula (3):

  $$
  h(x)=^{arg \  max}_{l \in L^{ \leq T}} \ p(l|x).\tag{3}
  $$

  When zeroInfinity is True

  $$
  h(x)=\begin{cases}0,&h(x) == Inf \text{ or } h(x) == -Inf \\h(x),&\text { else }\end{cases}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
    <col style="width: 110px">
    <col style="width: 151px">
    <col style="width: 145px">
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
      <td>Denotes the log probabilities of the output, y in the formula. </td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td>Denotes the labels containing the target sequences, π in the formula.</td>
      <td>INT64, INT32, BOOL, FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_lengths</td>
      <td>Input</td>
      <td>Denotes the actual lengths of the input sequences, where T in the formula is an element of inputLengths.</td>
      <td>int64_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_lengths</td>
      <td>Input</td>
      <td>Denotes the actual lengths of the target sequences, where the length of l in the formula is an element of targetLengths.</td>
      <td>int64_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>neg_log_likelihood</td>
      <td>Output</td>
      <td>Denotes the output loss value, h in the formula.</td>
      <td>Same as log_probs</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_alpha</td>
      <td>Output</td>
      <td>Denotes the probability of the possible trace from the input to the target, p(l|x) in the formula.</td>
      <td>Same as log_probs</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_ctc_loss_v2](./examples/arch35/test_aclnn_ctc_loss_v2.cpp) | Calls the CtcLossV2 operator through the [aclnnCtcLoss](./docs/aclnnCtcLoss_en.md) interface.    |
