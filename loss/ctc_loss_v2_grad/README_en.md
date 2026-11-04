# CtcLossV2Grad

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

- Operator function: Computes the backward of the Connectionist Temporal Classification loss value.
- Formula:
Let $y_{k}^{t}$ denote the probability that the real character is $k$ at time $t$. (Generally, $y_{k}^{t}$ is an element of the output matrix after softmax.) The set of all sequences that the character set $L^{'}$ can form is called $L^{'T}$, any sequence in $L^{'T}$ is called a path and is denoted as $π$. The distribution of $π$ is formula (1):

  $$
  p(π|x)=\prod_{t=1}^{T}y^{t}_{π_{t}} , \forall π \in L'^{T}. \tag{1}
  $$

  Define the many-to-one mapping B: $L^{'T} \to L^{\leq T}$. The conditional probability of $l \in L^{\leq T}$ computed through the mapping B equals the sum of the probabilities of all possible paths corresponding to $l$, formula (2):

  $$
  p(l|x)=\sum_{π \in B^{-1}(l)}p(π|x).\tag{2}
  $$

  The task of finding the path of the $l$ that maximizes the value of $p(l|x)$ is called decoding, formula (3):

  $$
  h(x)=^{arg \  max}_{l \in L^{ \leq T}} \ p(l|x).\tag{3}
  $$

  When zeroInfinity is True

  $$
  h(x)=\begin{cases}0,&h(x) == inf \text{ or } h(x) == -inf \\h(x),&\text { else }\end{cases}
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
      <td>grad_out</td>
      <td>Input</td>
      <td>Gradient update coefficient.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_probs</td>
      <td>Input</td>
      <td>Logarithm of the output probabilities.</td>
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
      <td>neg_log_likelihood</td>
      <td>Input</td>
      <td>Differentiable loss value with respect to each node.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_alpha</td>
      <td>Input</td>
      <td>Probability of the input sequence traced to the target sequence.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
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
      <td>grad</td>
      <td>Output</td>
      <td>Loss gradient of ctcloss.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

* The number of elements of grad_out must be N.
* The shape of log_probs is (T, N, C), where T >= max( input_lengths ).
* The shape of targets is (N, S) or ( sum( target_lengths ),).
* The number of elements of input_lengths must be N.
* The number of elements of target_lengths must be N.
* The shape of neg_log_likelihood must be 1D (N).
* The shape of log_alpha must be 3D (N, T, X), where X = ( 2 * S + 1 ).
* The value range of reduction is {'none', 'mean', 'sum'}.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_ctc_loss_backward](examples/arch35/test_aclnn_ctc_loss_backward.cpp) | Calls the CtclossV2Grad operator through the [aclnnCtcLossBackward](docs/aclnnCtcLossBackward_en.md) interface. |
