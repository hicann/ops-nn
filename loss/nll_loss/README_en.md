# NllLoss

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

  - Description: computes the negative log-likelihood loss value.

  - Formula:

    When `reduction` is `none`:

    $$
    \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
    l_n = - w_{y_n} x_{n,y_n}, \quad
    w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not=   \text{ignoreIndex}\},
    $$

    Here $x$ is self, $y$ is target, $w$ is weight, and $N$ is the batch size. If `reduction` is not `'none'`, then:

    $$
    \ell(x, y) = \begin{cases}
        \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
        \text{if reduction} = \text{`mean';}\\
        \sum_{n=1}^N l_n,  &
        \text{if reduction} = \text{`sum'.}
    \end{cases}
    $$

    $$
    totalWeight = \sum_{n=1}^N w_{y_n}
    $$

## Parameters

  <table style="undefined;table-layout: fixed; width: 980px"><colgroup>
    <col style="width: 100px">
    <col style="width: 150px">
    <col style="width: 280px">
    <col style="width: 330px">
    <col style="width: 120px">
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
        <td>The input parameter for the nll_loss computation, x in the formula.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>target</td>
        <td>Input</td>
        <td>The input parameter for the nll_loss computation, target in the formula.</td>
        <td>INT32, INT64, UINT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>The input parameter for the nll_loss computation, weight in the formula.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>Input</td>
        <td>The input parameter for the nll_loss computation, reduction in the formula.</td>
        <td>STRING</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>ignore_index</td>
        <td>Input</td>
        <td>The input parameter for the nll_loss computation, ignoreIndex in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>The output parameter of the nll_loss computation, L in the formula.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## Constraints

  None

## Invocation Description

  | Invocation method | Invocation sample                                                                   | Description                                                           |
  |--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
  | aclnn invocation | [test_aclnn_nll_loss](./examples/test_aclnn_nll_loss.cpp) | Invoke the NLLLoss operator through the [aclnnNLLLoss](./docs/aclnnNLLLoss_en.md) interface. |
  | aclnn invocation | [test_aclnn_nll_loss_2d](./examples/test_aclnn_nll_loss_2d.cpp) | Invoke the NLLLoss operator through the [aclnnNLLLoss2d](./docs/aclnnNLLLoss2d_en.md) interface. |
