# MseLossGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: the backpropagation of the mean square error function [aclnnMseLoss](../mse_loss/docs/aclnnMseLoss_en.md).

- Formula:

  When `reduction` is `mean`:

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2 / x.numel()
  $$

  In the formula, `x.numel()` denotes the number of elements in `x`. If `reduction` is not `mean`, then:

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2
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
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>predict</td>
      <td>Input</td>
      <td>The input x in the formula. </td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>Input</td>
      <td>The input y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dout</td>
      <td>Input</td>
      <td>The input grad in the formula, the gradient value.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Input</td>
      <td>The input reduction in the formula, specifying the computation mode of the loss function. It supports 0 ('none') | 1 ('mean') | 2 ('sum'). 'none' indicates that no reduction is applied, 'mean' indicates that the sum of the output is divided by the number of elements in self, and 'sum' indicates that the output is summed.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output MselossBackward in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| aclnn invocation | [test_aclnn_mse_loss_grad](./examples/test_aclnn_mse_loss_backward.cpp) | Calls the mse_loss_grad operator through the [aclnnMseLossGrad](../mse_loss_grad_v2/docs/aclnnMseLossBackward_en.md) API. |
