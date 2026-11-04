# PoissonNllLoss

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: computes the Poisson negative log likelihood loss, commonly used for count data regression, matching `torch.nn.PoissonNLLLoss` of PyTorch. The operator computes the loss element by element and then applies the reduction.
- Formula:

  When log_input is true:

  $$
  loss = e^{input\_x} - target \cdot input\_x
  $$

  When log_input is false:

  $$
  loss = input\_x - target \cdot ln(input\_x + eps)
  $$

  When full is true, the Stirling approximation term is additionally added for elements with $target > 1$ (the term is 0 when $target \leq 1$):

  $$
  loss = loss + \left(target \cdot ln(target) - target + 0.5 \cdot ln(2\pi \cdot target)\right)
  $$

  When reduction is none, the output is the elementwise loss (with the same shape as input_x); when reduction is sum, the output is the sum of all elements (a scalar); when reduction is mean, the output is the mean of all elements (a scalar, that is, divided by the total element count N).

## Parameters

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter name</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data type</th>
      <th style="white-space: nowrap">Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>input_x</td>
      <td>Input</td>
      <td>Prediction value tensor, input_x in the formula. The data type must be consistent with target, and the shape must be the same as target. Empty tensors (with a total element count of 0) are supported: when reduction is "none", the output is an empty tensor; when reduction is "sum", the output is 0; when reduction is "mean", the output is nan.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td>Target value tensor, target in the formula. The data type must be consistent with input_x, and the shape must be exactly the same as input_x.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_input</td>
      <td>Attribute</td>
      <td>Controls whether to exponentiate the input. The value is true/false, and the default value is true. For the computation details, see Formula.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>full</td>
      <td>Attribute</td>
      <td>Controls whether to add the Stirling approximation term. The value is true/false, and the default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>A small constant that prevents ln(0) when log_input is false. The default value is 1e-8, and it cannot be 0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Attribute</td>
      <td>Denotes the reduce operation applied to the elementwise loss. The value is "none"/"mean"/"sum", and the default value is "mean".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>Output</td>
      <td>Loss computation result, loss in the formula. The data type is consistent with the inputs; when reduction is none, the shape is the same as input_x, and when reduction is mean/sum, the output is a scalar.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_poisson_nll_loss.cpp](examples/arch35/test_geir_poisson_nll_loss.cpp) | Calls the PoissonNllLoss operator by building a graph through [operator IR](op_graph/poisson_nll_loss_proto.h). |
