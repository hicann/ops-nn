# HardShrinkGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: backward operator of [HardShrink](../hard_shrink/README.md), which computes the backward propagation gradient gradInput.

- Formula:

  $$
  HardshrinkBackward(x,grad)=
  \begin{cases}
  &0, &if(-\lambda \leq x \leq \lambda) \\
  &grad, &otherwise
  \end{cases}
  $$

  When x is NaN, the output at the corresponding position is grad; when grad is NaN and |x| > λ, the output at the corresponding position is NaN.

## Parameters

  <table style="undefined;table-layout: fixed; width: 800px"><colgroup>
  <col style="width: 110px">
  <col style="width: 130px">
  <col style="width: 300px">
  <col style="width: 180px">
  <col style="width: 80px">
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
      <td>gradients</td>
      <td>Input</td>
      <td>Gradient output by the previous step in backward propagation, used as the input of this backward operator, corresponding to grad in the formula. Non-contiguous tensor inputs are automatically converted to contiguous tensors by the framework before execution. Empty tensors (with zero elements) are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>features</td>
      <td>Input</td>
      <td>Indicates the input tensor, corresponding to x in the formula. Non-contiguous tensor inputs are automatically converted to contiguous tensors by the framework before execution. Empty tensors (with zero elements) are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>Output</td>
      <td>Backward propagation gradient output, corresponding to HardshrinkBackward(x,grad) in the formula. Non-contiguous tensor outputs are automatically converted to contiguous tensors by the framework before execution. Empty tensors are supported; when the inputs are empty tensors, gradInput is also an empty tensor and no computation is performed.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
   </tr>
    <tr>
      <td>lambd</td>
      <td>Attribute</td>
      <td>λ in the formula, threshold parameter, default value 0.5, with no restriction on its value.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas training products</term>: the data types FLOAT and FLOAT16 are supported.

## Constraints

- In GE graph mode, the shapes of gradients, features, and backprops must be the same; the data types of gradients, features, and backprops must be the same.

## Invocation Description

| Invocation method   | Sample code                                                                         | Description                                                                                 |
| ---------------- |------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| aclnn interface  | [test_aclnn_hard_shrink_backward.cpp](examples/test_aclnn_hard_shrink_backward.cpp) | Invokes the HardShrinkGrad operator through the [aclnnHardshrinkBackward](docs/aclnnHardshrinkBackward_en.md) interface. |
| GE graph mode | - | Invokes the HardShrinkGrad operator by constructing a graph with the [operator IR](./op_graph/hard_shrink_grad_proto.h). |
