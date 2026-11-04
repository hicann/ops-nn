# HardSigmoid

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: an activation function that applies the HardSigmoid transformation element by element to the input tensor self, and outputs a tensor with the same shape as the input.
- Formula:

  $$
  HardSigmoid(self)=clip(\alpha \times self + \beta, 0, 1)
  $$

  The default values are $\alpha=\frac{1}{6}$ and $\beta=\frac{1}{2}$, which is equivalent to the following piecewise formula:

  $$
  HardSigmoid(self)=\begin{cases}
  &1, &if(self\gt3) \\
  &0, &if(self\le-3) \\
  &\frac{self}{6} + \frac{1}{2}, &otherwise
  \end{cases}
  $$

  When the input is INT32, the computation result is truncated and converted to INT32 output.

## Parameters

  <table style="table-layout: fixed; width: 800px"><colgroup>
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
      <td>self</td>
      <td>Input</td>
      <td>Empty tensors are supported. Denotes the input of the activation function, the input self in the formula. The shape has no more than 8 dimensions.</td>
      <td>FLOAT, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Empty tensors are supported. Denotes the output of the activation function, the output HardSigmoid(self) in the formula. Both the data type and the shape are consistent with the input self.</td>
      <td>FLOAT, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Attribute</td>
      <td>Optional attribute, the slope of the activation function, α in the formula. The default value is 1/6.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Attribute</td>
      <td>Optional attribute, the offset of the activation function, β in the formula. The default value is 1/2.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas training products</term>: the data types support FLOAT, FLOAT16, and INT32.

## Constraints

- Graph mode supports dynamic shapes: a dimension value of `-1` indicates an unknown dimension size, and a shape of `{-2}` indicates an unknown rank; the corresponding dynamic flags are retained during inference.
- The input rank ranges from 0 to 8; when the rank is unknown, the actual rank is determined at runtime.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_hard_sigmoid](examples/test_aclnn_hard_sigmoid.cpp) | Calls the HardSigmoid operator through the [aclnnHardsigmoid&aclnnInplaceHardsigmoid](docs/aclnnHardsigmoid&aclnnInplaceHardsigmoid_en.md) interface. |
| aclnn API | [test_aclnn_inplace_hard_sigmoid](examples/test_aclnn_inplace_hard_sigmoid.cpp) | Calls the HardSigmoid operator through the [aclnnHardsigmoid&aclnnInplaceHardsigmoid](docs/aclnnHardsigmoid&aclnnInplaceHardsigmoid_en.md) interface. |
| GE graph mode | [test_geir_hard_sigmoid](examples/arch35/test_geir_hard_sigmoid.cpp) | Calls the HardSigmoid operator by building a graph through [operator IR](op_graph/hard_sigmoid_proto.h). |
