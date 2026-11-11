# ModulateGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>   |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Operator function: Completes the computation of the parameters in the Modulate backpropagation and performs the gradient update.
- Calculation formula:

    Assume that the shape of the input self is [B, L, D]. The calculation formula is as follows:
    Formula:

    $$
    \begin{cases}
    \text{grad\_x} = \text{grad\_output} \odot \text{scale}^{\uparrow L} \\
    \text{grad\_scale} = \sum_{l=1}^{L} (\text{grad\_output} \odot \text{input})_{b,l,d} \\
    \text{grad\_shift} = \sum_{l=1}^{L} \text{grad\_output}_{b,l,d}
    \end{cases}
    $$

    Symbol description:
    - $\odot$: denotes element-wise multiplication;
    - $\sum_{l=1}^{L}$: summation operation, performed along the sequence dimension $L$ (that is, dim=1)
    - $b,l,d$: subscripts indicating the dimension indexes of the tensor (usually Batch, Length, Dimension)
    - $\text{scale}^{\uparrow L}$: indicates that the scale tensor is broadcast (expanded) along the sequence dimension $L$

## Parameters

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>grad_output</td>
      <td>Input</td>
      <td>Denotes the input feature tensor, grad_output in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Denotes the input feature tensor, input in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Optional parameter, denoting the scaling coefficient, scale in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>Input</td>
      <td>Optional parameter, denoting the shift coefficient, shift in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>Output</td>
      <td>Denotes the scaled output tensor, grad_input in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>grad_scale</td>
      <td>Output</td>
      <td>Denotes the scaled output tensor, grad_scale in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>grad_shift</td>
      <td>Output</td>
      <td>Denotes the scaled output tensor, grad_shift in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_modulatebackward](./examples/test_aclnn_modulatebackward.cpp) | Invokes the ModulateGrad operator through the [aclnnModulateBackward](../../vfusion/modulate_grad/docs/aclnnModulateBackward.md) API. |
