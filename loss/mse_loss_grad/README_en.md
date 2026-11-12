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

- Operator function: Computes the gradient of the mean square error (MSE) loss with respect to the prediction value predict.

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

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>The input x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>Input</td>
      <td>The input y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dout</td>
      <td>Input</td>
      <td>The input grad in the formula, the gradient value.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output MselossBackward in the formula, whose shape is the broadcast result of the shapes of predict, label and dout.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Optional attribute</td>
      <td>The reduction in the formula, which is inherited from the forward loss and specifies the reduction mode of the loss. The default value is 'mean'. For the backward gradient: 'mean' scales each gradient element by the number of elements in predict (the coefficient is 2/numel(predict)); 'none' and 'sum' perform no extra scaling (the coefficient is 2). In all three modes, the output y retains the broadcast shape of predict, label and dout and is not summed.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The data types of the inputs predict, label and dout must be the same as that of the output y.
- The shapes of the inputs predict, label and dout must satisfy the broadcast relationship, and the shape of the output y is the broadcast result of the three.
- The attribute reduction only supports 'none', 'mean' and 'sum'; passing any other value causes an operator execution error.
- Empty tensor: when any input contains a zero-length dimension (empty tensor), the operator reports an error and rejects execution at the tiling stage; in particular, when reduction is 'mean', the input predict cannot be an empty tensor.
- Dynamic shape/dynamic rank: this operator supports dynamic-shape and dynamic-rank compilation. In dynamic-shape scenarios, an unknown dimension in a shape is represented by -1; in dynamic-rank scenarios, an unknown rank is represented by -2. After the shape and rank are specified, the above data-type consistency and broadcast constraints must be satisfied.
- Inputs that violate the above constraints are rejected at the graph compilation or tiling stage; the current version does not define an operator-specific error code, and the reported error is a generic error code.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| GE graph mode | - | Invokes the mse_loss_grad operator by constructing a graph through the [operator IR](./op_graph/mse_loss_grad_proto.h) (REG_OP(MseLossGrad)). |
