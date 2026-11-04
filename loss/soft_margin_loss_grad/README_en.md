# SoftMarginLossGrad

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

- Operator function: computes the gradient value of the SoftMarginLoss backpropagation.
- Formula:

  The forward computation formula of SoftMarginLoss is as follows:

  $$
  loss_i = ln(1 + exp(-predict_i \cdot label_i))
  $$

  Taking the partial derivative with respect to $predict$ gives the backward gradient:

  $$
  gradient = cof \cdot \frac{-label \cdot exp(-predict \cdot label)}{1 + exp(-predict \cdot label)} \cdot dout
  $$

  When reduction is mean, $cof = 1/N$; otherwise, $cof = 1.0$.

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
      <td>predict</td>
      <td>Input</td>
      <td>Prediction value of the previous layer in the forward pass of the network. The shape is broadcast with label and dout.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>Input</td>
      <td>Label value of the sample. The shape is broadcast with predict and dout.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dout</td>
      <td>Input</td>
      <td>Gradient value of the previous step in the backpropagation of the network. The shape is broadcast with predict and label.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradient</td>
      <td>Output</td>
      <td>Stores the gradient computation result. The data type is consistent with the inputs, and the shape is the shape after predict, label, and dout are broadcast with one another.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Attribute</td>
      <td>Denotes the reduce operation applied to the backward gradient computation result. The value is "none"/"mean"/"sum", and the default value is "mean".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Graph mode supports 0D to 25D inputs, and the aclnn interface supports 0D to 8D inputs.

- Data type constraint: when the operator is called in GE graph mode, the data types of predict, label, dout, and gradient must be the same (all one of FLOAT16, FLOAT32, or BFLOAT16); when the operator is called through the aclnn API, the input data types must satisfy the mutual conversion relationship, the computation automatically promotes the inputs to FP32, and the data type of the output gradient is consistent with out.
- Shape broadcast is supported: the shapes of predict, label, and dout are broadcast with one another (1s are prepended to the lower dimensions, and the non-1 values of each dimension must be consistent; otherwise, broadcast is not supported), and the output shape is the result of broadcasting the three shapes.
- Graph mode supports dynamic shapes and dynamic ranks.
- Empty tensors: the parameter validation and broadcast rules must still be satisfied; when self is empty, the interface directly returns success. When other inputs are empty, a direct short circuit at the ACLNN layer is not guaranteed, and the output shape must still conform to the broadcast result.
- When reduction is "mean", N in graph mode is the element count of the input predict; N in the aclnn interface is the element count after broadcast.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_soft_margin_loss_grad.cpp](examples/arch35/test_aclnn_soft_margin_loss_grad.cpp) | Calls the SoftMarginLossGrad operator through the [aclnnSoftMarginLossBackward](docs/aclnnSoftMarginLossBackward_en.md) interface. |
| GE graph mode  | [test_geir_soft_margin_loss_grad.cpp](examples/arch35/test_geir_soft_margin_loss_grad.cpp) | For compilation and running, see [Quick Operator Invocation (GE Graph Mode)](../../docs/zh/invocation/quick_op_invocation.md#ge图模式). The operator is called in GE graph mode; for the operator IR prototype definition, see [soft_margin_loss_grad_proto.h](op_graph/soft_margin_loss_grad_proto.h). |
