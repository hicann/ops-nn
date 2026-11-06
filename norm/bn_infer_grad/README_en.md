# BNInferGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: computes the backward gradient of the input in Batch Normalization inference mode.
- Formula:

  $$
  x\_backprop = grads \times \frac{scale}{\sqrt{batch\_variance + epsilon}}
  $$

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 520px">
<col style="width: 300px">
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
    <td>grads</td>
    <td>Input</td>
    <td>The upstream gradient, supporting 2D to 5D tensors.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>NHWC, NCHW, NC1HWC0</td>
  </tr>
  <tr>
    <td>scale</td>
    <td>Input</td>
    <td>The scaling parameter. It must be a 1D tensor, with a length equal to the C dimension of grads.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>batch_variance</td>
    <td>Input</td>
    <td>The variance used in the inference stage. It must be a 1D tensor, with a length equal to the C dimension of grads.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x_backprop</td>
    <td>Output</td>
    <td>The input gradient, with the same shape and data type as grads.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>NHWC, NCHW, NC1HWC0</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>Optional attribute</td>
    <td>A small constant that prevents division by zero. Default value: 0.0001.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- The logical C dimension of grads (the 2nd dimension of NCHW, the last dimension of NHWC) must equal the lengths of scale and batch_variance.
- scale and batch_variance support only FLOAT.
- Atlas A3 series products and Atlas A2 series products: the data format of parameters grads and x_backprop supports NC1HWC0, NCHW, and NHWC.
- Ascend 950PR&950DT series products: the data format of parameters grads and x_backprop supports only NCHW and NHWC.
- The shape, data type, and data format of x_backprop match grads.
- epsilon supports only the FLOAT scalar type. It must be greater than 0, with the recommended range [1e-6, 1e-3]; there are no enumeration or step constraints.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Sample code</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE graph mode</td>
    <td><a href="./examples/test_geir_bn_infer_grad.cpp">test_geir_bn_infer_grad</a></td>
    <td>Builds the graph through the <a href="./op_graph/bn_infer_grad_proto.h">operator IR</a> and calls the BNInferGrad operator.</td>
  </tr>
</tbody></table>
