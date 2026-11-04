# BNInfer

English | [简体中文](./README.md)


## Applicable Products

| Product                                                     | Support |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term> |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                  |    √     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

## Function

- Operator function: in the inference scenario, the operator performs batch normalization on the input `x` using the given mean `mean` and variance `variance`, and obtains the output `y`.

- Formula:

  $$
  y = scale \times \frac{x - mean}{\sqrt{variance + epsilon}} + offset
  $$

  `scale`, `offset`, `mean`, and `variance` are all one-dimensional tensors whose length equals the channel count of the input `x`.

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
  <td>x</td>
  <td>Input</td>
  <td>Input tensor to be normalized. For the channel dimension corresponding to each data format, see Constraints.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NC1HWC0, ND, NCHW, NCDHW, NHWC, NDHWC</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scaling parameter, a one-dimensional tensor with the shape [C].</td>
      <td>FLOAT</td>
      <td>ND, NC1HWC0, NCHW, NCDHW, NHWC</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Bias parameter, a one-dimensional tensor with the shape [C].</td>
      <td>FLOAT</td>
      <td>ND, NC1HWC0, NCHW, NCDHW, NHWC</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Mean used for inference, a one-dimensional tensor with the shape [C].</td>
      <td>FLOAT</td>
      <td>ND, NC1HWC0, NCHW, NCDHW, NHWC</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>Input</td>
      <td>Variance used for inference, a one-dimensional tensor with the shape [C].</td>
      <td>FLOAT</td>
      <td>ND, NC1HWC0, NCHW, NCDHW, NHWC</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Required attribute</td>
      <td>Numerical stability constant added to the variance, used to avoid division by zero.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Normalized output tensor, with the data type, data format, and shape consistent with x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NC1HWC0, ND, NCHW, NCDHW, NHWC, NDHWC</td>
    </tr>
  </tbody></table>

## Constraints

- The data formats in the parameter table are the union of the formats supported by each product and shape mode; for the specific support scope, refer to the product descriptions in this section.
- The data types of scale, offset, mean, and variance must be FLOAT, and all of them are one-dimensional tensors with the shape [C], where C is the channel count of x.
- <term>Ascend 950PR&950DT products</term>:
  - x and y support the ND, NCHW, NCDHW, NHWC, and NDHWC formats, and do not support the NC1HWC0 format.
  - scale, offset, mean, and variance support only the ND format.
  - In the ND format, the rank of x is not less than 2 and the channel dimension is dimension 1; in the NCHW and NCDHW formats, the channel dimension is the C dimension; in the NHWC and NDHWC formats, the channel dimension is the last dimension.
  - In the NCHW and NHWC formats, x must be 4-dimensional; in the NCDHW and NDHWC formats, x must be 5-dimensional.
  - Empty tensors are not supported, and all dimension sizes of x must be greater than 0.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - In the static shape scenario, x, scale, offset, mean, variance, and y support the NC1HWC0 and NCDHW formats.
  - In the dynamic shape scenario, x, scale, offset, mean, variance, and y support the NC1HWC0, NCHW, NHWC, and NCDHW formats.
  - x and y do not support the ND and NDHWC formats.
- <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>:
  - x and y do not support BFLOAT16.
- This operator supports GE graph mode and TensorFlow Parser invocation, and does not provide a public aclnn interface.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_bn_infer](./examples/test_geir_bn_infer.cpp) | Calls the BNInfer operator by building a graph through [operator IR](./op_graph/bn_infer_proto.h). |
