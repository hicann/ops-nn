# BNTrainingUpdateV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

> Note: This table is determined by the registration/delivery support scope of the operator on each product — the Ascend 950PR&950DT products use the arch35 implementation of this repository (ND, rank 2~8, see the product limits in Parameters and Constraints); the other products use the TBE implementation delivered by CANN (NC1HWC0/NCDHW/NCHW/NDC1HWC0, rank 4~6). The data type BFLOAT16 is supported only by the Ascend 950PR&950DT products, Atlas A2 products, and Atlas A3 products; the Atlas training products, Atlas inference products, and Atlas 200I/500 A2 inference products support only FLOAT16/FLOAT32.

## Function

- Operator function: the update stage of the forward pass of batch normalization training (Batch Normalization Training Update V2). Given the per-channel sum/square_sum produced by BNTrainingReduce, combined with the scaling factor scale and the bias offset, this operator performs the batch normalization affine transformation on input x and outputs the normalized result y; it also outputs the statistics of the current batch, batch_mean/batch_variance (for use by backpropagation). It applies to scenarios that do not include a moving average update, and it is used together with [BNTrainingReduce](../bn_training_reduce/README.md).

- Formula:

  Assume that the shape of x is [N, C, R...] (dim0 is N, dim1 is C, and the trailing dimensions are flattened into R), and num = N * R:

  $$
  batch\_mean = {sum\over num}
  $$

  $$
  batch\_variance = {square\_sum\over num} - batch\_mean^2
  $$

  $$
  y = {scale\over\sqrt {batch\_variance + ε}} * x + (offset - {scale * batch\_mean\over\sqrt {batch\_variance + ε}})
  $$

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
      <td><ul><li>Denotes the input tensor to be normalized, corresponding to <code>x</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is [N, C, R...], 2~8 dimensions are supported, dim0 is N, dim1 is C, and the trailing dimensions are flattened into the normalization axis R.</li><li>Other products: 4/5/6 dimensions (5HD/6HD internal layout, or the 4-dimension NCHW special case [N,C,1,W] that goes through the dynamic branch).</li><li>Empty tensors are not supported (each dimension must be a positive number).</li><li>fp16/bf16 inputs are upconverted to fp32 for computation inside the operator and written back with a single rounding.</li></ul></td>
      <td>FLOAT32, FLOAT16; BFLOAT16 is supported only by the Ascend 950PR&950DT products, Atlas A2 products, and Atlas A3 products</td>
      <td>Ascend 950PR&950DT products: ND (including the NCDHW label; the layout is the same in the 5D case); other products: NC1HWC0/NCDHW/NCHW/NDC1HWC0 (the Atlas inference products support only NC1HWC0/NDC1HWC0)</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>Input</td>
      <td><ul><li>Denotes the per-channel sum of x over the N and R dimensions, that is, the sum output of BNTrainingReduce, corresponding to <code>sum</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is [C] (recommended; validated by the number of elements, so shapes with the same number of elements such as [1,C] are also accepted); the number of elements must equal dim1 (C) of x.</li><li>Other products: the shape has the same rank as x (for example, [1,C1,1,1,C0] under NC1HWC0, where the number of elements equals C1*C0).</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: the same layout as x</td>
    </tr>
    <tr>
      <td>square_sum</td>
      <td>Input</td>
      <td><ul><li>Denotes the per-channel sum of squares of x over the N and R dimensions, that is, the square_sum output of BNTrainingReduce, corresponding to <code>square_sum</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is [C] (recommended; validated by the number of elements, so shapes with the same number of elements such as [1,C] are also accepted); the number of elements must equal dim1 (C) of x.</li><li>Other products: the shape has the same rank as x (for example, [1,C1,1,1,C0] under NC1HWC0, where the number of elements equals C1*C0).</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: the same layout as x</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td><ul><li>Denotes the per-channel scaling factor, corresponding to <code>scale</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is [C] (recommended; validated by the number of elements, so shapes with the same number of elements such as [1,C] are also accepted); the number of elements must equal dim1 (C) of x.</li><li>Other products: the shape has the same rank as x (for example, [1,C1,1,1,C0] under NC1HWC0, where the number of elements equals C1*C0).</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: the same layout as x</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td><ul><li>Denotes the per-channel scaling bias, corresponding to <code>offset</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is [C] (recommended; validated by the number of elements, so shapes with the same number of elements such as [1,C] are also accepted); the number of elements must equal dim1 (C) of x.</li><li>Other products: the shape has the same rank as x (for example, [1,C1,1,1,C0] under NC1HWC0, where the number of elements equals C1*C0).</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: the same layout as x</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Attribute</td>
      <td><ul><li>Denotes the small value added to batch_variance to ensure numerical stability, corresponding to <code>ε</code> in the formula. Required attribute.</li><li>The value must be greater than 0; if 0 or a negative value is passed in and batch_variance is negative, the host does not validate it and silently outputs NaN.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Denotes the normalized affine result, corresponding to <code>y</code> in the formula.</li><li>The shape and data type are both consistent with <code>x</code>.</li></ul></td>
      <td>Consistent with x</td>
      <td>Consistent with x</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>Output</td>
      <td><ul><li>Denotes the per-channel mean of the current batch, corresponding to <code>batch_mean</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is consistent with scale (that is, [C] when scale is [C]).</li><li>Other products: the same rank as x.</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: the same layout as x</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>Output</td>
      <td><ul><li>Denotes the per-channel variance of the current batch, corresponding to <code>batch_variance</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is consistent with scale (that is, [C] when scale is [C]).</li><li>Other products: the same rank as x.</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: the same layout as x</td>
    </tr>
  </tbody></table>

## Constraints

**Ascend 950PR&950DT products:**

- Only the ND format is supported (dim0=N, dim1=C, and the trailing dimensions are the normalization axis R, rank 2~8; in graph mode, NCHW/NCDHW labels are normalized and delivered by the framework or passed through directly, where NCDHW is a 5D label and its memory layout is the same as ND).
- The number of dimensions of x must be ≥2; x is FLOAT16/FLOAT32/BFLOAT16.
- sum/square_sum/scale/offset are always FLOAT32, and the number of elements must equal dim1 (C) of x (the recommended shape is [C], validated by the number of elements).
- The shapes of batch_mean/batch_variance are consistent with scale.
- Empty tensors are not supported: the operator refuses to execute when any dimension of x is 0 (num=N*R cannot be defined as the denominator).

**Other products (Atlas A2 products, Atlas A3 products, Atlas training products, Atlas inference products):**

- The data format is NC1HWC0/NCDHW/NCHW/NDC1HWC0 (the Atlas inference products support only NC1HWC0/NDC1HWC0), and x has 4/5/6 dimensions (or the 4-dimension NCHW special case [N,C,1,W]).
- The statistics sum/square_sum/scale/offset have the same rank as x and are always FLOAT32; under NC1HWC0/NDC1HWC0, C is aligned to C0=16, and the number of elements of the statistics equals C1*C0.
- Empty tensors are not supported (as declared in the proto of each product).

**General:** the value of epsilon must be greater than 0 (the host does not validate it; see Parameters).

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_bn_training_update_v2](./examples/arch35/test_geir_bn_training_update_v2.cpp) | Invokes the BNTrainingUpdateV2 operator by constructing the graph through the [operator IR](op_graph/bn_training_update_v2_proto.h) (including two groups of shape/epsilon test cases). |
