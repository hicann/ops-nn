# BNTrainingUpdate

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

>Note: This table is determined by the registration/delivery support scope of the operator on each product — the <term>Ascend 950PR&950DT products</term> use the arch35 implementation of this repository (4-dim NCHW/NHWC and 5-dim NCDHW, see the product restrictions in the Parameters and Constraints sections); the other products use the TBE implementation delivered by CANN (NC1HWC0/NDC1HWC0/NCDHW, and NCHW/NHWC under dynamic templates).

## Function

- Operator function: the update stage of the forward pass of batch normalization training (including the moving average update). Given the per-channel sum/square_sum produced by BNTrainingReduce, combined with the scaling factor scale and the bias offset, the operator performs the batch normalization affine transformation on the input x and outputs the normalized result y; at the same time, it updates the running mean/variance with factor as the weight, and outputs the statistics of the current batch, batch_mean/batch_variance (biased variance). Use it together with [BNTrainingReduce](../bn_training_reduce/README.md).

- Formula:

  Assume that the shape of x is [N, C, H, W] and num = N \* H \* W (number of samples). batch_variance is computed as a biased estimate, and the Bessel correction is used only for the running variance update:

  $$
  batch\_mean = {sum\over num}
  $$

  $$
  batch\_variance = {square\_sum\over num} - batch\_mean^2
  $$

  $$
  y = {scale\over\sqrt {batch\_variance + \epsilon}} * x + (offset - {scale * batch\_mean\over\sqrt {batch\_variance + \epsilon}})
  $$

  $$
  mean = factor * batch\_mean + (1 - factor) * mean
  $$

  $$
  variance = factor * {num\over num - 1} * batch\_variance + (1 - factor) * variance
  $$

  When num=1, the Bessel correction term is 0.

## Parameters

<table style="table-layout: fixed; width: 1576px">
<colgroup>
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
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>Input</td>
<td><ul><li>The input tensor to be normalized, corresponding to <code>x</code> in the formula. It is a 4-dim (NCHW/NHWC) or 5-dim (NCDHW) tensor.</li><li>For the empty tensor semantics when any dimension is 0 (including C=0), see the Constraints section.</li><li>fp16/bf16 inputs are upcast to fp32 for computation inside the operator and converted back to the original type when written back.</li></ul></td>
<td>FLOAT16, FLOAT32, BFLOAT16</td>
<td>NCHW, NHWC, NCDHW, NC1HWC0, NDC1HWC0</td>
</tr>
<tr>
<td>sum</td>
<td>Input</td>
<td><ul><li>Per-channel sum of x over the N, H, and W dimensions, that is, the sum output of BNTrainingReduce, corresponding to <code>sum</code> in the formula.</li><li>For the shape constraints, see the Constraints section.</li></ul></td>
<td>FLOAT32</td>
<td>ND, NC1HWC0, NDC1HWC0, NCDHW, NCHW, NHWC</td>
</tr>
<tr>
<td>square_sum</td>
<td>Input</td>
<td><ul><li>Per-channel sum of squares of x over the N, H, and W dimensions, that is, the square_sum output of BNTrainingReduce, corresponding to <code>square_sum</code> in the formula.</li><li>The shape constraints are the same as those of <code>sum</code>.</li></ul></td>
<td>FLOAT32</td>
<td>ND, NC1HWC0, NDC1HWC0, NCDHW, NCHW, NHWC</td>
</tr>
<tr>
<td>scale</td>
<td>Input</td>
<td><ul><li>Per-channel scaling factor, corresponding to <code>scale</code> in the formula.</li><li>The shape constraints are the same as those of <code>sum</code>.</li></ul></td>
<td>FLOAT32</td>
<td>ND, NC1HWC0, NDC1HWC0, NCDHW, NCHW, NHWC</td>
</tr>
<tr>
<td>offset</td>
<td>Input</td>
<td><ul><li>Per-channel scaling bias, corresponding to <code>offset</code> in the formula.</li><li>The shape constraints are the same as those of <code>sum</code>.</li></ul></td>
<td>FLOAT32</td>
<td>ND, NC1HWC0, NDC1HWC0, NCDHW, NCHW, NHWC</td>
</tr>
<tr>
<td>mean</td>
<td>Input</td>
<td><ul><li>Running mean before the update, corresponding to <code>mean</code> in the formula.</li><li>The shape constraints are the same as those of <code>sum</code>.</li></ul></td>
<td>FLOAT32</td>
<td>ND, NC1HWC0, NDC1HWC0, NCDHW, NCHW, NHWC</td>
</tr>
<tr>
<td>variance</td>
<td>Input</td>
<td><ul><li>Running variance before the update, corresponding to <code>variance</code> in the formula.</li><li>The shape constraints are the same as those of <code>sum</code>.</li></ul></td>
<td>FLOAT32</td>
<td>ND, NC1HWC0, NDC1HWC0, NCDHW, NCHW, NHWC</td>
</tr>
<tr>
<td>factor</td>
<td>Required attribute</td>
<td><ul><li>Weight of the new batch statistics in the running statistics update, corresponding to <code>factor</code> in the formula.</li><li>For the value constraints, see the Constraints section.</li></ul></td>
<td>FLOAT32</td>
<td>-</td>
</tr>
<tr>
<td>epsilon</td>
<td>Required attribute</td>
<td><ul><li>Small value added to batch_variance to ensure numerical stability, corresponding to <code>ε</code> in the formula.</li><li>For the value constraints, see the Constraints section.</li></ul></td>
<td>FLOAT32</td>
<td>-</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td><ul><li>Normalized affine result, corresponding to <code>y</code> in the formula.</li><li>The shape and data type are both the same as those of <code>x</code>.</li></ul></td>
<td>Same as x</td>
<td>Same as x</td>
</tr>
<tr>
<td>mean</td>
<td>Output</td>
<td><ul><li>Running mean after the update, corresponding to <code>mean</code> in the formula.</li><li>For the shape constraints, see the Constraints section.</li></ul></td>
<td>FLOAT32</td>
<td>ND, NC1HWC0, NDC1HWC0, NCDHW, NCHW, NHWC</td>
</tr>
<tr>
<td>variance</td>
<td>Output</td>
<td><ul><li>Running variance after the update (after the Bessel correction), corresponding to <code>variance</code> in the formula.</li><li>The shape constraints are the same as those of the output <code>mean</code>.</li></ul></td>
<td>FLOAT32</td>
<td>ND, NC1HWC0, NDC1HWC0, NCDHW, NCHW, NHWC</td>
</tr>
<tr>
<td>batch_mean</td>
<td>Output</td>
<td><ul><li>Per-channel mean of the current batch, corresponding to <code>batch_mean</code> in the formula.</li><li>The shape constraints are the same as those of the output <code>mean</code>.</li></ul></td>
<td>FLOAT32</td>
<td>ND, NC1HWC0, NDC1HWC0, NCDHW, NCHW, NHWC</td>
</tr>
<tr>
<td>batch_variance</td>
<td>Output</td>
<td><ul><li>Per-channel biased variance of the current batch, corresponding to <code>batch_variance</code> in the formula.</li><li>The shape constraints are the same as those of the output <code>mean</code>.</li></ul></td>
<td>FLOAT32</td>
<td>ND, NC1HWC0, NDC1HWC0, NCDHW, NCHW, NHWC</td>
</tr>
</tbody></table>

## Constraints

- The data formats in the parameter table are the union of the formats supported by all products. For the specific support scope, refer to the product descriptions in this section.

**<term>Ascend 950PR&950DT products</term>:**

- x is a 4-dim (NCHW/NHWC) or 5-dim (NCDHW) tensor, and supports the three data formats NCHW, NHWC, and NCDHW; under NCHW/NHWC the channel dimension is dim 1/dim 3, and under NCDHW the channel dimension is dim 1. If the format and the number of dimensions do not match, the operator rejects execution on the host side.
- x supports FLOAT16/FLOAT32/BFLOAT16; sum/square_sum/scale/offset/mean/variance are always FLOAT32 in ND format, with shape [C], and the number of elements must equal the channel dimension C of x.
- The value range of factor is [0.0, 1.0], and epsilon must be greater than 0; when the value is out of range, the operator rejects execution on the host side.
- The empty tensor semantics align with TensorFlow FusedBatchNormV3: when any non-channel dimension is 0 and C>0, the input is valid — y is empty, the updated mean/variance are all NaN, and batch_mean/batch_variance are all 0; C=0 (empty channel vector) is not supported, and the host side rejects execution.
- The normalization denominator is sqrt(|batch_variance + epsilon|), and batch_variance is not clamped to non-negative values; when the batch_variance output is a tiny negative value caused by cancellation errors, it is output as is. The fp16 output y is saturated to ±65504, with NaN set to 0, before being written back.

**Other products (<term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Atlas training products</term>, <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>):**

- The data formats are NC1HWC0/NDC1HWC0/NCDHW (NCHW/NHWC are supported under dynamic templates), and x is a 4-dim tensor.
- The statistics sum/square_sum/scale/offset/mean/variance have the same rank and layout as x and are always FLOAT32; under NC1HWC0/NDC1HWC0, C is aligned to C0=16, and the number of elements of the statistics equals C1*C0.
- Each dimension must be a positive number (the proto declaration does not support empty tensors).

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_bn_training_update](./examples/test_geir_bn_training_update.cpp) | Calls the BNTrainingUpdate operator by constructing a graph through the [operator IR](op_graph/bn_training_update_proto.h) (static shape case, verifying the values of all 5 outputs). |
| GE graph mode | [test_geir_bn_training_update_dynamic](./examples/test_geir_bn_training_update_dynamic.cpp) | Graph construction cases with dynamic shape (-1) and dynamic rank (-2). The same Session runs 3 groups of shapes consecutively, verifying the values of all 5 outputs. |
