# BNTrainingUpdateV3

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

> Note: This table is determined by the registration and delivery support scope of the operator on each product — the Ascend 950PR&950DT products use the arch35 implementation in this repository (ND/NHWC, rank 2~8, see the product limits in Parameters and Constraints); the other products use the TBE implementation delivered by CANN (4-dimensional NHWC/NCHW input, internal NC1HWC0/NDC1HWC0 layout). The BFLOAT16 data type is supported only by the Ascend 950PR&950DT products, Atlas A2 products, and Atlas A3 products; the Atlas training products, Atlas inference products, and Atlas 200I/500 A2 inference products support only FLOAT16/FLOAT32.

## Function

- Operator function: the update stage of the forward pass of batch normalization training (Batch Normalization Training Update V3). Given the per-channel sum/square_sum produced by BNTrainingReduce, and combined with the scaling factor scale and the bias offset, the operator performs the batch normalization affine transformation on the input x and outputs the normalized result y. It also outputs the statistics of the current batch, batch_mean/batch_variance (batch_variance is the unbiased estimate), and the intermediate values used for backpropagation, reserve_1 (save_mean) and reserve_2 (save_variance, biased variance). The operator applies to scenarios without moving average updates and is used together with [BNTrainingReduce](../bn_training_reduce/README.md).

- Formula:

  Let the shape of x be [N, C, R...] (dim0 is N, dim1 is C, and the trailing dimensions are flattened into R); num = N * R:

  $$
  save\_mean = {sum\over num}
  $$

  $$
  save\_variance = {square\_sum\over num} - save\_mean^2
  $$

  $$
  batch\_mean = save\_mean
  $$

  $$
  batch\_variance = save\_variance * {num\over num - 1} \quad (0\ when\ num=1)
  $$

  $$
  reserve\_1 = save\_mean
  $$

  $$
  reserve\_2 = save\_variance
  $$

  $$
  y = {scale\over\sqrt {save\_variance + ε}} * x + (offset - {scale * save\_mean\over\sqrt {save\_variance + ε}})
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
      <td><ul><li>Denotes the input tensor to be normalized, corresponding to <code>x</code> in the formula.</li><li>Ascend 950PR&950DT products: the ND layout shape is [N, C, R...], supporting 2 to 8 dimensions; dim0 is N, dim1 is C, and the trailing dimensions are flattened into the normalization axis R; the NHWC layout supports any rank≥2, with C as the last dimension.</li><li>Other products: 4 dimensions (source format NHWC or NCHW, computed internally in the 5HD/6HD layout).</li><li>Empty tensors are not supported (all dimensions must be positive).</li><li>fp16/bf16 inputs are computed in fp32 inside the operator and written back with a single rounding.</li></ul></td>
      <td>FLOAT32, FLOAT16; BFLOAT16 is supported only by the Ascend 950PR&950DT products, Atlas A2 products, and Atlas A3 products</td>
      <td>Ascend 950PR&950DT products: ND/NHWC; other products: NC1HWC0/NCHW/NDC1HWC0/NHWC</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>Input</td>
      <td><ul><li>Denotes the per-channel sum of x over the N and R dimensions, that is, the sum output of BNTrainingReduce, corresponding to <code>sum</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is [C] (recommended; validated by element count, so shapes with the same element count, such as [1,C], are also accepted); the element count must equal the channel count C (dim1 of x under the ND/NCHW layout, and the last dimension under the NHWC layout).</li><li>Other products: the shape has the same rank as x (for example, [1,C1,1,1,C0] under NC1HWC0, with the element count equal to C1*C0).</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: same layout as x</td>
    </tr>
    <tr>
      <td>square_sum</td>
      <td>Input</td>
      <td><ul><li>Denotes the per-channel sum of squares of x over the N and R dimensions, that is, the square_sum output of BNTrainingReduce, corresponding to <code>square_sum</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is [C] (recommended; validated by element count, so shapes with the same element count, such as [1,C], are also accepted); the element count must equal the channel count C (dim1 of x under the ND/NCHW layout, and the last dimension under the NHWC layout).</li><li>Other products: the shape has the same rank as x (for example, [1,C1,1,1,C0] under NC1HWC0, with the element count equal to C1*C0).</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: same layout as x</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td><ul><li>Denotes the per-channel scaling factor, corresponding to <code>scale</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is [C] (recommended; validated by element count, so shapes with the same element count, such as [1,C], are also accepted); the element count must equal the channel count C (dim1 of x under the ND/NCHW layout, and the last dimension under the NHWC layout).</li><li>Other products: the shape has the same rank as x (for example, [1,C1,1,1,C0] under NC1HWC0, with the element count equal to C1*C0).</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: same layout as x</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td><ul><li>Denotes the per-channel scaling bias, corresponding to <code>offset</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is [C] (recommended; validated by element count, so shapes with the same element count, such as [1,C], are also accepted); the element count must equal the channel count C (dim1 of x under the ND/NCHW layout, and the last dimension under the NHWC layout).</li><li>Other products: the shape has the same rank as x (for example, [1,C1,1,1,C0] under NC1HWC0, with the element count equal to C1*C0).</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: same layout as x</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Required attribute</td>
      <td><ul><li>Denotes the small value added to save_variance to ensure numerical stability, corresponding to <code>ε</code> in the formula.</li><li>The value must be greater than 0; if 0 or a negative value is passed in and save_variance is negative, the host does not validate it and the operator silently outputs NaN.</li><li>On this product, this attribute is required and has no default value (the TBE side of the other products defaults to 1e-7).</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>before_split_ori_shape / before_split_ori_format</td>
      <td>N/A (a difference between the Ascend 950PR&950DT products and the Atlas A2 products/Atlas A3 products; historical background note)</td>
      <td><ul><li><strong>Ascend 950PR&950DT products: no such input or attribute exists.</strong> These two parameters are not part of the operator prototype inputs and are strongly related to the FFTS splitting feature — FFTS is a deprecated feature and is neither included in nor supported by the code or prototype of the Ascend 950PR&950DT products.</li><li>Historical background: the Atlas A2 products, Atlas A3 products, and other products register them as optional attributes (LIST_LIST_INT/LIST_INT, defaults [[]]/[]) to compute num from the original shape/format before splitting in BN FFTS splitting scenarios; this feature is deprecated. The Ascend 950PR&950DT products compute num from the actual input shape, and passing these attributes has no effect (the prototype does not register them, and the GE side ignores them as unknown attributes).</li></ul></td>
      <td>N/A</td>
      <td>N/A</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Denotes the normalized affine result, corresponding to <code>y</code> in the formula.</li><li>Both the shape and the data type are the same as those of <code>x</code>.</li></ul></td>
      <td>Same as x</td>
      <td>Same as x</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>Output</td>
      <td><ul><li>Denotes the per-channel mean of the current batch, corresponding to <code>batch_mean</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is the same as scale ([C] when scale is [C]).</li><li>Other products: same rank as x.</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: same layout as x</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>Output</td>
      <td><ul><li>Denotes the per-channel variance of the current batch (unbiased estimate, that is, save_variance * num/(num-1)), corresponding to <code>batch_variance</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is the same as scale ([C] when scale is [C]).</li><li>Other products: same rank as x.</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: same layout as x</td>
    </tr>
    <tr>
      <td>reserve_1</td>
      <td>Output</td>
      <td><ul><li>Denotes the per-channel mean of the current batch (save_mean, the same as batch_mean), used for backpropagation, corresponding to <code>reserve_1</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is the same as scale ([C] when scale is [C]).</li><li>Other products: same rank as x.</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: same layout as x</td>
    </tr>
    <tr>
      <td>reserve_2</td>
      <td>Output</td>
      <td><ul><li>Denotes the per-channel variance of the current batch (save_variance, biased estimate), used for backpropagation, corresponding to <code>reserve_2</code> in the formula.</li><li>Ascend 950PR&950DT products: the shape is the same as scale ([C] when scale is [C]).</li><li>Other products: same rank as x.</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR&950DT products: ND; other products: same layout as x</td>
    </tr>
  </tbody></table>

## Constraints

**Ascend 950PR&950DT products:**

- Supports the ND and NCHW formats (tiling accepts both; dim0=N, dim1=C, and the trailing dimensions form the normalization axis R, rank 2~8; in graph mode the NCHW tag is normalized and delivered by the framework, with the same layout).
- Supports the NHWC format (x supports any rank≥2, C is the last dimension, and num=numel/C; the statistics are still validated by element count=C). Under NHWC, the channel count C has no upper limit: when C is a multiple of 64, the operator processes data in a streaming manner by vector windows; when C is not a multiple of 64 and the whole row fits within the budget, the operator processes whole-row tiles; for extremely large C, the operator processes data in c-window streaming mode (coefficients are rebuilt per window, and UB usage is independent of C). The source formats NC1HWC0/NDC1HWC0 are rejected by tiling on this product (GRAPH_FAILED) — BN graphs delivered in these formats on the Atlas A2 products and Atlas A3 products must be converted to ND/NCHW/NHWC before migration to the Ascend 950PR&950DT products.
- The number of dimensions of x must be ≥2; x is FLOAT16/FLOAT32/BFLOAT16.
- The tiling stage requires all shape values to be definite positive values: any dimension of 0 is rejected (empty tensor), and any negative or unknown dynamic dimension is rejected with "dynamic shape dim is not supported in tiling".
- The BN FFTS splitting scenario is not supported (a difference between the Ascend 950PR&950DT products and the Atlas A2 products/Atlas A3 products): before_split_ori_shape/before_split_ori_format are not part of the operator prototype inputs, are strongly related to the FFTS splitting feature, and that feature is deprecated — the code and prototype of the Ascend 950PR&950DT products do not contain these two inputs, and num is computed from the actual input shape. Historical background: the Atlas A2 products and Atlas A3 products register them as optional attributes to compute num from the original shape before splitting; for details, see the row description in the parameter table.
- sum/square_sum/scale/offset are always FLOAT32, and their element count must equal the channel count C (dim1 of x under the ND/NCHW layout, and the last dimension of x under the NHWC layout; the recommended shape is [C], validated by element count).
- The shapes of batch_mean/batch_variance/reserve_1/reserve_2 are the same as scale.
- Empty tensors are not supported: the operator rejects execution when any dimension of x is 0 (num=N*R cannot be defined as a denominator).

**Other products (Atlas A2 products, Atlas A3 products, Atlas training products, Atlas inference products):**

- The data formats are NC1HWC0/NCHW/NDC1HWC0/NHWC, and x has 5/6 dimensions.
- The statistics sum/square_sum/scale/offset have the same rank as x and are always FLOAT32; under NC1HWC0/NDC1HWC0, C is aligned to C0=16, and the element count of the statistics equals C1*C0.
- Empty tensors are not supported (declared identically in the proto of each product).
- The optional attributes before_split_ori_shape (listListInt, default [[]]) and before_split_ori_format (listInt, default []) are legacy registrations of the BN FFTS splitting scenario (a deprecated feature); in that scenario, num is computed from the original shape before splitting.

**General:** the value of epsilon must be greater than 0 (the host does not validate it; see Parameters).

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_bn_training_update_v3](./examples/arch35/test_geir_bn_training_update_v3.cpp) | Calls the BNTrainingUpdateV3 operator by building a graph through [operator IR](op_graph/bn_training_update_v3_proto.h) (including two sets of shape/epsilon test cases). |
