# BNTrainingUpdateGrad

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

## Function

- Description: The update-grad stage of batch normalization training backward (Batch Normalization Training Update Grad). Given the upstream gradient grads, the forward input x, and the per-channel statistics batch_mean/batch_variance saved in the forward pass, it performs per-channel reduction over the batch and spatial dimensions (N and the trailing dimensions R), and outputs the gradients diff_scale/diff_offset of the affine parameters scale/offset for use in updating the scale/offset parameters (the gradient of x is computed separately by BNTrainingReduceGrad). It is used together with the update stage of batch normalization training forward ([BNTrainingUpdateV3](../bn_training_update_v3/README.md)).

- Formula:

  Assume that grads/x have the ND layout [N, C, R...] (dim0 is N, dim1 is C, and the trailing dimensions are flattened into the normalization axis R); the reduction axes are the N dimension and all R dimensions (in the NHWC layout, C is the last dimension, and the reduction axes are all leading dimensions N·H·W):

  $$
  rstd = {1\over\sqrt {batch\_variance + ε}}
  $$

  $$
  diff\_scale = \sum_{n,r} grads * (x - batch\_mean) * rstd
  $$

  $$
  diff\_offset = \sum_{n,r} grads
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
      <td>grads</td>
      <td>Input</td>
      <td><ul><li>Denotes the upstream gradient (the gradient of the loss function with respect to the batch normalization output y), corresponding to <code>grads</code> in the formula.</li><li>Ascend 950PR&950DT products: in the ND layout, the shape is [N, C, R...] (at least 2 dimensions are supported; dim0 is N, dim1 is C, and the trailing dimensions are flattened into the normalization axis R; NCDHW is a 5D label whose memory layout is the same as ND); in the NHWC layout, C is the last dimension (the shape can be of any rank ≥ 2, and the leading dimensions are flattened into the normalization axis). For other products, the rank is fixed by the input format (4D/5D/6D), and the channel axis follows the format.</li><li>Empty tensors are not supported (all dimensions must be positive).</li><li>fp16/bf16 inputs are converted to fp32 for computation inside the operator.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW, NCDHW, NHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td><ul><li>Denotes the input tensor of forward propagation, corresponding to <code>x</code> in the formula.</li><li>The shape, data type, and layout format are all consistent with those of <code>grads</code>.</li><li>fp16/bf16 inputs are converted to fp32 for computation inside the operator.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW, NCDHW, NHWC</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>Input</td>
      <td><ul><li>Denotes the per-channel mean of x, corresponding to <code>batch_mean</code> in the formula. It is the statistics saved in the forward pass (corresponding to the reserve_1 output of <a href="../bn_training_update_v3/README.md">BNTrainingUpdateV3</a>, that is, save_mean; in the TF FusedBatchNormGrad scenario, it corresponds to reserve_space_1), and is not the batch_mean output of V3.</li><li>The shape is [C], and the number of elements must equal the channel count C of grads (dim1 for ND, and the last dimension for NHWC).</li><li>Always FLOAT32.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>Input</td>
      <td><ul><li>Denotes the per-channel <code>biased variance save_variance</code> of x (the statistics saved in the forward pass, corresponding to the reserve_2 output of <a href="../bn_training_update_v3/README.md">BNTrainingUpdateV3</a>; in the TF FusedBatchNormGrad scenario, it corresponds to reserve_space_2), <strong>not</strong> the batch_variance output of V3 (the unbiased variance containing the num/(num-1) correction) — wiring the unbiased variance makes the gradients systematically smaller by sqrt((num-1)/num).</li><li>The shape is [C], and the number of elements must equal the channel count C of grads.</li><li>Always FLOAT32.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes a small value added to batch_variance to ensure numerical stability, corresponding to <code>ε</code> in the formula.</li><li>Default value: 0.0001.</li><li>A non-negative value is recommended: the host does not validate the sign, and the square root outputs NaN when batch_variance+ε is negative.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diff_scale</td>
      <td>Output</td>
      <td><ul><li>Denotes the gradient of the per-channel scaling factor scale, corresponding to <code>diff_scale</code> in the formula.</li><li>The shape is consistent with that of batch_mean (the number of elements is C; infershape copies the shape of batch_mean as is, and when [1,C,1,1] is passed, the output has the same shape).</li><li>Always FLOAT32.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diff_offset</td>
      <td>Output</td>
      <td><ul><li>Denotes the gradient of the per-channel bias offset, corresponding to <code>diff_offset</code> in the formula.</li><li>The shape is consistent with that of batch_mean (the number of elements is C).</li><li>Always FLOAT32.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- <term>Ascend 950PR&950DT products</term>: two layout types are supported, namely ND (dim0=N, dim1=C, and the trailing dimensions are the normalization axis R; in graph mode, NCHW/NCDHW labels are either normalized and delivered by the framework or passed through directly; NCDHW is a 5D label whose memory layout is the same as ND, with consistent layouts) and NHWC (C is the last dimension, the leading dimensions N·H·W are flattened into the normalization axis, any rank ≥ 2, and C has no upper limit). Although the OpDef declares dynamic shape/rank support (infershape exits early for unknown ranks), tiling requires all shape values to be definite positive values: if any dimension is negative (including unknown dynamic dimensions), the operator is rejected with the error "dynamic shape dim is not supported in tiling", and the shape must be materialized into a static shape before graph building.
- **Other products (<term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, and so on)**: grads/x support NCHW/NHWC/NC1HWC0/NCDHW (NDC1HWC0 in the 5D scenario), the rank is fixed by the format (4D/5D/6D), and the channel axis follows the format; the statistics and outputs have the same format as grads. Among them, <term>Atlas A2 products</term> and <term>Atlas A3 products</term> support BFLOAT16; <term>Atlas 200I/500 A2 inference products</term> support only NC1HWC0/NCHW (4D; NHWC/NCDHW/NDC1HWC0 and BFLOAT16 are not supported), and Atlas inference products do not support BFLOAT16.
- The shape, data type, and layout format of x must be consistent with those of grads.
- batch_mean/batch_variance are always FLOAT32, and the number of elements must equal the channel count C of grads (dim1 for ND, and the last dimension for NHWC).
- Empty tensors are not supported: the operator refuses to execute when any dimension of grads is 0. When the reduction axes (N and R dimensions) are empty, the sum can mathematically be defined as 0, but when the channel axis C=0, the number of output elements and the statistics are both undefined, and num (N·R), as the division denominator, cannot be empty — this is consistent with the "Empty tensors are not supported" statement in the protos of the operators of the same family on <term>Atlas A2 products</term> (BNTrainingReduce/BNTrainingUpdate/BNTrainingUpdateV3) (the absence of this sentence in the proto comment of BNTrainingUpdateGrad on that product side is a documentation omission rather than a behavioral difference).
- For <term>Ascend 950PR&950DT products</term> in the ND layout, when C==1 and the total number of elements is ≥ 2^20, or R==1 and C>2048, multi-core splitting is performed internally in an NHWC-isomorphic layout (with unchanged semantics); in the NHWC scenario with small C and many rows, cross-core partial sums are merged with floating-point atomic addition, and the output is the floating-point sum of the partial sums of all cores (the order is not fixed; the rounding error of floating-point atomic addition is on the order of machine precision O(eps), and the number of partial sums equals the number of cores ≤ 64).

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_bn_training_update_grad](./examples/arch35/test_geir_bn_training_update_grad.cpp) | Invokes the BNTrainingUpdateGrad operator by building a graph through the [operator IR](op_graph/bn_training_update_grad_proto.h) (including two groups of shape/epsilon test cases). |
