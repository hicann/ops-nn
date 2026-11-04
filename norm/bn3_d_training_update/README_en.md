# BN3DTrainingUpdate

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: The BN3DTrainingUpdate operator is used for the "update" step in the 3D BatchNorm training flow. The operator receives the per-channel statistics (sum, square_sum) produced by the preceding
  Bn3dTrainingReduce operator, and performs batch normalization on input x with scale and offset to obtain output y. It also computes the save statistics of the current batch (batch_mean, batch_variance) for reuse in backpropagation,
  and updates the running mean and running variance with factor (equivalent to the momentum of PyTorch) as the EMA weight
  (outputs mean and variance, written back in-place to the input mean and variance).

- Formula:

  Let the number of channels be `C = sum.shape[0]` and the number of elements in the reduce domain be `num = x.size / C` (the following formulas all require `C >= 1`;
  when `C == 0`, the expression is `0/0` and undefined, `num` does not participate in the computation, and the empty tensor branch is taken directly; see below). For each channel:

  $$
  \mu = \frac{sum}{num}
  $$

  $$
  \sigma^2_{biased} = \frac{square\_sum}{num} - \mu^2
  $$

  $$
  y = \frac{x - \mu}{\sqrt{\sigma^2_{biased} + \epsilon}} \cdot scale + offset
  $$

  $$
  mean\_out = factor \cdot \mu + (1 - factor) \cdot mean
  $$

  $$
  variance\_out = factor \cdot \sigma^2_{unbiased} + (1 - factor) \cdot variance
  $$

  Where $\sigma^2_{unbiased} = \frac{num}{num - 1} \cdot \sigma^2_{biased}$ (Bessel correction);
  when `num == 1`, the denominator of the Bessel correction is 0, and the unbiased batch variance is explicitly set to 0. In this case, the running variance does not incorporate
  the statistics of this batch and only decays by EMA: $variance\_out = (1 - factor) \cdot variance$, while `batch_variance`
  (biased) is output normally.

  $$
  batch\_mean = \mu, \quad batch\_variance = \sigma^2_{biased}
  $$

  When `C >= 1` and `num == 0` (any non-channel axis of x is 0, that is, an empty batch), the zero-statistics contract is used for the output: both `batch_mean` and
  `batch_variance` are 0, `mean_out = (1 - factor) \cdot mean` and
  `variance_out = (1 - factor) \cdot variance`, and y has no elements to output.

  When `C == 0` (empty channels), x and all 6 statistics are empty tensors (for example, `x=(2,0,1,28)` and each statistic is `(0,)`),
  there are no phase X/phase S units, the kernel gracefully runs as an empty operation in both phases without any memory access, and it produces all-empty outputs: `y` has the same shape as x (including 0 dimensions),
  and the 4 statistics outputs `mean_out`, `variance_out`, `batch_mean`, and `batch_variance` are all `(C,)`, that is, `(0,)`.
  This path is independent of the zero-statistics contract of `num == 0` above (when `C == 0`, `num` is undefined and the zero-statistics contract does not apply).

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
  <td>Source data tensor of 3D batch normalization, corresponding to x in the formula.</td>
  <td>FLOAT16, FLOAT32, BFLOAT16</td>
  <td>NCHW, NCDHW, NHWC, NDHWC, NDC1HWC0</td>
</tr>
<tr>
  <td>sum</td>
  <td>Input</td>
  <td>Per-channel sum produced by the preceding Bn3dTrainingReduce operator. It is a 1D tensor whose length is the number of channels C, corresponding to sum in the formula.</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>square_sum</td>
  <td>Input</td>
  <td>Per-channel square sum produced by the preceding Bn3dTrainingReduce operator. It is a 1D tensor whose length is the number of channels C, corresponding to square_sum in the formula.</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>scale</td>
  <td>Input</td>
  <td>Scale coefficient of batch normalization. It is a 1D tensor whose length is the number of channels C, corresponding to scale in the formula.</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>offset</td>
  <td>Input</td>
  <td>Bias term of batch normalization. It is a 1D tensor whose length is the number of channels C, corresponding to offset in the formula.</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>mean</td>
  <td>Input</td>
  <td>Running mean (historical mean). It is a 1D tensor whose length is the number of channels C. It is written back in-place after the EMA update, corresponding to mean in the formula.</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>variance</td>
  <td>Input</td>
  <td>Running variance (historical variance). It is a 1D tensor whose length is the number of channels C. It is written back in-place after the EMA update, corresponding to variance in the formula.</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>factor</td>
  <td>Attribute</td>
  <td>EMA weight, equivalent to the momentum of PyTorch. The recommended value range is [0, 1].</td>
  <td>FLOAT32</td>
  <td>-</td>
</tr>
<tr>
  <td>epsilon</td>
  <td>Attribute</td>
  <td>Small constant that prevents division by zero in the normalization denominator. It participates in sqrt(var + epsilon) and must not be less than 0.</td>
  <td>FLOAT32</td>
  <td>-</td>
</tr>
<tr>
  <td>y</td>
  <td>Output</td>
  <td>Batch normalization output, corresponding to y in the formula.</td>
  <td>FLOAT16, FLOAT32, BFLOAT16</td>
  <td>NCHW, NCDHW, NHWC, NDHWC, NDC1HWC0</td>
</tr>
<tr>
  <td>mean</td>
  <td>Output</td>
  <td>Running mean after the EMA update, written back in-place to the input mean, corresponding to mean_out in the formula.</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>variance</td>
  <td>Output</td>
  <td>Running variance after the EMA update (the merged value is the unbiased batch variance), written back in-place to the input variance, corresponding to variance_out in the formula.</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>batch_mean</td>
  <td>Output</td>
  <td>Save mean of the current batch (= sum / num), for reuse in backpropagation, corresponding to batch_mean in the formula.</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>batch_variance</td>
  <td>Output</td>
  <td>Biased save variance of the current batch (= square_sum/num - μ²), for reuse in backpropagation, corresponding to batch_variance in the formula.</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
</tbody>
</table>

- <term>Ascend 950PR&950DT products</term>: the data format NDC1HWC0 is not supported.
- <term>Atlas A3 products</term>: the data formats NHWC and NDHWC are not supported.
- <term>Atlas A2 products</term>: the data formats NHWC and NDHWC are not supported.
- <term>Atlas 200I/500 A2 inference products</term>: the data type BFLOAT16 is not supported, and the data formats NCHW, NCDHW, NHWC, and NDHWC are not supported.
- <term>Atlas inference products</term>: the data type BFLOAT16 is not supported, and the data formats NCHW, NHWC, and NDHWC are not supported.
- <term>Atlas training products</term>: the data type BFLOAT16 is not supported, and the data formats NCDHW, NHWC, and NDHWC are not supported.

## Constraints

- The channel axis C is determined by the data format of x.
- <term>Ascend 950PR&950DT products</term>: the dimension rank of x supports only 4 (NCHW/NHWC) and 5 (NCDHW/NDHWC).
- The data type, data format, and shape of y are all the same as those of x.
- The shapes of mean, variance, batch_mean, and batch_variance must be the same as sum.
- mean and variance are in-place input and output: after the invocation completes, the original tensor content is overwritten by the running statistics after the EMA update.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_bn3_d_training_update](examples/arch35/test_geir_bn3_d_training_update.cpp) | Invokes the BN3DTrainingUpdate operator through the operator IR-based [graph construction](op_graph/bn3_d_training_update_proto.h) method. |
