# GNTrainingUpdate

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Fused GroupNorm training forward update operator. It consumes the per-group `sum` / `square_sum` output by the companion operator GNTrainingReduce, computes the mean and (biased) variance of each group on the fly, performs normalization and optional affine transformation on `x`, and bypasses `batch_mean` / `batch_variance` as outputs for reuse by the backward GroupNormGrad.

- Formula (`M = (C/num_groups)·H·W` is the number of elements per group):

  $$
  mean = \frac{sum}{M},\quad variance = \frac{square\_sum}{M} - mean^2
  $$

  $$
  y = \frac{x - mean}{\sqrt{variance + epsilon}} \times scale + offset
  $$

  When both `scale` / `offset` are missing, the operator degrades to pure normalization. `mean` / `variance` are reserved inputs of the official IR and do not participate in the calculation (the statistics are always computed on the fly by `sum` / `square_sum`).

## Parameters

<table style="undefined;table-layout: fixed; width: 910px"><colgroup>
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
    <tr><td>x</td><td>Input</td><td>Feature map to be normalized, 4D [N,C,H,W] (NCHW) or [N,H,W,C] (NHWC).</td><td>FLOAT16, FLOAT</td><td>ND, NCHW, NHWC</td></tr>
    <tr><td>sum</td><td>Input</td><td>Per-group element sum output by GNTrainingReduce, 5D. It is [N,G,1,1,1] for NCHW and [N,1,1,G,1] for NHWC.</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>square_sum</td><td>Input</td><td>Per-group sum of squares, with the same shape as sum.</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>scale</td><td>Optional input</td><td>Affine γ, 5D. It is [1,G,1,1,1] for NCHW and [1,1,1,G,1] for NHWC. When both scale and offset are missing, the operator degrades to pure normalization.</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>offset</td><td>Optional input</td><td>Affine β, with the same shape as scale.</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>mean</td><td>Optional input</td><td>Reserved slot of the official IR (same shape as sum). It does not participate in the calculation.</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>variance</td><td>Optional input</td><td>Reserved slot of the official IR (same shape as sum). It does not participate in the calculation.</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>num_groups</td><td>Optional attribute</td><td>Number of groups G. The default value is 2, and it must divide C.</td><td>INT</td><td>-</td></tr>
    <tr><td>epsilon</td><td>Optional attribute</td><td>Small value added to variance to prevent division by zero. The default value is 0.0001.</td><td>FLOAT</td><td>-</td></tr>
    <tr><td>y</td><td>Output</td><td>Normalization (+ optional affine) result. Its shape and data type are the same as those of x.</td><td>FLOAT16, FLOAT</td><td>ND, NCHW, NHWC</td></tr>
    <tr><td>batch_mean</td><td>Output</td><td>Per-group mean computed on the fly (sum/M), with the same shape as sum.</td><td>FLOAT</td><td>ND</td></tr>
    <tr><td>batch_variance</td><td>Output</td><td>Per-group biased variance computed on the fly (square_sum/M − mean², without epsilon), with the same shape as sum.</td><td>FLOAT</td><td>ND</td></tr>
  </tbody>
</table>

## Constraints

- `G == num_groups` and `C % num_groups == 0` (hard validation on Host).
- Tensors other than `x` / `y` support only ND. When `x` / `y` are ND, the layout is inferred from the position of the group dimension of the statistics. When an explicit NCHW/NHWC label conflicts with the group dimension position of the statistics, the input is rejected.
- An empty batch with `N=0` is valid: all three outputs are empty and no kernel is dispatched. C / H / W / G being 0 is regarded as invalid.
- Dynamic shape: Unknown dimensions (-1) are supported. Unknown dimensions pass through InferShape, and kernel compilation handles them uniformly as static (DynamicCompileStatic). Dynamic rank: the interface enables DynamicRankSupport, but when x or sum has an unknown rank, all statically checkable shape validations are deferred to runtime, and the interception point of invalid inputs is deferred accordingly.
- The main computation chain uses float32 throughout (when `x` is float16, the output is rounded to float16).

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_gn_training_update.cpp](examples/arch35/test_geir_gn_training_update.cpp) | Calls the operator by building a graph through [GNTrainingUpdate IR](op_graph/gn_training_update_proto.h). |

Building and running the sample depends on the CANN development environment and the installation of the custom operator package (see the Environment Preparation section of the repository root README for details):

1. Complete the CANN Toolkit installation and `source` its `set_env` environment script;
2. In this repository, run `bash build.sh --pkg --soc=ascend950 --ops=gn_training_update` to build the custom operator package and install it (`--install-path` specifies the installation path), and then `source` the `bin/set_env.bash` inside the installed package;
3. Build and run the sample: `bash build.sh --run_example gn_training_update graph cust --soc=ascend950` (cust means using the custom operator package);
4. Expected output: all three cases (NCHW ×2 + NHWC ×1) print `PASS`, and the actual values of y/batch_mean/batch_variance are printed.
