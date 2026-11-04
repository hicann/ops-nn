# GNTrainingReduce

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: completes the statistical reduction of the GroupNorm training forward pass. It divides the channel dimension of the input feature map `x` into several groups according to the number of channel groups `num_groups`, and computes $\Sigma x$ and $\Sigma x^2$ over the $D \cdot H \cdot W$ elements within each group $(n,g)$, outputting two fp32 tensors. This operator produces only the raw moments and is used in pairs with the companion operator GNTrainingUpdate.
- Computation formula: let $D = C / num\_groups$, and $M = D \cdot H \cdot W$ be the number of reduced elements per group. When input `x` is not fp32, it is first converted to fp32 (the fp32 path performs no conversion).

  - NCHW input: rearrange $x[N,C,H,W]$ into $x_g[N,G,D,H,W]$, and reduce over the axes $(2,3,4)$:

    $$
    sum[n,g] = \sum_{d=0}^{D-1}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1} x_g[n,g,d,h,w]
    $$

    $$
    square\_sum[n,g] = \sum_{d=0}^{D-1}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1} \left(x_g[n,g,d,h,w]\right)^2
    $$

  - NHWC input: rearrange $x[N,H,W,C]$ into $x_g[N,H,W,G,D]$, and reduce over the axes $(1,2,4)$, with the same formula form as above.

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
<tr><td>x</td><td>Input</td><td>GroupNorm input feature map, that is, x in the formula.</td><td>FLOAT16, FLOAT</td><td>NCHW, NHWC</td></tr>
<tr><td>sum</td><td>Output</td><td>Σx of each (n,g) group, that is, sum in the formula.</td><td>FLOAT</td><td>ND</td></tr>
<tr><td>square_sum</td><td>Output</td><td>Σx² of each (n,g) group, that is, square_sum in the formula.</td><td>FLOAT</td><td>ND</td></tr>
<tr><td>num_groups</td><td>Optional attribute</td><td>Number of channel groups G, that is, num_groups in the formula; the default value is 2, and it must be consistent with the companion operator GNTrainingUpdate.</td><td>INT64</td><td>-</td></tr>
</tbody>
</table>

## Constraints

- The channel dimension `C` must be divisible by `num_groups`; the value of `num_groups` must be no less than 1.
- The shapes of the outputs `sum` and `square_sum` are always rank-5: for an NCHW input, the shapes of the outputs `sum` and `square_sum` are [N,num_groups,1,1,1]; for an NHWC input, the output shapes are [N,1,1,num_groups,1].
- Empty tensors are legal: when N=0, the outputs are empty tensors; when N>0 and C=0, H=0, or W=0, the group is empty and the reduction is an empty sum, so both `sum` and `square_sum` are written as 0.
- This operator does not support input broadcasting.

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_gn_training_reduce.cpp](examples/arch35/test_geir_gn_training_reduce.cpp) | Calls the GNTrainingReduce operator by constructing a graph through the [operator IR](op_graph/gn_training_reduce_proto.h). |
