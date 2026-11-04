# INTrainingUpdateV2

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

- Operator function: it serves as the update stage of the instance normalization training fusion flow and works with INTrainingReduceV2. The operator computes per-instance, per-channel statistics of the current batch from `sum` and `square_sum`, normalizes `x`, and optionally performs an affine transformation and a sliding statistics update.
- The following formulas describe the public mathematical semantics of the operator. For the trigger rules of the optional inputs and the dynamic shape capabilities of each product, see "Product Differences" and "Constraints".
- Let $R=H\times W$. The current mean, biased variance, and unbiased variance are:

  $$
  \mu={sum\over R},\qquad
  v_b=\max\left({square\_sum\over R}-\mu^2,0\right)
  $$

  $$
  v_u=\begin{cases}
  0,&R=1\\
  v_b{R\over R-1},&R>1
  \end{cases}
  $$

- When both `gamma` and `beta` are provided:

  $$
  y={x-\mu\over\sqrt{v_b+\epsilon}}\times gamma+beta
  $$

- When at least one of `gamma` or `beta` is not provided, neither takes part in the computation:

  $$
  y={x-\mu\over\sqrt{v_b+\epsilon}}
  $$

- When both `mean` and `variance` are provided:

  $$
  batch\_mean=momentum\times\mu+(1-momentum)\times mean
  $$

  $$
  batch\_variance=momentum\times v_u+(1-momentum)\times variance
  $$

- When at least one of `mean` or `variance` is not provided, neither takes part in the update, and the operator outputs the current statistics:

  $$
  batch\_mean=\mu,\qquad batch\_variance=v_u
  $$

## Parameters

<table style="table-layout: fixed"><colgroup>
<col style="width: 150px">
<col style="width: 150px">
<col style="width: 410px">
<col style="width: 180px">
<col style="width: 150px">
</colgroup>
<thead><tr>
<th>Parameter name</th><th>Input/Output/Attribute</th><th>Description</th><th>Data type</th><th>Data format</th>
</tr></thead>
<tbody>
<tr><td>x</td><td>Input</td><td>The 4D tensor to be normalized.</td><td>FLOAT16, FLOAT</td><td>NCHW, NHWC</td></tr>
<tr><td>sum</td><td>Input</td><td>The 4D spatial-dimension sum output by INTrainingReduceV2; both H and W are 1.</td><td>FLOAT</td><td>NCHW, NHWC</td></tr>
<tr><td>square_sum</td><td>Input</td><td>The 4D spatial-dimension square sum output by INTrainingReduceV2; both H and W are 1.</td><td>FLOAT</td><td>NCHW, NHWC</td></tr>
<tr><td>gamma</td><td>Optional input</td><td>The 4D affine scaling parameter; both H and W are 1.</td><td>FLOAT</td><td>NCHW, NHWC</td></tr>
<tr><td>beta</td><td>Optional input</td><td>The 4D affine offset parameter; both H and W are 1.</td><td>FLOAT</td><td>NCHW, NHWC</td></tr>
<tr><td>mean</td><td>Optional input</td><td>The 4D sliding mean to be updated; both H and W are 1.</td><td>FLOAT</td><td>NCHW, NHWC</td></tr>
<tr><td>variance</td><td>Optional input</td><td>The 4D sliding variance to be updated; both H and W are 1.</td><td>FLOAT</td><td>NCHW, NHWC</td></tr>
<tr><td>momentum</td><td>Optional attribute</td><td>Update weight for the statistics. Default value: 0.1.</td><td>FLOAT</td><td>-</td></tr>
<tr><td>epsilon</td><td>Optional attribute</td><td>Numerical stability term added to the biased variance. Default value: 0.00001.</td><td>FLOAT</td><td>-</td></tr>
<tr><td>y</td><td>Output</td><td>The normalization result; its shape, data type, and data format match x.</td><td>FLOAT16, FLOAT</td><td>NCHW, NHWC</td></tr>
<tr><td>batch_mean</td><td>Output</td><td>The updated mean; its shape matches sum.</td><td>FLOAT</td><td>NCHW, NHWC</td></tr>
<tr><td>batch_variance</td><td>Output</td><td>The updated variance; its shape matches sum.</td><td>FLOAT</td><td>NCHW, NHWC</td></tr>
</tbody></table>

### Product Differences

- <term>Ascend 950PR&950DT products</term>: support 4D NCHW and NHWC; support dynamic dimensions under a fixed 4D rank, and do not support dynamic rank.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: support 4D NCHW and NHWC.

## Constraints

### Public constraints of all supported products

- All inputs are 4D tensors and support only the NCHW and NHWC public logical formats. The data types of `x` and `y` are FLOAT16 or FLOAT, and the other inputs and outputs are FLOAT.
- H and W of `sum`, `square_sum`, `gamma`, `beta`, `mean`, `variance`, and the two statistics outputs are all 1. The four optional inputs are each optional in the public prototype.

### Supplementary constraints of only <term>Ascend 950PR&950DT products</term>

- Dynamic dimensions are supported under a fixed 4D rank; dynamic rank is not supported.
- Under NCHW, the spatial dimensions are the H and W axes of `x`; under NHWC, the rule is equivalent.
- `sum` and `square_sum` must always use the same logical format as `x`, and their shapes are exactly [N,C,1,1] or [N,1,1,C].
- When both `gamma` and `beta` are provided, both must use the same logical format as `x`. Each supports G=1 or G=N; the two G values may differ, and C must match `x`.
- When both `mean` and `variance` are provided, both must use the same logical format as `x`, and their shapes match `sum`.
- The four optional inputs can each be absent independently. The affine branch is enabled only when both `gamma` and `beta` are provided; the sliding update branch is enabled only when both `mean` and `variance` are provided. Half pairs can be passed in, but an isolated input is not read. An isolated input is still validated for FP32, 4D, NCHW/NHWC, and spatial dimensions of 1; its G/N/C coupling with `x` is not validated.
- When N=0 or C=0, the operator treats it as an empty tensor. When both N and C are greater than 0, H and W must be greater than 0.
- When R=1, the unbiased variance is defined as 0.
- `N*C` derived from the shape must be representable by a signed 64-bit integer. For non-empty inputs with both N and C greater than 0, `H*W`, `N*C*H*W`, and the byte lengths of the input and statistics tensors must also be representable by a signed 64-bit integer. The actually usable scale is also constrained by device memory and framework general limits.
- Under NHWC, the GM byte stride of the 2D data movement, that is `(C-currentC)*sizeof(x)` for each channel block, must not exceed $2^{40}-1$ bytes. The implementation keeps each channel block at no more than 64 elements and checks the worst-case stride with the minimum block.

## Invocation Description

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| GE graph mode | [test_geir_in_training_update_v2](./examples/arch35/test_geir_in_training_update_v2.cpp) | Builds the graph through the [operator IR](./op_graph/in_training_update_v2_proto.h) and executes it on <term>Ascend 950PR&950DT products</term>. |

This operator is an internal graph node of the instance normalization training fusion flow and does not provide an aclnn single-operator interface with the same name.
