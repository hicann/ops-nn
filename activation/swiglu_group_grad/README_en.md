# SwigluGroupGrad

English | [简体中文](./README.md)

## Applicable Products

|Product      | Support |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      ×     |
|<term>Atlas A2 products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function

- **Operator function**: performs the backward gradient computation of the ClampedSwiglu activation function. It recomputes the clamp mask and sigmoid from the upstream gradient `grad_y` and the forward input `x`, and outputs `grad_x` and the optional `grad_weight`.
- **Formula**:

  Forward decomposition: x is split in half along the hidden dimension to obtain gate (g) and up (u); the optional clamp produces g̃ = min(c, g) and ũ = clip(u, −c, c); SiLU(g̃) = g̃·σ(g̃); y = SiLU(g̃)·ũ·w_t.

  $$silu'(g̃) = s + f − f·s$$

  $$dg = grad\_y \cdot silu'(g̃) \cdot ũ \cdot w_t \cdot I(g < c) \cdot m_r$$

  $$du = grad\_y \cdot f \cdot w_t \cdot I(−c < u < c) \cdot m_r$$

  $$grad\_weight = \Sigma(grad\_y \cdot y\_origin) \text{ along hidden dim}$$

  When group_index is provided, only the weight gradients of the first trunc rows are valid, and the remaining rows are set to 0:

  $$grad\_weight[t] = grad\_weight[t] \cdot I(t < trunc)$$

  where I is the open-interval indicator function (the mask is 0 at boundary values), trunc is the sum of the token counts of the groups in group_index, m_r is the group_index mask (that is, I(t < trunc)), and w_t is the broadcast of weight.

  **Constraint**: weight and y_origin must be provided together or be empty at the same time; when they are provided as a pair, grad_weight is computed.

  grad_x concatenation: gradX[..., :H] = dg, gradX[..., H:] = du.

- **Key features**: supports group_index dynamic grouping and weight gradient computation in MoE scenarios.

## Parameters

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>grad_y</td>
      <td>Input</td>
      <td>Upstream gradient, with the shape (..., H). Common shapes are (T, H) or (B, S, H).</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Forward input, with the shape (..., 2H). Common shapes are (T, 2H) or (B, S, 2H). It contains the gate and up branches.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td>MoE top-k routing weight. The number of its elements equals the number of elements on the non-trailing axes of grad_y, and its dtype is FP32. By default it is regarded as all 1s.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_origin</td>
      <td>Optional input</td>
      <td>Forward output y, whose last dimension is H. The number of its elements on the non-trailing axes equals the number of elements on the non-trailing axes of grad_y, and its dtype is the same as grad_y. When weight exists, y has already been multiplied by this weight. It must be provided together with weight.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Optional input</td>
      <td>Quantity index of the tokens/batches of each group, with the shape (G,) where G > 0, and the dtype INT64. By default all rows are regarded as valid.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>Attribute</td>
      <td>Clamp threshold scalar c. The value is -1.0 or >0.0, where -1.0 means that clamp is not enabled. When clamp is enabled, the value must be >0.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>Output</td>
      <td>Gradient of x, with the same shape as x and the same dtype as grad_y.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_weight</td>
      <td>Optional output</td>
      <td>Gradient of weight, with the same shape as weight and the dtype FP32. It is computed only when weight and y_origin are provided at the same time.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- H > 0
- x.shape[-1] = 2 × H (grad_y.shape[-1])
- The trailing axes of grad_y and x satisfy x.shape[-1] == 2 * grad_y.shape[-1], and the total numbers of elements on the non-trailing axes are equal
- weight and y_origin must be provided at the same time to compute grad_weight
- The default value of clamp_limit is -1.0. Passing -1.0 means that clamping is not performed, and when clamping is enabled, a value greater than 0 must be passed in
- When group_index is not empty, it must be a 1-dimensional non-empty tensor (G > 0)
- When group_index is absent, all rows after flattening the leading dimensions are valid rows

## Invocation Description

| Invocation method  | Sample code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn interface | [test_aclnn_swiglu_group_grad](./examples/test_aclnn_swiglu_group_grad.cpp) | Invokes the SwigluGroupGrad operator through [aclnnSwigluGroupGrad](./docs/aclnnSwigluGroupGrad_en.md) |
| Graph mode | [test_geir_swiglu_group_grad](./examples/test_geir_swiglu_group_grad.cpp) | Invokes the SwigluGroupGrad operator through the [operator IR](./op_graph/swiglu_group_grad_proto.h) |
| torch interface | [test_torch_swiglu_group_grad](./examples/test_torch_swiglu_group_grad.py) | Invokes the SwigluGroupGrad operator through `torch.ops.cann_ops_nn.swiglu_group_backward` |
