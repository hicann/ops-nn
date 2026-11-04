# ApplyFtrlV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------:|
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |
| Atlas 200I/500 A2 inference products | × |
| Atlas inference products | × |
| Atlas training products | × |

## Function

ApplyFtrlV2 implements the V2 version of the parameter update rule of the FTRL (Follow The Regularized Leader) optimization algorithm. This operator is a pure element-wise computation operator. In each iteration of model training, it updates the weight variable (var), the gradient square accumulator (accum), and the linear accumulator (linear) in place based on the gradient, learning rate, and regularization parameters. It applies to online learning scenarios with large-scale sparse features, such as advertisement click-through rate (CTR) prediction, recommendation systems, and so on.

**Difference between V2 and V1 (ApplyFtrl)**: V2 adds the `l2Shrinkage` parameter and supports shrinkage-type L2 regularization. V2 adds a correction term of `2 * l2Shrinkage * var` on the gradient, which is equivalent to directly adding an L2 penalty term to the loss function (`Loss += l2Shrinkage * ||w||²`). When `l2Shrinkage = 0`, V2 degrades to V1.

**Formula** (7-step update):

$$
\begin{aligned}
\hat{g} &= g + 2 \cdot \lambda_s \cdot w \\
n' &= n + g^2 \\
\sigma &= \frac{{n'}^{-p} - n^{-p}}{\eta} \\
z &\leftarrow z + \hat{g} - \sigma \cdot w \\
x &= \lambda_1 \cdot \text{sign}(z) - z \\
y &= \frac{{n'}^{-p}}{\eta} + 2 \cdot \lambda_2 \\
w &\leftarrow \begin{cases} x / y & \text{if } |z| > \lambda_1 \\ 0 & \text{otherwise} \end{cases} \\
n &\leftarrow n'
\end{aligned}
$$

Here: $w$=var (weight), $n$=accum (gradient square accumulation), $z$=linear (linear accumulator), $g$=grad (gradient), $\eta$=lr (learning rate), $\lambda_1$=l1, $\lambda_2$=l2, $\lambda_s$=l2Shrinkage, and $p$=lrPower.

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
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
  <tr>
    <td>var</td>
    <td>Input/Output (inplace)</td>
    <td>Model weight tensor, corresponding to $w$ in the formula. The kernel updates it in place, and the single-output view of the GE IR shares device memory with the input var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>Input (inplace update)</td>
    <td>Gradient square accumulator, corresponding to $n$ in the formula. Its shape/dtype must be the same as those of var; the kernel explicitly writes back to the input GM address.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>linear</td>
    <td>Input (inplace update)</td>
    <td>Linear accumulator, corresponding to $z$ in the formula. Its shape/dtype must be the same as those of var; the kernel explicitly writes back to the input GM address.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Gradient of the current step, corresponding to $g$ in the formula. Its shape/dtype must be the same as those of var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate (0-d tensor), corresponding to $\eta$ in the formula. Its dtype must be the same as that of var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l1</td>
    <td>Input</td>
    <td>L1 regularization strength (0-d tensor), corresponding to $\lambda_1$ in the formula. Its dtype must be the same as that of var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2</td>
    <td>Input</td>
    <td>L2 regularization strength (0-d tensor), corresponding to $\lambda_2$ in the formula. Its dtype must be the same as that of var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2Shrinkage</td>
    <td>Input</td>
    <td>L2 shrinkage strength (0-d tensor), corresponding to $\lambda_s$ in the formula (added in V2). Its dtype must be the same as that of var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lrPower</td>
    <td>Input</td>
    <td>Learning rate decay power (0-d tensor), corresponding to $p$ in the formula. Its dtype must be the same as that of var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Specifies whether to lock during the update. Default: false. The current implementation does not enforce a mutex lock; it only serves as a semantic placeholder.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>Output</td>
    <td>Updated var tensor, which shares device memory with the input var (inplace).</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

> **Note**: accum and linear are updated in place (consistent with TensorFlow). They are not returned as explicit outputs, and the computation results are written back to the input addresses directly.

## Constraints

- **Data type consistency**: The data types of the four tensors var, accum, linear, and grad must be identical; the data types of the five scalars lr, l1, l2, l2Shrinkage, and lrPower must be the same as those of the tensors.
- **Shape constraints**: var, accum, linear, and grad must have exactly the same shape; lr, l1, l2, l2Shrinkage, and lrPower must be scalars (0-d tensors); the tensor dimension range is 0 to 8 dimensions.
- **Parameter value ranges**: lr > 0 (the learning rate must be a positive number), l1 >= 0, l2 >= 0, l2Shrinkage >= 0, and lrPower <= 0.
- **In-place semantics**: The three parameters var, accum, and linear are updated in place, and the original data is overwritten after execution.
- **FP16/BF16 computation**: FP16 and BF16 inputs are promoted to FP32 for computation inside the kernel, and the results are cast back to the original type for output.
- **Determinism**: The default implementation is deterministic. This operator is a pure element-wise operation without reduction or sorting, so the computation is fully deterministic.
- **The `multiply_linear_by_lr` attribute is not supported**: The standard TensorFlow ApplyFtrlV2 path applies (`multiply_linear_by_lr=false`).
- **Empty tensor**: Empty tensors (numel=0) are supported, and the kernel skips the computation.

## Invocation Description

| Invocation method | Invocation sample | Description |
|:---------|:---------|:-----|
| Graph-mode invocation | [test_geir_apply_ftrl_v2](./examples/arch35/test_geir_apply_ftrl_v2.cpp) | Invokes the ApplyFtrlV2 operator through graph construction with the [operator IR](./op_graph/apply_ftrl_v2_proto.h). |
