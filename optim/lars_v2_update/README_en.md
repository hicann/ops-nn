# LarsV2Update

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: performs the LARS-V2 optimizer gradient update, which adaptively adjusts the trust coefficient of the gradient according to the L2 norm of the weight, and is used in large-batch training scenarios.

- Formula:

  $$
  w\_norm = \sqrt{w\_square\_sum}
  $$

  $$
  g\_norm = \sqrt{g\_square\_sum}
  $$

  $$
  coeff = \frac{hyperpara \times w\_norm}{weight\_decay \times w\_norm + g\_norm + epsilon}
  $$

  - If use_clip = True:

    $$
    coeff = \max\left(0,\ \min\left(\frac{coeff}{learning\_rate},\ 1\right)\right)
    $$

  $$
  grad\_weight = w \times weight\_decay + g
  $$

  $$
  g\_new = grad\_weight \times coeff
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
<col style="width: 100px">
<col style="width: 150px">
<col style="width: 280px">
<col style="width: 330px">
<col style="width: 120px">
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
    <td>w</td>
    <td>Input</td>
    <td>Weight tensor.</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>g</td>
    <td>Input</td>
    <td>Gradient tensor, with the same data type and shape as w.</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>w_square_sum</td>
    <td>Input</td>
    <td>Sum of squares of the weights (precomputed by SquareSumAll), a scalar.</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>g_square_sum</td>
    <td>Input</td>
    <td>Sum of squares of the gradients, a scalar.</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>weight_decay</td>
    <td>Input</td>
    <td>Weight decay coefficient, a scalar.</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>learning_rate</td>
    <td>Input</td>
    <td>Learning rate, a scalar. It participates in the computation only when use_clip=True.</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>hyperpara</td>
    <td>Attribute</td>
    <td>LARS trust coefficient (eta), with the default value 0.001.</td>
    <td>Float</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epsilon</td>
    <td>Attribute</td>
    <td>Small constant to prevent division by zero, with the default value 0.00001.</td>
    <td>Float</td>
    <td>-</td>
    </tr>
    <tr>
    <td>use_clip</td>
    <td>Attribute</td>
    <td>Whether to clip coeff to [0,1], with the default value False.</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>g_new</td>
    <td>Output</td>
    <td>Updated gradient, with the same data type and shape as w.</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraints

- Inputs w and g must have the same shape and data type (the Tiling stage checks that the dtype is DT_FLOAT and that the shapes are exactly identical; see `op_host/arch35/lars_v2_update_tiling_arch35.cpp`).
- Output g_new has the same type, shape, and dtype as w (InferShape directly copies the shape of w without broadcasting; the Tiling stage performs a fallback check that g_new and w have the same shape).
- w_square_sum, g_square_sum, weight_decay, and learning_rate are always FLOAT-type scalars (the number of elements is 1, and they are broadcast to the shape of w for the computation; the Tiling stage checks the dtype and the number of elements).
- Supports 0 to 8 dimensions (a 0-dimensional scalar is processed as 1 dimension, and the Tiling stage checks that the number of dimensions does not exceed 8).
- Supports empty tensors (the number of elements is 0; for example, when the shape contains a 0 dimension, the output is an empty tensor without an error; on the kernel side, totalTiles==0 short-circuits the entire execution, and each tile is skipped when its count <= 0).
- NaN in the input data propagates according to IEEE 754 semantics (the clipping semantics for NaN in hardware vector instructions may differ from the framework reference implementation; the upper-layer framework must guarantee NaN input scenarios, and the operator does not intercept NaN).
- When inf/nan is passed in for attributes hyperpara/epsilon, no check or interception is performed, and they directly participate in the computation according to IEEE 754 floating-point semantics (hyperpara=inf→g_new is all ±Inf; hyperpara/epsilon=nan→g_new is all NaN; epsilon=inf→coeff=0→g_new is all zeros), consistent with the NaN propagation policy for inputs.
- The total number of elements has no independent software limit: the single-axis length and the total scale are constrained only by the device HBM capacity (the three full-size fp32 buffers of w/g/g_new occupy 12×numel bytes in total); GM offsets and the entire shape/stride chain use int64, and extra-large shapes are handled through Tiling chunking.
- When use_clip=True, learning_rate must be non-zero (it participates in the coeff/learning_rate clipping, and a value of 0 causes division by zero); epsilon must not be configured with a value that makes the denominator weight_decay×w_norm+g_norm+epsilon equal to 0 (for example, all-zero norms and epsilon=0).

## Invocation Description

| Invocation method | Sample code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GE graph mode | [test_geir_lars_v2_update](./examples/arch35/test_geir_lars_v2_update.cpp)   | Invokes the LarsV2Update operator through graph construction with the [operator IR](./op_graph/lars_v2_update_proto.h); the sample includes CPU golden numerical verification (dtype/shape/element-wise). |
