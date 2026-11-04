# ApplyAdaMax

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- **Description**: Performs a single-step parameter update of the AdaMax optimizer. AdaMax is a variant of the Adam optimizer that uses the infinity norm $L_{\infty}$ instead of the second-moment estimate, and performs an **in-place** update (inplace semantics) on the weight `var`, the first moment `m`, and the infinity norm `v`. It aligns with the computation semantics of the TensorFlow `tf.raw_ops.ApplyAdaMax` interface and PyTorch `torch.optim.Adamax`.

- **Formula**:

  Given the gradient $g_t$ at time step $t$, the decay coefficients $\beta_1, \beta_2$, the learning rate $lr$, the numerical stability constant $\epsilon$, and the bias correction factor $\beta_1^t$ passed in externally:

  $$
  \begin{aligned}
  m_{t} &= \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \\
  v_{t} &= \max(\beta_2 \cdot v_{t-1},\ |g_t|) \\
  var_{t} &= var_{t-1} - \frac{lr}{1 - \beta_1^t} \cdot \frac{m_t}{v_t + \epsilon}
  \end{aligned}
  $$

  The operator prototype aligns with canndev `REG_OP(ApplyAdaMax)`: 9 inputs + 1 output + 1 attribute. `m` / `v` are not explicitly used as graph output ports; the update is completed by writing back in place to the input GM addresses (GE Variable inplace semantics).

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 600px">
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
    <td>Input</td>
    <td>Weight parameter tensor to be updated, corresponding to var in the formula. It shares the GM address with the graph output port var (in-place update). 1-8 dimensional ND format.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>m</td>
    <td>Input</td>
    <td>First-moment tensor, corresponding to m in the formula. The shape/dtype must be consistent with those of var; the Kernel writes back to the input GM address to complete the inplace update, with no explicit output port.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>v</td>
    <td>Input</td>
    <td>Infinity norm estimate tensor ($L_{\infty}$), corresponding to v in the formula. The shape/dtype must be consistent with those of var; the Kernel writes back to the input GM address to complete the inplace update, with no explicit output port.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta1_power</td>
    <td>Input</td>
    <td>Bias correction factor $\beta_1^t$, passed in externally. The shape must be a scalar Tensor of [1]. $1 - \beta_1^t \neq 0$ must be guaranteed.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate, corresponding to lr in the formula. The shape must be a scalar Tensor of [1].</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta1</td>
    <td>Input</td>
    <td>First-moment decay coefficient $\beta_1$, with the value range [0, 1). The shape must be a scalar Tensor of [1].</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta2</td>
    <td>Input</td>
    <td>Infinity norm decay coefficient $\beta_2$, with the value range [0, 1). The shape must be a scalar Tensor of [1].</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>Input</td>
    <td>Numerical stability constant $\epsilon$, which must be greater than 0. The shape must be a scalar Tensor of [1].</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Current gradient $g_t$, corresponding to grad in the formula. The shape/dtype must be consistent with those of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>A bool attribute for semantic placeholder, with the default value false. The current implementation does not enforce a mutex lock.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var</td>
    <td>Output</td>
    <td>Updated weight tensor, sharing Device memory with the input var (inplace). The shape/dtype is exactly the same as that of the input var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph mode | [test_geir_apply_adamax](./examples/test_geir_apply_adamax.cpp) | Invokes the ApplyAdaMax operator by building a graph through the [operator IR](./op_graph/apply_adamax_proto.h). |
