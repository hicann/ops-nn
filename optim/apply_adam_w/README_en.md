# ApplyAdamW

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Implements the adamW optimizer function.

- Formula:

  $$
  g_t=\begin{cases}-g_t
  & \text{ if } maxmize= true\\
  g_t  & \text{ if } maxmize=false
  \end{cases}
  $$
  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$
  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$
  $$
  \beta_{1}^{t}=\beta_{1}^{t-1}\times\beta_{1}
  $$
  $$
  \beta_{2}^{t}=\beta_{2}^{t-1}\times\beta_{2}
  $$
  $$
  v_t=\begin{cases}\max(maxGradNorm, v_t)
  & \text{ if } amsgrad = true\\
  v_t  & \text{ if } amsgrad = false
  \end{cases}
  $$
  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$
  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$
  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

<table style="undefined;table-layout: fixed; width: 1305px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 247px">
  <col style="width: 108px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>var</td>
      <td>Computation input/computation output</td>
      <td>Weight input to be computed and also the output, which is theta in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>m</td>
      <td>Computation input/computation output</td>
      <td>m parameter in the adamW optimizer, which is m in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>v</td>
      <td>Computation input/computation output</td>
      <td>v parameter in the adamW optimizer, which is v in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>

      <td>beta1_power</td>
      <td>Computation input</td>
      <td>beta1^(t-1) parameter</td>
      <td>FLOAT16, BFLOAT16, FLOAT32<</td>
      <td>ND</td>
    </tr>
    <tr>

      <td>beta2_power</td>
      <td>Computation input</td>
      <td>beta2^(t-1) parameter</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>

      <td>lr</td>
      <td>Computation input</td>
      <td>Learning rate, which is eta in the formula</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>

      <tr>
      <td>weight_decay</td>
      <td>Computation input</td>
      <td>Weight decay coefficient</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>

      <tr>
      <td>beta1</td>
      <td>Computation input</td>
      <td>beta1 parameter.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>

      <tr>
      <td>beta2</td>
      <td>Computation input</td>
      <td>beta2 parameter.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>

    <tr>

      <td>epsilon</td>
      <td>Computation input</td>
      <td>Prevents the divisor from being 0.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>

    <tr>

      <td>grad</td>
      <td>Computation input</td>
      <td>Gradient data, which is g_t in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>

      <td>max_grad_norm</td>
      <td>Computation input</td>
      <td>Stores the maximum value of the v parameter, which is v in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>amsgrad</td>
      <td>Attribute</td>
      <td>Specifies whether to use the maxGradNormOptional variable</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maximize</td>
      <td>Attribute</td>
      <td>Specifies whether to negate the gradient grad and apply gradient ascent to optimize the weights to maximize the loss function</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## Constraints

- The data types of the input tensors should be consistent. The supported data types are FLOAT16, BFLOAT16, and FLOAT32.
- The shapes of the input tensors beta1Power, beta2Power, lr, weightDecay, beta1, beta2, and eps should be of size 1.
- When the input Boolean value maximize is true, the maxGradNormOptional parameter is mandatory, and its data type and shape should be consistent with those of varRef.

- Deterministic computation:
  - aclnnApplyAdamW uses a deterministic implementation by default.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_apply_adam_w](./examples/arch35/test_aclnn_apply_adam_w.cpp) | Invokes the ApplyAdamW operator through the [aclnnApplyAdamW](./docs/aclnnApplyAdamW_en.md) interface. |
