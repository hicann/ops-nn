# ApplyAdamWV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function:

  Implements the adamW optimizer.

- Formula:

  $$
  if(maximize) : g_{t} = - g_{t}
  $$

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t}
  $$

  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$

  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}
  $$

  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}
  $$

  $$
  if(amsgrad) : maxGradNorm = max(maxGradNorm,\hat{v}_{t})
  $$

  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 370px">
  <col style="width: 200px">
  <col style="width: 70px">
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
      <td>varRef</td>
      <td>Input/Output</td>
      <td>Weight input to be computed and also the output, input/output θ in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mRef</td>
      <td>Input/Output</td>
      <td>First moment corresponding to the parameter to be updated, input/output m in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vRef</td>
      <td>Input/Output</td>
      <td>Second moment corresponding to the parameter to be updated, input/output v in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maxGradNormOptionalRef</td>
      <td>Input/Output</td>
      <td>Stores the maximum value of the v parameter, maxGradNorm in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Gradient data, input g in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>step</td>
      <td>Input</td>
      <td>Number of iterations, t in the formula.</td>
      <td>INT64, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Attribute</td>
      <td><ul><li>Learning rate.</li><li>The value range is (0,1), and the default is 0.1. It is η in the formula.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta1</td>
      <td>Attribute</td>
      <td><ul><li>beta1 parameter.</li><li>The value range is (0,1), and the default is 0.1. It is β1 in the formula.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta2</td>
      <td>Attribute</td>
      <td><ul><li>beta2 parameter.</li><li>The value range is (0,1), and the default is 0.1. It is β2 in the formula.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightDecay</td>
      <td>Attribute</td>
      <td><ul><li>Weight decay coefficient.</li><li>The value range is (0,1), and the default is 0.1. It is λ in the formula.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td><ul><li>Parameter that prevents division by 0.</li><li>The default is 1e-8. It is ϵ in the formula.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>amsgrad</td>
      <td>Attribute</td>
      <td><ul><li>Whether to use the AMSGrad variant of the algorithm. The default is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maximize</td>
      <td>Attribute</td>
      <td><ul><li>Whether to maximize the parameters. The default is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_apply_adam_w_v2](./examples/test_aclnn_apply_adam_w_v2.cpp) | Invokes the ApplyAdamWV2 operator through the [aclnnApplyAdamWV2](./docs/aclnnApplyAdamWV2_en.md) interface.    |
