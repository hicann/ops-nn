# ApplyFusedEmaAdam

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

- **Operator function**: Implements the FusedEmaAdam fused optimizer function, which combines the Adam optimizer with the exponential moving average (EMA) technique.
- **Formula**:

  $$
  (correction_{\beta_1},correction_{\beta_2},)=\begin{cases}
  (1,1),&biasCorrection=False\\
  (1-\beta_1^{step},1-\beta_2^{step}),&biasCorrection=True
  \end{cases}
  $$

  $$
  grad=\begin{cases}
  grad+weightDecay*var,&mode=0\\
  grad,&mode=1
  \end{cases}
  $$

  $$
  m_{out}=\beta_1*m+(1-\beta_1)*grad
  $$

  $$
  v_{out}=\beta_2*v+(1-\beta_2)*grad^2
  $$

  $$
  m_{next}=m_{out}/correction_{\beta_1}
  $$

  $$
  v_{next}=v_{out}/correction_{\beta_2}
  $$

  $$
  denom=\sqrt{v_{next}}+eps
  $$

  $$
  update=\begin{cases}
  m_{next}/denom,&mode=0\\
  m_{next}/denom+weightDecay*var,&mode=1
  \end{cases}
  $$

  $$
  var_{out}=var-lr*update
  $$

  $$
  s_{out}=emaDecay*s+(1-emaDecay)*var_{out}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
<col style="width: 155px">
<col style="width: 162px">
<col style="width: 380px">
<col style="width: 276px">
<col style="width: 107px">
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
    <td>grad</td>
    <td>Input</td>
    <td>Gradient corresponding to the parameter to be updated, which corresponds to grad in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>varRef</td>
    <td>Input/Output</td>
    <td>Parameter to be updated, which corresponds to var in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mRef</td>
    <td>Input/Output</td>
    <td>First moment corresponding to the parameter to be updated, which corresponds to m in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>vRef</td>
    <td>Input/Output</td>
    <td>Second moment corresponding to the parameter to be updated, which corresponds to v in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sRef</td>
    <td>Input/Output</td>
    <td>EMA weight corresponding to the parameter to be updated, which corresponds to s in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>step</td>
    <td>Input</td>
    <td>Current number of updates of the optimizer, which corresponds to step in the formula.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Attribute</td>
    <td>Learning rate, which corresponds to lr in the formula. Value range: (0,1). Default value: 1e-3.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>emaDecay</td>
    <td>Attribute</td>
    <td>Decay rate of the exponential moving average (EMA), which corresponds to emaDecay in the formula. Value range: (0,1). Default value: 0.9999.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>beta1</td>
    <td>Attribute</td>
    <td>Coefficient for computing the first moment, which corresponds to β1 in the formula. Value range: (0,1). Default value: 0.9.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>beta2</td>
    <td>Attribute</td>
    <td>Coefficient for computing the second moment, which corresponds to β2 in the formula. Value range: (0,1). Default value: 0.999.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>eps</td>
    <td>Attribute</td>
    <td>Term added to the denominator for numerical stability, which corresponds to eps in the formula. Default value: 1e-8.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>mode</td>
    <td>Attribute</td>
    <td>Controls whether to apply L2 regularization or weight decay, which corresponds to mode in the formula. 1 indicates adamw, and 0 indicates L2. Default value: 1.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>biasCorrection</td>
    <td>Attribute</td>
    <td>Controls whether to perform bias correction, which corresponds to biasCorrection in the formula. true indicates that correction is performed, and false indicates that correction is not performed. Default value: True.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>weightDecay</td>
    <td>Attribute</td>
    <td>Weight decay, which corresponds to weightDecay in the formula. Value range: (0,1). Default value: 0.0.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- The data types and shapes of the inputs grad, var, m, v, and s must be consistent.

## Invocation Description

| Invocation method  | Sample code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_apply_fused_ema_adam](./examples/test_aclnn_apply_fused_ema_adam.cpp) | Calls the ApplyFusedEmaAdam operator through the [aclnnApplyFusedEmaAdam](docs/aclnnApplyFusedEmaAdam_en.md) API. |
