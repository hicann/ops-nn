# FusedAdamw

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function:

  It is a fused AdamW optimizer. It fuses operations such as parameter updates of multiple parameter lists (params, grads, expAvgs, expAvgSqs, maxExpAvgSqs, stateSteps), first-order/second-order momentum updates, and gradient scaling into a single kernel, improving the optimizer execution efficiency in training scenarios.

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
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t}
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
      <td>paramsRef</td>
      <td>Input/Output</td>
      <td>The weight list to be computed (TensorList), also the output, the input/output θ in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expAvgsRef</td>
      <td>Input/Output</td>
      <td>The first-order momentum list (TensorList), the input/output m in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expAvgSqsRef</td>
      <td>Input/Output</td>
      <td>The second-order momentum list (TensorList), the input/output v in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maxExpAvgSqsRef</td>
      <td>Input/Output</td>
      <td>The maximum second-order momentum list (TensorList), the maxGradNorm in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grads</td>
      <td>Input</td>
      <td>The gradient list (TensorList), the input g in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stateSteps</td>
      <td>Input</td>
      <td>The iteration count list (TensorList), the t in the formula.</td>
      <td>INT64, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradScaleOptional</td>
      <td>Input</td>
      <td>An optional input, the gradient scaling factor. When foundInfOptional=1, the gradients are cleared and parameter updates stop.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>foundInfOptional</td>
      <td>Input</td>
      <td>An optional input, indicating whether Inf/NaN occurs. Updates stop when it equals 1.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Attribute</td>
      <td><ul><li>The learning rate.</li><li>The value range is (0,1), and the default value is 0.001. It is the η in the formula.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta1</td>
      <td>Attribute</td>
      <td><ul><li>The beta1 parameter.</li><li>The value range is (0,1), and the default value is 0.9. It is the β1 in the formula.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta2</td>
      <td>Attribute</td>
      <td><ul><li>The beta2 parameter.</li><li>The value range is (0,1), and the default value is 0.999. It is the β2 in the formula.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightDecay</td>
      <td>Attribute</td>
      <td><ul><li>The weight decay coefficient.</li><li>The value range is (0,1), and the default value is 0.0. It is the λ in the formula.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td><ul><li>The parameter that prevents division by 0.</li><li>The default value is 1e-8. It is the ϵ in the formula.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>amsgrad</td>
      <td>Attribute</td>
      <td><ul><li>Whether to use the AMSGrad variable of the algorithm. Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maximize</td>
      <td>Attribute</td>
      <td><ul><li>Whether to maximize the parameters. Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The data types of paramsRef, expAvgsRef, expAvgSqsRef, maxExpAvgSqsRef, and grads must be the same, and support FLOAT16, BFLOAT16, and FLOAT32.
- Each tensor in the same TensorList must have the same shape. The tensors of paramsRef, expAvgsRef, expAvgSqsRef, maxExpAvgSqsRef, and grads at corresponding positions must also have the same shape.
- stateSteps supports INT64 and FLOAT32, with an element count of 1.
- When amsgrad is false, maxExpAvgSqsRef can be empty. When amsgrad is true, maxExpAvgSqsRef is required and its shape must match paramsRef.

## Invocation Description

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| aclnn invocation | [test_aclnn_fused_adamw](./examples/test_aclnn_fused_adamw.cpp) | Calls the FusedAdamw operator through the [aclnnFusedAdamw](./docs/aclnnFusedAdamw_en.md) API. |
