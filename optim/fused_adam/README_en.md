# FusedAdam

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- **Operator function**:

Implements the Adam optimizer function and supports completing the Adam optimizer function for multiple groups of parameter lists (TensorList) in one call.

- **Formula**:

$$
\begin{aligned}
&t=t+1 \\

&\tilde{g}_t = \begin{cases}
g_t / s & s \neq \text{None} \\
g_t & \text{otherwise}
\end{cases} \\

&g_{t+1} = \tilde{g}_t \\

&\hat{g}_t = \begin{cases}
-\tilde{g}_t & \text{maximize} \\
\tilde{g}_t & \text{otherwise}
\end{cases} \\

&\bar{g}_t = \hat{g}_t + \lambda \cdot \theta_t \\

&m_t=\beta_1 m_{t-1} + (1-\beta_1) \bar{g}_t\\

&v_t=\beta_2 v_{t-1} + (1-\beta_2) \bar{g}_t^2\\

&max\_v_t= \begin{cases} \max(v_t,max\_v_{t-1}) & \text{amsgrad} \\
max\_v_{t-1} & \text{otherwise}
\end{cases} \\

&\hat{m}_t=\frac{m_t}{1-\beta_1^t}\\

&\hat{v}_t= \begin{cases} \frac{max\_v_t}{1-\beta_2^t} & \text{amsgrad} \\
\frac{v_t}{1-\beta_2^t} & \text{otherwise}
\end{cases} \\

&\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t \\
\end{aligned}
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
    <td>paramsRef (aclTensorList*)</td>
    <td>Input/Output</td>
    <td><ul><li>Empty tensors are not supported.</li><li>The list of weights to be computed, θ in the formula.</li></ul></td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradsRef (aclTensorList*)</td>
    <td>Input/Output</td>
    <td><ul><li>Empty tensors are not supported.</li><li>The list of gradient data, g<sub>t</sub> in the formula. The gradients are updated only when the gradScale input is not empty.</li></ul></td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>expAvgsRef (aclTensorList*)</td>
    <td>Input/Output</td>
    <td><ul><li>Empty tensors are not supported.</li><li>The list of first moments, m in the formula.</li></ul></td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>expAvgSqsRef (aclTensorList*)</td>
    <td>Input/Output</td>
    <td><ul><li>Empty tensors are not supported.</li><li>The list of second moments, v in the formula. It cannot be negative.</li></ul></td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>maxExpAvgSqsRef (aclTensorList*)</td>
    <td>Input/Output</td>
    <td><ul><li>Empty tensors are not supported.</li><li>Saves the list of maximum second moments. It is compared with the updated expAvgSqsRef, and the maximum value is output.</li><li>This parameter is required when the amsgrad parameter is true, and optional when the amsgrad parameter is false.</li></ul></td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>stateSteps (aclTensorList*)</td>
    <td>Input</td>
    <td><ul><li>Empty tensors are not supported.</li><li>The list of iteration counts, t in the formula. It must be greater than 0.</li></ul></td>
    <td>INT64, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradScaleOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Optional input, the gradient scale factor s. When gradScaleOptional is not empty, the gradients are updated and output based on it (overwriting the original gradients).</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>foundInfOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Optional input, indicating whether Inf/NaN occurs. When foundInfOptional equals 1, the update stops.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr (double)</td>
    <td>Attribute</td>
    <td>The learning rate, η in the formula.</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>beta1 (double)</td>
    <td>Attribute</td>
    <td>The β<sub>1</sub> parameter.</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>beta2 (double)</td>
    <td>Attribute</td>
    <td>The β<sub>2</sub> parameter.</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>weightDecay (double)</td>
    <td>Attribute</td>
    <td>The weight decay coefficient, λ in the formula.</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>eps (double)</td>
    <td>Attribute</td>
    <td>Prevents the divisor from being 0.</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>amsgrad (bool)</td>
    <td>Attribute</td>
    <td>Specifies whether to use the AMSGrad variant of the algorithm.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>maximize (bool)</td>
    <td>Attribute</td>
    <td>Specifies whether to maximize the parameters.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- In the input tensorLists paramsRef, gradsRef, expAvgsRef, and expAvgSqsRef, each tensor must not be empty, the data types must be consistent, and only the data types FLOAT16, BFLOAT16, and FLOAT32 are supported.

- In the input tensorLists, the numbers of tensors in paramsRef, gradsRef, expAvgsRef, and expAvgSqsRef must be consistent, and the tensors with the same index must have the same shape.

- stateSteps is of the tensorList type and supports INT64 and FLOAT32. Its number of tensors must be the same as the numbers of tensors in paramsRef, gradsRef, expAvgsRef, expAvgSqsRef, and maxExpAvgSqsRef. Each tensor has at least 1 element. If the number of elements is greater than 1, the value of the 0th element is used as the value of stateSteps.

- When amsgrad is false, maxExpAvgSqsRef can be empty. When amsgrad is true, maxExpAvgSqsRef is required, and its number of tensors, as well as the shape and dtype of each tensor, must be consistent with paramsRef, gradsRef, expAvgsRef, and expAvgSqsRef.

- Deterministic Computation:
  - aclnnFusedAdam is deterministic by default.

## Invocation Description

| Invocation method  | Sample code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn invocation | [test_aclnn_fused_adam](./examples/arch35/test_aclnn_fused_adam.cpp) | Invoke the FusedAdam operator through the [aclnnFusedAdam](docs/aclnnFusedAdam_en.md) interface. |
