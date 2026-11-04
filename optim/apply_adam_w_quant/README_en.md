# aclnnApplyAdamWQuant

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function:

  Takes the m and v input to the optimizer as indices, retrieves the values from their respective qmaps, and multiplies them by the absmax corresponding to each blockSize to perform dequantization. Then it implements the adamW optimizer function. For the updated m and v, one maximum value is taken from each blockSize, and each blockSize of m and v corresponds to one absmax. A norm normalization is performed, and binary search is used to find the indices of the corresponding m and v in their respective qmaps as outputs. The absmax is also the input of the next round of quantization.

- Formula:

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$

  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
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

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>Input/Output</td>
      <td>theta in the formula, the weight to be computed; it is both an input and an output.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Input gt in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>m</td>
      <td>Input/Output</td>
      <td>The index value of parameter m in the formula before quantization.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>v</td>
      <td>Input/Output</td>
      <td>The index value of parameter v in the formula before quantization.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qmap_m</td>
      <td>Input</td>
      <td>The quantization mapping table of parameter m in the formula, sorted in ascending order.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qmap_v</td>
      <td>Input</td>
      <td>The quantization mapping table of parameter v in the formula, sorted in ascending order.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>absmax_m</td>
      <td>Input/Output</td>
      <td>For the computation input, this dequantization stage denormalizes the quantile (multiplied by absmaxMRef); for the computation output, it is the parameter of this quantization and of the next round of dequantization.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>absmax_v</td>
      <td>Input/Output</td>
      <td>For the computation input, this dequantization stage denormalizes the quantile (multiplied by absmaxVRef); for the computation output, it is the parameter of this quantization and of the next round of dequantization.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>step</td>
      <td>Input</td>
      <td><ul><li>The number of iterations, t in the formula.</li><li>The value range is positive integers greater than 1.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Attribute</td>
      <td><ul><li>The learning rate, eta in the formula.</li><li>The value range is 0~1.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta1</td>
      <td>Attribute</td>
      <td><ul><li>The beta1 parameter in the formula.</li><li>The value range is 0~1.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta2</td>
      <td>Attribute</td>
      <td><ul><li>The beta2 parameter in the formula.</li><li>The value range is 0~1.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight_decay</td>
      <td>Attribute</td>
      <td><ul><li>The weight decay coefficient, the lambda parameter in the formula.</li><li>The value range is 0~1.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td><ul><li>The epsilon parameter in the formula, added to the denominator to prevent division by 0.</li><li>The value range is 1e-8.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gnorm_scale</td>
      <td>Attribute</td>
      <td><ul><li>The parameter that scales input parameter grad.</li><li>The value range is 0~1.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>Attribute</td>
      <td><ul><li>The size of each block.</li><li>The value range is fixed to 256.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td><ul><li>Reserved parameter.</li><li>Reserved parameter.</li></ul></td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_apply_adam_w_quant](./examples/test_aclnn_apply_adam_w_quant.cpp) | Calls the ApplyAdamWQuant operator through the [aclnnApplyAdamWQuant](./docs/aclnnApplyAdamWQuant_en.md) interface.    |
