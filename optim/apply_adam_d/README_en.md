# ApplyAdamD

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

- Operator function: Updates the variable "var" according to the Adam algorithm. This operator has the same computation logic as ApplyAdam, but additionally outputs the updated m and v.

- Formula:

  $$
  lr = learning\_rate \times \frac{\sqrt{1 - beta2\_power}}{1 - beta1\_power}
  $$

  $$
  m = m + (1 - beta1) \times (grad - m)
  $$

  $$
  v = v + (1 - beta2) \times (grad \times grad - v)
  $$

  - If use_nesterov = True:

    $$
    var = var - lr \times \frac{m \times beta1 + (1 - beta1) \times grad}{\epsilon + \sqrt{v}}
    $$

  - If use_nesterov = False:

    $$
    var = var - lr \times \frac{m}{\epsilon + \sqrt{v}}
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
    <td>var</td>
    <td>Input</td>
    <td>Parameter tensor to be updated, which should come from Variable.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>m</td>
    <td>Input</td>
    <td>First-moment estimate, which should come from Variable.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>v</td>
    <td>Input</td>
    <td>Second-moment estimate, which should come from Variable.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta1_power</td>
    <td>Input</td>
    <td>Power of beta1, a scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta2_power</td>
    <td>Input</td>
    <td>Power of beta2, a scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate, a scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta1</td>
    <td>Input</td>
    <td>Decay rate of the first-moment estimate, a scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta2</td>
    <td>Input</td>
    <td>Decay rate of the second-moment estimate, a scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>epsilon</td>
    <td>Input</td>
    <td>Small constant for numerical stability, a scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Gradient tensor.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Specifies whether to use a locking mechanism to protect the update operation. Default: False.</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>use_nesterov</td>
    <td>Attribute</td>
    <td>Specifies whether to use Nesterov accelerated gradient. Default: False.</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>var</td>
    <td>Output</td>
    <td>Updated parameter tensor.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>m</td>
    <td>Output</td>
    <td>Updated first-moment estimate.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>v</td>
    <td>Output</td>
    <td>Updated second-moment estimate.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraints

The input tensors must have the same shape.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_apply_adam_d](./examples/test_geir_apply_adam_d.cpp)   | Invokes the ApplyAdamD operator through graph construction with the [operator IR](./op_graph/apply_adam_d_proto.h). |
