# ApplyFtrl

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

- Operator function: Updates the variable "var" according to the Ftrl-proximal scheme.

- Formula:

  $$
  accum\_new = accum + grad \times grad
  $$

  $$
  linear = linear + grad - \frac{accum\_new^{-lr\_power} - accum^{-lr\_power}}{lr} \times var
  $$

  $$
  quadratic = \frac{accum\_new^{-lr\_power}}{lr} + 2 \times l2
  $$

  - If $|linear| > l1$:

    $$
    var = \frac{sign(linear) \times l1 - linear}{quadratic}
    $$

  - Otherwise:

    $$
    var = 0.0
    $$

  $$
  accum = accum\_new
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
    <td>Parameter tensor to be updated. It should come from Variable.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>accum</td>
    <td>Input</td>
    <td>Gradient accumulation value. It should come from Variable.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>linear</td>
    <td>Input</td>
    <td>Correction term. It should come from Variable.</td>
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
    <td>lr</td>
    <td>Input</td>
    <td>Scale factor, scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>l1</td>
    <td>Input</td>
    <td>L1 regularization coefficient, scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>l2</td>
    <td>Input</td>
    <td>L2 regularization coefficient, scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>lr_power</td>
    <td>Input</td>
    <td>Power of the scale factor, scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Whether to use the lock mechanism to protect the update operation. The default is False. Only False is supported.</td>
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
</tbody></table>

## Constraints

`var`, `accum`, `linear`, and `grad` must have the same shape; `lr`, `l1`, `l2`, and `lr_power` must be scalars or single-element tensors (0D and shape `[1]` are supported). All input tensors must have the same data type.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_apply_ftrl](./examples/test_geir_apply_ftrl.cpp)   | Invokes the ApplyFtrl operator through the operator IR-based [graph construction](./op_graph/apply_ftrl_proto.h) method. |
