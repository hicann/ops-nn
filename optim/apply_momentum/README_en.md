# ApplyMomentum

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

- Operator function: Updates the variable "var" according to the momentum scheme. If use_nesterov=True is set, Nesterov momentum is used.

- Formula:

  $$
  accum = accum \times momentum + grad
  $$

  - If use_nesterov = True:

    $$
    var = var - grad \times lr + accum \times momentum \times lr
    $$

  - If use_nesterov = False:

    $$
    var = var - lr \times accum
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
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate, scalar.</td>
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
    <td>momentum</td>
    <td>Input</td>
    <td>Momentum coefficient, scalar.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>use_nesterov</td>
    <td>Attribute</td>
    <td>Whether to use Nesterov momentum. The default is False.</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Whether to use the lock mechanism to protect the update operation. The default is False.</td>
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

The input tensors must have the same shape and type.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_apply_momentum](./examples/test_geir_apply_momentum.cpp)   | Invokes the ApplyMomentum operator through the operator IR-based [graph construction](./op_graph/apply_momentum_proto.h) method. |
