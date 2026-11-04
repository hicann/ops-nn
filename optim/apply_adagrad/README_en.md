# ApplyAdagrad

English | [简体中文](./README.md)

<!-- codespell:ignore accum adagrad -->

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: ApplyAdagrad is the core update operation of the adaptive gradient algorithm, mainly used to update model parameters in optimizers (such as stochastic gradient descent).

- Formula:
  $$
  \begin{align}
  accum = update\_slots ? accum + grad * grad : accum\\
  var -= lr * grad * (1 / sqrt(accum))
  \end{align}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>The variable tensor.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Input</td>
      <td>The variable tensor. It has the same type as "var", and is updated in place when update_slots is true.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Input</td>
      <td>Has the same type as "var".</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>The gradient tensor. It has the same type as "var".</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>The updated variable tensor.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update_slots</td>
      <td>Attribute</td>
      <td>Specifies whether to update "accum"; the default value is true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>Specifies whether to use a lock. Currently, there is no special concurrency control implementation; the default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The shapes of var, accum, and grad must be the same, and lr must be a scalar.
- The data types of var, accum, lr, and grad must be the same.
- The use_locking attribute is integrated as a compatibility attribute; currently, there is no special concurrency control implementation.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_apply_adagrad.cpp](examples/test_geir_apply_adagrad.cpp)  | Calls the ApplyAdagrad operator by building a graph through the [operator IR](op_graph/apply_adagrad_proto.h).         |
