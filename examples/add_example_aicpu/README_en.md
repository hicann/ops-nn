# AddExample

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|×|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|

## Function

- Operator function: performs the addition computation.

- Formula:

$$
y = x1 + x2
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
      <td>x1</td>
      <td>Input</td>
      <td>The input parameter for the add_example computation, the x1 in the formula.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The input parameter for the add_example computation, the x2 in the formula.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output parameter for the add_example computation, the y in the formula.</td>
      <td>FLOAT, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn invocation</td>
    <td><a href="../../examples/add_example/examples/test_aclnn_add_example.cpp">test_aclnn_add_example_aicpu</a></td>
    <td rowspan="2">See <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_add_example.cpp">test_geir_add_example_aicpu</a></td>
  </tr>
</tbody>
</table>
