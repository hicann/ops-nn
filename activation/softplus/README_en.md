# Softplus

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: an activation function used to map any real number to the positive number range.

- Formula:

  $$
  y = log(1 + e^x)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 150px">
<col style="width: 150px">
<col style="width: 300px">
<col style="width: 250px">
<col style="width: 150px">
</colgroup>
<thead>
  <tr>
   <th>Parameter name</th>
   <th>Input/Output</th>
   <th>Description</th>
   <th>Data type</th>
   <th>Data format</th>
  </tr>
  </thead>
  <tbody>
  <tr>
   <td>x</td>
   <td>Input</td>
   <td>Denotes the input of the softplus activation function, x in the formula.</td>
   <td>FLOAT, FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>y</td>
   <td>Output</td>
   <td>Denotes the output of the softplus activation function, y in the formula.</td>
   <td>FLOAT, FLOAT16, BFLOAT16</td>
   <td>ND</td>
  </tr>
  </tbody>
</table>

## Constraints

The shapes and data types of the input x and the output y must be consistent.

## Invocation Description

| Invocation method | Sample code | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_softplus.cpp](examples/test_geir_softplus.cpp) | Invokes the softplus operator by constructing the graph through the [operator IR](./op_graph/softplus_proto.h). |
