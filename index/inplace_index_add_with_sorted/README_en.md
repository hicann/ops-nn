# InplaceIndexAddWithSorted

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √   |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |
| <term>Kirin X90 processor series products</term>                        |    √   |
| <term>Kirin 9030 processor series products</term>                       |    √   |

## Function

- Operator function: Multiplies alpha and value, and accumulates the result into the specified axis dimension of the var tensor in the order of sorted_indices.

- Formula:

  $$
  var[indices]+= alpha * updates
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 60px">
  <col style="width: 60px">
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 60px">
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
      <td>`var` in the formula, representing the tensor to be accumulated.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>Input</td>
      <td>`updates` in the formula, representing the tensor to be added to var.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_indices</td>
      <td>Input</td>
      <td>`indices` in the formula, representing the sorted indices.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pos</td>
      <td>Input</td>
      <td>Represents the position of each sorted index in the original index sequence.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Optional input</td>
      <td><ul><li>`alpha` in the formula, representing the scaling of value. It is multiplied with value and then accumulated into var.</li><li>The default value is 1.0.</li></ul></td>
      <td>FLOAT, INT32, INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td><ul><li>Specifies the axis of var along which the addition is performed.</li><li>Currently, only 0 is supported.</li></ul></td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: the data types of var and value do not support BFLOAT16.
- Ascend 950PR&950DT products: the data types of var and value support only FLOAT16 and BFLOAT16, and the data type of alpha supports only FLOAT.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_inplace_index_add_with_sorted](./examples/test_geir_inplace_index_add_with_sorted.cpp) | Invokes the InplaceIndexAddWithSorted operator through the operator IR-based [graph construction](./op_graph/inplace_index_add_with_sorted_proto.h) method. |
| aclnn invocation | [test_aclnn_index_add](../inplace_scatter_add/examples/test_aclnn_index_add.cpp) | Invokes the InplaceIndexAddWithSorted operator through the [aclnnIndexAdd](../inplace_scatter_add/docs/aclnnIndexAdd_en.md) interface. |
