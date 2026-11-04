# InplaceUpdate

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function

- Operator function: updates the values in updates into x according to indices, implementing the in-place update operation.

- Formula:

$$
y = x.copy();\quad y[\text{indices}[i], ...] = v[i, ...]
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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor to be updated, supporting 1D~8D.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>One-dimensional vector of the int32 type, specifying the indices to be updated.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>Input</td>
      <td>Update value tensor, with the same dtype as x; the size of the 0th dimension equals the length of indices.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Updated tensor, with the same shape as x and the same dtype as x.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

The input and output support only the ND format, the dtypes of x and v must be the same, v.shape[0] must equal the length of indices, and x.shape[1:] must equal v.shape[1:]. When duplicate values exist in indices, the result is non-deterministic.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_inplace_update.cpp">test_geir_inplace_update</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete the operator compilation and verification.</td>
  </tr>
</tbody>
</table>
