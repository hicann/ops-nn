# IndexFillD

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Description: Along the given axis dim of the input self, replaces the values at the positions specified by index with value.
- Example:
The input self is:

  &emsp;&emsp;[[1, 2, 3],

  &emsp;&emsp;&nbsp;[4, 5, 6],

  &emsp;&emsp;&nbsp;[7, 8, 9]]

  When dim = 0, index = [0, 2], and value = 0, the computation result of the operator is:

    &emsp;&emsp;[[0, 0, 0],

    &emsp;&emsp;&nbsp;[4, 5, 6],

    &emsp;&emsp;&nbsp;[0, 0, 0]]

  When dim = 1, index = [0, 2], and value = 0, the computation result of the operator is:

    &emsp;&emsp;[[0, 2, 0],

    &emsp;&emsp;&nbsp;[0, 5, 0],

    &emsp;&emsp;&nbsp;[0, 8, 0]]

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
        <th>Input/Output</th>
        <th>Description</th>
        <th>Data type</th>
        <th>Data format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>Input</td>
        <td>The self in the function example, that is, the tensor whose values at specified positions are to be replaced with value.</td>
        <td>FLOAT16, FLOAT, INT32, INT64, BOOL, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Input</td>
        <td>Specifies the dimension of self to be filled. When self has 1-8 dimensions, the value range of dim is [-self.dim(), self.dim()); when self has 0 dimensions, the value range of dim is [-1, 1).</td>
        <td>int64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>Input</td>
        <td>Specifies the indices where self is to be filled along the dim dimension. The element values are smaller than the dimension size of self along dim.</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Specifies the data value to fill. It must be convertible to the data type of self.</td>
        <td>Consistent with self</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>The specified output tensor.</td>
        <td>Consistent with self</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_index_fill_tensor](./examples/test_aclnn_index_fill_tensor.cpp) | Invokes the IndexFillD operator through the [aclnnIndexFillTensor](./docs/aclnnIndexFillTensor&aclnnInplaceIndexFillTensor_en.md) interface. |
