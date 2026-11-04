# IndexFill

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: along the given axis dim of input x, fills and replaces the values at the positions specified by indices with value in a non-inplace manner.
- Example:
The input x is:

  &emsp;&emsp;[[1, 2, 3],

  &emsp;&emsp;&nbsp;[4, 5, 6],

  &emsp;&emsp;&nbsp;[7, 8, 9]]

  When dim = 0, indices = [0, 2], and value = 0, the computation result of the operator is:

    &emsp;&emsp;[[0, 0, 0],

    &emsp;&emsp;&nbsp;[4, 5, 6],

    &emsp;&emsp;&nbsp;[0, 0, 0]]

  When dim = 1, indices = [0, 2], and value = 0, the computation result of the operator is:

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
        <th>Input/Output/Attribute</th>
        <th>Description</th>
        <th>Data type</th>
        <th>Data format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>x in the function example, that is, the tensor whose values at the specified positions are to be replaced with value in a non-inplace manner.</td>
        <td>FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64, BOOL, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>Input</td>
        <td>Specifies the indices at which x will be filled on the dim dimension. The element values are smaller than the size of the corresponding dim of x, and negative indices are supported.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Specifies the data value for filling. The data type must be consistent with x.</td>
        <td>Same as x</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>Attribute</td>
        <td>Specifies the dimension of x to be filled. The value range of dim is [-x.dim(), x.dim()).</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>Output tensor after filling, with the shape consistent with input x.</td>
        <td>Same as input x</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_index_fill](../index_fill_d/examples/test_aclnn_index_fill.cpp) | Invokes the IndexFill operator through the [aclnnIndexFill](../index_fill_d/docs/aclnnIndexFill&aclnnInplaceIndexFill_en.md) interface. |
