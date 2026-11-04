# MatMulCompress

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|×|
|Atlas A3 products|×|
|Atlas A2 products|×|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|√|
|Atlas training products|×|

## Function

- **Operator function**: During matrix multiplication computation, the right matrix can first be losslessly compressed with the msModelSlim tool to reduce memory usage, and then lossless decompression and matrix multiplication computation are completed through this interface.
- **Formula**:

  ```Cpp
  x2_unzip = unzip(x2, compressIndex)
  result = x1 @ x2_unzip + bias
  ```

  Here, x2 is the 1D data of the right matrix after compression with the msModelSlim tool, compressIndex is the information related to the compression algorithm, and x2_unzip is the data after lossless decompression inside the interface (the same as the original right matrix data).

## Parameters

<table style="undefined;table-layout: fixed; width: 869px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 343px">
<col style="width: 114px">
<col style="width: 102px">
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
    <td>Input tensor</td>
    <td>Left input of the matrix multiplication, a 2D tensor.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>Input tensor</td>
    <td>Compressed right input of the matrix multiplication, a 1D tensor.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>compressIndex</td>
    <td>Input tensor</td>
    <td>Compression index table of the right input of the matrix multiplication, a 1D tensor.</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Input tensor</td>
    <td>Bias term; passing a null pointer is supported.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>Output tensor</td>
    <td>Output of the computation result.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- The Reduce dimension sizes of x1 and x2_unzip must be equal.
- All input tensors do not support non-contiguous tensors.
