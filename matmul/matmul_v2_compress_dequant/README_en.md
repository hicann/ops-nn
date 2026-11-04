# MatMulV2CompressDequant

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

- **Operator function**: when performing matrix multiplication computation, you can first use the msModelSlim tool to perform lossless compression on the right matrix to reduce memory usage, and then use this interface to complete the lossless decompression, matrix multiplication, and dequantization computations.
- **Formula**:

  ```text
  x2_unzip = unzip(x2, compressIndex, compressInfo)
  result = (x1 @ x2_unzip + bias) * deqScale
  ```

  Here x2 is the 1D data obtained after the right matrix is compressed by the msModelSlim tool, and x2_unzip is the data after the lossless decompression performed inside the interface (consistent with the original right matrix data).

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
    <td>The left input of the matrix multiplication, a 2D tensor.</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>Input tensor</td>
    <td>The compressed right input of the matrix multiplication, a 1D tensor.</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>compressIndex</td>
    <td>Input tensor</td>
    <td>The compression index table of the right input of the matrix multiplication, a 1D tensor.</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Input tensor</td>
    <td>The bias term. Passing a null pointer is supported.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>deqScale</td>
    <td>Input tensor</td>
    <td>The dequantization parameter. The data type is UINT64.</td>
    <td>UINT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>offsetW</td>
    <td>Input tensor</td>
    <td>The offset of the right input of the matrix multiplication. Currently, only a null pointer can be passed.</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>offsetX</td>
    <td>Input attribute</td>
    <td>The offset of the left input of the matrix multiplication. Currently, only 0 is supported.</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>compressInfo</td>
    <td>Input array</td>
    <td>Information about the compressed data, including the compressed block information and the original shape.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>Output tensor</td>
    <td>The computation result output.</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- The Reduce dimension sizes of x1 and x2_unzip must be equal.
- Non-contiguous tensors are not supported for all input tensors.
- deqScale requires converting the original float parameters to the UINT64 data format.
- Currently, offsetW supports only a null pointer, and offsetX supports only 0.
