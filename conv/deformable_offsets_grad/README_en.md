# DeformableOffsetsGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>      |    √     |
| <term>Atlas A2 products</term>       |    √     |

## Function

- Operator function: The DeformableOffsets operator collects the feature sampling points used for convolution according to offsets (offset values of the kernel sampling points) and reorganizes them, so that the Conv2d operator can perform the convolution computation conveniently. DeformableOffsetsGrad is the backward pass of this process.

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
      <td>grad</td>
      <td>Input</td>
      <td>Input gradient.</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input coordinates.</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Input</td>
      <td>Offset values.</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>Output</td>
      <td>Gradient of x.</td>
      <td>DT_FLOAT16, DT_FLOAT</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>grad_offsets</td>
      <td>Output</td>
      <td>Gradient relative to input_offsets.</td>
      <td>DT_FLOAT16, DT_FLOAT</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Strides of the sliding window, a list containing 4 integers.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Padding dimensions, a list containing 4 integers.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Size of the convolution kernel, a list containing 2 integers.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>Attribute</td>
      <td>Scaling factor of each dimension, a list containing 4 integers.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute</td>
      <td>Data format of the input x.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformable_groups</td>
      <td>Attribute</td>
      <td> Grouping on the C axis.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>Attribute</td>
      <td>Optional bool specifying the version of DeformableConv2D. true indicates v2, and false indicates v1. Only " true " is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Sample code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph mode | [test_geir_deformable_offsets_grad](./examples/arch35/test_geir_deformable_offsets_grad.cpp)   | Calls the DeformableOffsetsGrad operator by building a graph through the [operator IR](./op_graph/deformable_offsets_grad_proto.h). |
