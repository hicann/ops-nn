# DeformableOffsets

English | [简体中文](./README.md)

## Applicable Products

| Product                                                        | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>      |    √     |
| <term>Atlas A2 products</term>       |    √     |

## Function

- Operator function: a function used to compute the output of Deformable Convolution. By introducing the offset parameter offsets, the positions of the convolution kernel on the input feature map can be dynamically adjusted, thereby adapting to irregular geometric variations.
- Computation formula:

  Assume that the shape of input input is [N, inH, inW, inC] and the shape of output deformOut is [N, outH\*K_H, outW\*K_W, inC]. Compute outH and outW based on the existing attributes:

  $$
  outH = (inH + pads[0] + pads[1] - ((K\_H - 1) * dilations[1] + 1)) // strides[1] + 1
  $$

  $$
  outW = (inW + pads[2] + pads[3] - ((K\_W - 1) * dilations[2] + 1)) // strides[2] + 1
  $$

  Sampling point subscripts of the standard convolution computation:

  $$
  x = -pads[2] + ow*strides[2] + kw*dilations[2], \text{ where } ow \text{ ranges over } (0, outW-1) \text{ and } kw \text{ ranges over } (0, K\_W-1)
  $$

  $$
  y = -pads[0] + oh*strides[1] + kh*dilations[1], \text{ where } oh \text{ ranges over } (0, outH-1) \text{ and } kh \text{ ranges over } (0, K\_H-1)
  $$

  Perform the deformable convolution based on the input `offsets`, and compute the subscripts after offsetting:

  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  Use bilinear interpolation to compute the values of the offset points:

  $$
  (x_{0}, y_{0}) = (int(x), int(y)) \\
  (x_{1}, y_{1}) = (x_{0} + 1, y_{0} + 1)
  $$

  $$
  weight_{00} = (x_{1} - x) * (y_{1} - y) \\
  weight_{01} = (x_{1} - x) * (y - y_{0}) \\
  weight_{10} = (x - x_{0}) * (y_{1} - y) \\
  weight_{11} = (x - x_{0}) * (y - y_{0}) \\
  $$

  $$
  deformOut(x, y) = mask * (weight_{00} * input(x0, y0) + weight_{01} * input(x0, y1) + weight_{10} * input(x1, y0) + weight_{11} * input(x1, y1))
  $$

  where `input` denotes the input tensor (corresponding to `x` in the parameter table), `offsets` denotes the offset values, `deformOut` denotes the output tensor (corresponding to `y` in the parameter table), `strides` specifies the strides in the height and width directions, `pads` specifies the padding in the height and width directions, `ksize` specifies the convolution kernel size (corresponding to `K_H` and `K_W` in the formula), `dilations` specifies the dilation factor of each dimension, `offsetX` and `offsetY` denote the offset components in `offsets`, and `mask` is the third component of `offsets` at the corresponding convolution kernel position (always present when modulated=true). The offset sampling point (x, y) and its four interpolation neighbors may fall outside the input image boundaries, and the kernel actually processes them as "out-of-bounds values are taken as 0".

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
      <td>Denotes the original input data. It corresponds to `input` in the formula.</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Input</td>
      <td>Denotes the offset values. It corresponds to `offsets` in the formula.</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output of the deformable convolution. It corresponds to `deformOut` in the formula.</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Denotes the strides of the convolution kernel in the height and width directions, that is, (1, stridesH, stridesW, 1). It corresponds to `strides` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Specifies the padding added in the height and width directions of the input. It corresponds to `pads` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Specifies the size of the convolution kernel. It corresponds to `K_H` and `K_W` in the formula. size is 2 (K_H, K_W), and each element is greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>Attribute</td>
      <td>Specifies the dilation factor of each dimension, that is, (1, dilationsH, dilationsW, 1). It corresponds to `dilations` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute</td>
      <td>Specifies the data format of input x.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformable_groups</td>
      <td>Attribute</td>
      <td>Specifies the number of groups of input x along the C axis.</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>Attribute</td>
      <td>Specifies the version of the deformable convolution, indicating whether the offset contains a mask. If it is true, `offset` contains a mask; if it is false, `offset` does not contain a mask. Currently, only true is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Sample code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph mode | [test_geir_deformable_offsets](./examples/test_geir_deformable_offsets.cpp)   | Calls the DeformableOffsets operator by constructing a graph through the [operator IR](./op_graph/deformable_offsets_proto.h). |
