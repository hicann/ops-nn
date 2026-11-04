# DeformableConv2d

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Implements the convolution function, supports 2D convolution, and also supports deformable convolution and grouped convolution.

- Formula:

  Assume the shape of the input (input) is [N, inH, inW, inC] and the shape of the output (out) is [N, outH, outC, outW]. Compute outH and outW from the existing parameters:

  $$
  outH = (inH + padding[0] + padding[1] - ((K_H - 1) * dilation[2] + 1)) // stride[2] + 1
  $$

  $$
  outW = (inW + padding[2] + padding[3] - ((K_W - 1) * dilation[3] + 1)) // stride[3] + 1
  $$

  Subscripts of the sampling points for standard convolution computation:

  $$
  x = -padding[2] + ow*stride[3] + kw*dilation[3], kw\ takes\ values\ in\ (0, K\_W-1)
  $$

  $$
  y = -padding[0] + oh*stride[2] + kh*dilation[2], kh\ takes\ values\ in\ (0, K\_H-1)
  $$

  Based on the passed-in offset, perform deformable convolution and compute the shifted subscripts:

  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  Use bilinear interpolation to compute the value of the shifted point:

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
  deformOut(x, y) = weight_{00} * input(x0, y0) + weight_{01} * input(x0,y1) + weight_{10} * input(x1, y0) + weight_{11} * input(x1,y1)
  $$

  Perform the convolution computation to obtain the final output:

  $$
  \text{out}(N_i, C_{\text{out}_j}) = \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{deformOut}(N_i, k)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>Represents the original input data, corresponding to `input` in the formula. The shape is [N, inH, inW, inC], where inH * inW cannot exceed 2147483647.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Represents the 4D tensor of learnable filters, corresponding to `weight` in the formula. The data type and data format are the same as those of input parameter `x`. The shape is [outC, K_H, K_W, inC/groups].</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Input</td>
      <td>Represents the 4D tensor of x-y coordinate offsets and the mask, corresponding to `offset` in the formula. The data type and data format are the same as those of input parameter `x`. When `modulated` is true, the shape is [N, outH, outW, 3 * deformable_groups * K_H * K_W]; when `modulated` is false, the shape is [N, outH, outW, 2 * deformable_groups * K_H * K_W].</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Represents the 1D tensor of the additional bias of the filter output; it is a reserved parameter. The data type is the same as that of input parameter `x`. It is a null pointer when not needed, and the shape is [outC] when present.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kernel_size</td>
      <td>Optional attribute</td>
      <td>Represents the convolution kernel size, corresponding to `K_H` and `K_W` in the formula. The size is 2 (K_H, K_W), all elements are greater than zero, K_H * K_W cannot exceed 2048, and K_H * K_W * inC/groups cannot exceed 65535.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Optional attribute</td>
      <td>Represents the sliding window step of each input dimension, corresponding to `stride` in the formula. The size is 4, all elements are greater than zero, and the dimension order is interpreted according to the data format of `x`. The N and C dimensions must be set to 1.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Optional attribute</td>
      <td>Represents the number of pixels added to each side of the input (top, bottom, left, right), corresponding to `padding` in the formula. The size is 4.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Attribute</td>
      <td><ul><li>Represents the dilation coefficient of each input dimension, corresponding to `dilation` in the formula. The size is 4, all elements are greater than zero, and the dimension order is interpreted according to the data format of x. The N and C dimensions must be set to 1.</li><li>The default value is {1, 1, 1, 1}.</li></ul></td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>Attribute</td>
      <td><ul><li>Represents the number of grouped connections from input channels to output channels. Both inC and outC must be divisible by the value of `groups`, and the value of `groups` is greater than zero.</li><li>The default value is 1.</li></ul></td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformable_groups</td>
      <td>Attribute</td>
      <td><ul><li>Represents the number of deformable group partitions. inC must be divisible by the value of `deformable_groups`, and the value of `deformable_groups` is greater than zero.</li><li>The default value is 1.</li></ul></td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>Attribute</td>
      <td><ul><li>Specifies whether `offset` contains a mask. If true, `offset` contains a mask; if false, it does not.</li><li>The default value is true, and currently only true can be configured.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Represents the computed output tensor, corresponding to `out` in the formula. The data type and data format are the same as those of `x`. The shape is [N, outH, outC, outW].</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>deform_out</td>
      <td>Output</td>
      <td>Represents the computed output tensor, corresponding to `deformOut` in the formula. The data type and data format are the same as those of `x`. The shape is [N, outH * K_H, outW * K_W, inC].</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_deformable_conv2d](examples/test_aclnn_deformable_conv2d.cpp) | Calls the DeformableConv2d operator through the [aclnnDeformableConv2d](docs/aclnnDeformableConv2d_en.md) interface. |
