# Conv3DBackpropFilterV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Support |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term>|    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                 |    ×     |
| <term>Atlas inference products</term>                        |    ×     |
| <term>Atlas training products</term>                         |    ×     |

## Function

- Operator function: Computes the gradient $\frac{\partial L}{\partial w}$ of the 3D convolution weight tensor $w$.

- Formula:

  $$
  \frac{\partial L}{\partial w_{c_{out}, c_{in}, r, p, q}} = \sum_{n=1}^{N} \sum_{d=1}^{D_{out}} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, d \cdot s_D + r, i \cdot s_H + p, j \cdot s_W + q} \cdot \frac{\partial L}{\partial y_{n, c_{out}, d, i, j}}
  $$

  Here, $r$, $p$, and $q$ are the indexes of the convolution weight on the D, H, and W dimensions respectively. The convolution weight sizes $k_D$, $k_H$, and $k_W$ are given by 'filter_size'.

  Assume that the shape of $x$ is ($N,C_{in},D_{in},H_{in},W_{in}$) and the shape of $out\_backprop$ is ($N,C_{out},D_{out},H_{out},W_{out}$). In the transposed convolution scenario (output_padding is non-zero), the relationship between them and the convolution strides ($s_D$,$s_H$,$s_W$), convolution weight sizes ($k_D$,$k_H$,$k_W$), dilation parameter ($dilation$), padding parameter ($pads$), and output extra sizes ($output\_padding$) is as follows:

  $$
  D_{in} = (D_{out} - 1) \cdot s_D - (pads[0] + pads[1]) + dilation[0] \cdot (k_D - 1) + output\_padding[0] + 1
  $$

  $$
  H_{in} = (H_{out} - 1) \cdot s_H - (pads[2] + pads[3]) + dilation[1] \cdot (k_H - 1) + output\_padding[1] + 1
  $$

  $$
  W_{in} = (W_{out} - 1) \cdot s_W - (pads[4] + pads[5]) + dilation[2] \cdot (k_W - 1) + output\_padding[2] + 1
  $$

  Here, $output\_padding[0]$, $output\_padding[1]$, and $output\_padding[2]$ are the values of the output_padding attribute on the D, H, and W dimensions respectively (the N and C dimensions must be 0). $pads[0]$ to $pads[5]$ correspond to the padding amounts in the front, back, top, bottom, left, and right directions in order.

## Parameters

| <div style="width:120px">Parameter name</div>  | <div style="width:120px">Input/Output/Attribute</div>  | <div style="width:350px">Description</div> | <div style="width:350px">Data type</div>  | <div style="width:220px">Data format</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------------- | ----------------- | --------------------- |
| x | Input | <ul><li>A 5D tensor, which is the feature map and corresponds to $x_{n, c_{in}, d \cdot s_D + r, i \cdot s_H + p, j \cdot s_W + q}$ in the formula.</li><li>$N$ corresponds to the N dimension of 'x'.</li><li>$c_{in}$ corresponds to the C dimension of 'x'.</li><li>The data format must be the same as that of 'out_backprop'.</li></ul> | FLOAT16, FLOAT32, BFLOAT16 | NDHWC, NCDHW |
| filter_size | Input | <ul><li>A 1D tensor, which represents the shape of the convolution weight. The convolution weight is a 5D tensor.</li><li>The order of the integers in the tensor is determined by the convolution weight format. These integers indicate the length of each dimensions of the convolution weight.</li></ul> | INT32, INT64 | - |
| out_backprop | Input | <ul><li>A 5D tensor, which corresponds to $\frac{\partial L}{\partial y_{n, c_{out}, d, i, j}}$ in the formula.</li><li>$c_{out}$ corresponds to the C dimension of 'out_backprop'.</li><li>$H_{out}$, $W_{out}$, and $D_{out}$ correspond to the H dimension, W dimension, and D dimension of 'out_backprop'.</li><li>The data format must be the same as that of 'x'.</li></ul> | FLOAT16, FLOAT32, BFLOAT16 | NDHWC, NCDHW |
| strides | Required attribute | <ul><li>A tuple or list containing 5 integers, which specifies the stride of the sliding window along each dimension of the feature map.</li><li>The axis order is the same as the feature map format.</li><li>$s_H$, $s_W$, and $s_D$ correspond to the H dimension, W dimension, and D dimension of the 'strides' attribute.</li></ul> | - | - |
| pads | Required attribute | <ul><li>A tuple/list containing 6 integers, which specifies the padding amount of the feature map in each direction.</li><li>Padding is performed only along the D, H, and W dimensions.</li><li>By setting proper padding values for each direction, you can implement both the "SAME" and "VALID" padding modes.</li></ul> | - | - |
| dilations | Optional attribute | <ul><li>A tuple/list containing 5 integers, which indicates the dilation (hole) factor of each input dimension.</li><li>The axis order is the same as the feature map format.</li><li>Default value: [1,1,1,1,1].</li></ul> | -  | -  |
| groups | Optional attribute  |  <ul><li>An integer. Valid range: [1,65535]. Default value: 1.</li><li>Indicates the number of grouped connections from $c_{in}$ to $c_{out}$.</li><li>Currently, both $c_{in}$ and $c_{out}$ must be divisible by 'groups'.</li><li>When 'groups' is not 1, HIFLOAT8 is not supported.</li></ul>  | INT | - |
| data_format | Optional attribute | <ul><li>A string. The value must be one of ["NDHWC","NCDHW"]. Default value: "NDHWC". Meanings of the letters: batch(N), depth(D), height(H), width(W), channels(C).</li><li>Specifies the data layout format of 'x' and 'out_backprop'.</li></ul> | STRING | - |
| enable_hf32 | Optional attribute | <ul><li>A Boolean value. Default value: false. Controls whether to enable the HF32 computation mode.</li><li>Takes effect only when the input data type is FLOAT32.</li></ul> | BOOL | - |
| output_padding | Optional attribute | <ul><li>A tuple/list containing 5 integers, which specifies the extra sizes of the output shape.</li><li>Takes effect only in the transposed convolution scenario and is used to supplement the output sizes.</li><li>Default value: [0,0,0,0,0].</li><li>Only <term>Ascend 950PR&950DT products</term> support non-zero values.</li></ul> | - | - |
| y | Output | <ul><li>Corresponds to $\frac{\partial L}{\partial w_{c_{out}, c_{in}, r, p, q}}$ in the formula.</li><li>Its shape is given by 'filter_size'.</li></ul>  |  FLOAT32  | NCDHW, NDHWC, DHWCN |

## Constraints

* x
    - The value range of the N, C, D, H, and W dimensions must be [1,2147483647].
* filter_size
    - The value range of the W and H dimensions must be [1,511].
    - The following axis sequences can be input:
        - [out_channels, in_channels, filter_depth, filter_height, filter_width]
        - [out_channels, filter_depth, filter_height, filter_width, in_channels]
* out_backprop
    - The value range of the N, C, D, H, and W dimensions must be [1,2147483647].
* strides
    - The N and C dimensions must be 1.
    - Ascend 950PR&950DT products: The value range of the D, H, and W dimensions must be [1,2147483646].
    - Atlas A2 products and Atlas A3 products: The value range of the H and W dimensions must be [1,63], and the value range of the D dimension must be [1,255].
* pads
    - The padding order is: [front, back, top, bottom, left, right].
    - Ascend 950PR&950DT products: The value range of the H, W, and D dimensions must be [1,2147483646].
    - Atlas A2 products and Atlas A3 products: The value range of the H, W, and D dimensions must be [0,255].
* dilations
    - The N and C dimensions must be 1.
    - Ascend 950PR&950DT products: The value range of the H, W, and D dimensions must be [1,2147483646].
    - Atlas A2 products and Atlas A3 products: The value range of the H, W, and D dimensions must be [1,255].
* output_padding
    - Only <term>Ascend 950PR&950DT products</term> support non-zero values. Other products support only all-zero values.
    - The value range of the D, H, and W dimensions must be [0,2147483646].
    - In the transposed convolution scenario, each dimension value of output_padding must be less than the corresponding stride or dilation.
* Due to hardware resource limits, the operator fails to execute under some parameter value combinations. Analyze and troubleshoot the problem according to the log information. If you cannot resolve the problem, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

| Invocation method  | Sample code  | Description                 |
| -----------  | ------------------- | ---------- |
| aclnn API   | [test_aclnn_conv3d_backprop_filter_v2](examples/test_aclnn_conv3d_backprop_filter_v2.cpp)  | Calls the Conv3DBackpropFilterV2 operator through the [aclnnConvolutionBackward](../convolution_backward/docs/aclnnConvolutionBackward_en.md) API |
