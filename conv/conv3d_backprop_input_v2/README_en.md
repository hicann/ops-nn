# Conv3DBackpropInputV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Support |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term>|    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
| <term>Atlas inference products</term>    |     ×    |
| <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: computes the gradient $\frac{\partial L}{\partial x}$ of the loss function $L$ with respect to the input tensor $x$ of the forward 3D convolution.

- Formula:

  $$
    \frac{\partial L}{\partial x_{n, c_{in}, d, i, j}} = \sum_{c_{out}=1}^{C_{out}} \sum_{r=1}^{k_D} \sum_{p=1}^{k_H} \sum_{q=1}^{k_W} \frac{\partial L}{\partial y_{n, c_{out}, d-r, i-p, j-q}}\cdot w_{c_{out}, c_{in}, r, p, q}
  $$

## Parameters

| <div style="width:120px">Parameter name</div>  | <div style="width:120px">Input/Output/Attribute</div>  | <div style="width:380px">Description</div> | <div style="width:350px">Data type</div>  | <div style="width:220px">Data format</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------ | ----------------- | --------------------- |
| input_size | Input | <ul><li>One-dimensional tensor that gives the 5D shape of the feature map whose gradient is to be computed.</li><li>The order of the integers in the tensor is determined by the feature map format, and the integers indicate the length of each dimension of the feature map.</li></ul> | INT32, INT64 | - |
| filter | Input   | <ul><li>5D convolution kernel tensor $w_{c_{out}, c_{in}, r, p, q}$.</li><li>$c_{out}$ corresponds to the N dimension of 'filter'.</li><li>$k_H$, $k_W$, and $k_D$ correspond to the H dimension, W dimension, and D dimension of 'filter'.</li></ul> | FLOAT16, BFLOAT16, FLOAT32 | NCDHW, NDHWC, DHWCN |
| out_backprop | Input | <ul><li>5D output gradient tensor, equivalent to $\frac{\partial L}{\partial y_{n, c_{out}, d-r, i-p, j-q}}$ in the formula.</li><li>The data format is the same as that of 'y'.</li></ul> | FLOAT16, FLOAT32, BFLOAT16 | NDHWC, NCDHW |
| strides | Mandatory attribute | <ul><li>A tuple or list of 5 integers that specifies the stride of the sliding window along each dimension of the feature map.</li><li>The axis order is consistent with the feature map format.</li></ul> | - | - |
| pads | Mandatory attribute | <ul><li>A tuple or list of 6 integers that specifies the padding amounts of the feature map in each direction.</li><li>Padding is performed only along the depth (D), height (H), and width (W) dimensions.</li><li>By setting appropriate padding values for each direction, you can implement the "SAME" and "VALID" padding modes.</li></ul> | - | - |
| dilations  | Optional attribute | <ul><li>A tuple or list of 5 integers that indicates the dilation (hole) factor of each dimension of the feature map.</li><li>The default value is [1,1,1,1,1].</li><li>The axis order is consistent with the feature map format.</li></ul> | - | - |
| groups  | Optional attribute | <ul><li>An integer in the range [1,65535], with the default value 1.</li><li>Indicates the number of grouped connections from $c_{in}$ to $c_{out}$.</li><li>$c_{in}$ and $c_{out}$ must be divisible by 'groups'.</li><li>For the supported combinations of different 'groups' values and dtypes, see the table below.</li></ul> | INT | - |
| data_format  | Optional attribute | <ul><li>A string whose value must be one of ["NDHWC","NCDHW"], with the default value "NDHWC". The correspondence is batch (N), depth (D), height (H), width (W), and channels (C).</li><li>Specifies the data layout format of 'out_backprop' and 'y'.</li></ul> | STRING | - |
| y  | Output | <ul><li>5D tensor in the same format as 'out_backprop', that is, $\frac{\partial L}{\partial x_{n, c_{in}, d, i, j}}$ in the formula.</li></ul> | FLOAT16, BFLOAT16, FLOAT32 | NDHWC, NCDHW |

- Feature map: the [输入特征图'x'](../conv3d_v2/README.md) in the forward convolution.
- Combinations of different groups values and dtypes

    | groups |        dtype           | out_backprop format | filter format |     y format   |
    |--------|------------------------|---------------------|---------------|----------------|
    | =1     |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      NCDHW    |     NCDHW      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      NDHWC    |     NCDHW      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      DHWCN    |     NCDHW      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|       NDHWC         |      NDHWC    |     NDHWC      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|       NDHWC         |      NCDHW    |     NDHWC      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|       NDHWC         |      DHWCN    |     NDHWC      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      NCDHW    |     NCDHW      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      NDHWC    |     NCDHW      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      DHWCN    |     NCDHW      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|       NDHWC         |      NDHWC    |     NDHWC      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|       NDHWC         |      NCDHW    |     NDHWC      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|       NDHWC         |      DHWCN    |     NDHWC      |

## Constraints

* input_size
    - The following axis sequences can be input:
        - [batch, in_depth, in_height, in_width, in_channels]
        - [batch, in_channels, in_depth, in_height, in_width]
* out_backprop
    - The values of the N, C, D, H, and W dimensions must be in the range [1,2147483647].
* filter
    - The values of the N (out_channels), C (in_channels), and D dimensions must be in the range [1,2147483647].
    - The values of the H and W dimensions must be in the range [1,511].
* strides
    - The N and C dimensions must be 1.
    - Ascend 950PR&950DT products: the values of the D, H, and W dimensions must be in the range [1,2147483647].
    - Atlas A2 products and Atlas A3 products: the values of the H and W dimensions must be in the range [1,63], and the value of the D dimension must be in the range [1,255].
* pads
    - The padding order is [front, back, top, bottom, left, right].
    - Ascend 950PR&950DT products: the values of the D, H, and W dimensions must be in the range [0,2147483647].
    - Atlas A2 products and Atlas A3 products: the values of the D, H, and W dimensions must be in the range [0,255].
* dilations
    - The N and C dimensions must be 1.
    - Ascend 950PR&950DT products: the values of the D, H, and W dimensions must be in the range [1,2147483647].
    - Atlas A2 products and Atlas A3 products: the values of the H, W, and D dimensions must be in the range [1,255].
* Due to hardware resource limitations, the operator fails to execute in some parameter value combination scenarios. Analyze the log information and troubleshoot the problem. If the problem cannot be solved, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

| Invocation method  | Sample code  | Description                 |
| -----------  | ------------------- | ---------- |
| aclnn interface   | [test_aclnn_conv3d_backprop_input_v2](examples/test_aclnn_conv3d_backprop_input_v2.cpp)  | Invokes the Conv3DBackpropInputV2 operator through the [aclnnConvolutionBackward](../convolution_backward/docs/aclnnConvolutionBackward_en.md) interface |
