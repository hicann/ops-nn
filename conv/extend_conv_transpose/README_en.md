# ExtendConvTranspose

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term>  |    ×     |
| <term>Atlas A2 products</term>   |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                   |    ×     |
| <term>Atlas inference products</term>                           |    ×     |
| <term>Atlas training products</term>                           |    ×     |

## Function

- Operator function: computes the 3D transposed convolution (deconvolution), used in quantization inference scenarios.

- Formula:

  Assume that the input shape is ($N,C_{in},D_{in},H_{in},W_{in}$) and the output shape is ($N,C_{out},D_{out},H_{out},W_{out}$). Their relationships with the convolution stride ($stride$), padding ($pads$), kernel size ($kernel\_size$, kD, kH, kW), and dilation parameter ($dilation$) are:

  $$
    D_{out}=(D_{in} - 1) * stride[0] - (pads[0] + pads[1]) + dilation[0] * (kernel\_size[0] - 1) + output\_padding[0] + 1
  $$

  $$
    H_{out}=(H_{in} - 1) * stride[1] - (pads[2] + pads[3]) + dilation[1] * (kernel\_size[1] - 1) + output\_padding[1] + 1
  $$

  $$
    W_{out}=(W_{in} - 1) * stride[2] - (pads[4] + pads[5]) + dilation[2] * (kernel\_size[2] - 1) + output\_padding[2] + 1
  $$

## Parameters

| <div style="width:120px">Parameter name</div>  | <div style="width:120px">Input/Output/Attribute</div>  | <div style="width:380px">Description</div> | <div style="width:350px">Data type</div>             | <div style="width:220px">Data format</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------------------ |-------------------------------------------------|-------------------------------------|
| input_size | Input  | <ul><li>A 1-D tensor that denotes the shape of the input tensor, that is, the shape of the forward convolution input (the output 'y' of this operator).</li></ul> | INT32, INT64                                     | -                                   |
| x  | Input | <ul><li>Feature map, equivalent to ($N,C_{in},D_{in},H_{in},W_{in}$) in the formula.</li><li>A tensor whose data format is consistent with 'y'.</li></ul> | INT8                   | NCDHW                         |
| filter | Input | <ul><li>A 5D tensor that denotes the convolution kernel weight, with the shape ($C_{out}$, $C_{in}$/groups, kD, kH, kW), where kD, kH, and kW correspond to kernelSize[0], kernelSize[1], and kernelSize[2] in the formula.</li></ul> | INT8                   | NCDHW                         |
| bias | Optional input | <ul><li>A 1-D tensor, the convolution bias tensor, stored in the order of $out\_channels$.</li></ul> | INT32                           | ND                                  |
| scale | Optional input | <ul><li>A 1-D tensor, the channel-wise quantization/dequantization parameter corresponding to the output (channelwise).</li></ul> | UINT64                                          | ND                                  |
| strides | Required attribute | <ul><li>A tuple or list of 5 integers that specifies the stride of the sliding window on each dimension of input tensor 'x'.</li><li>The axis order is consistent with the feature map format.</li></ul> | -                                               | -                                   |
| pads | Required attribute | <ul><li>A tuple or list of 6 integers that specifies the padding amount in each direction, corresponding to pads[0] to pads[5] in the formula.</li></ul> | -                                               | -                                   |
| dilations | Optional attribute | <ul><li>A tuple or list of 5 integers that denotes the dilation (hole) factor of each input dimension, corresponding to dilation[0], dilation[1], and dilation[2] in the formula.</li><li>The default value is [1,1,1,1,1].</li><li>The axis order is consistent with the feature map format.</li></ul>| -                                               | -                                   |
| groups | Optional attribute | <ul><li>An integer in the range [1,65535], with the default value 1.</li><li>It denotes the number of grouped connections from $c_{in}$ to $c_{out}$.</li><li>$c_{in}$ and $c_{out}$ must be divisible by 'groups'.</li></ul> | INT                                             | -                                   |
| data_format | Optional attribute | <ul><li>A string. Currently only the value "NCDHW" is supported. The mappings are batch (N), channels (C), depth (D), height (H), and width (W).</li><li>It specifies the data layout format of 'x' and 'y'.</li></ul> | STRING                                          | -                                   |
| output_padding | Optional attribute | <ul><li>The size additionally added to the end of the output shape, with the default value [0,0,0,0,0].</li><li>It corresponds to output_padding[0], output_padding[1], and output_padding[2] in the formula.</li></ul> | -                                   | -                                   |
| offset_x  | Optional attribute | <ul><li>The default value is 0. This is a reserved field.</li></ul> | INT                                             | -                                   |
| fusion_mode | Optional attribute | <ul><li>An integer that takes the value 0 or 1, with the default value 0. It indicates whether ReLU is enabled for the output: 0 means disabled, and 1 means enabled.</li></ul> | INT                                             | -                                   |
| y_quant_mode | Optional attribute | <ul><li>The default value is 0. This is a reserved field.</li></ul> | INT                                             | -                                   |
| y | Output | <ul><li>Equivalent to ($N,C_{out},D_{out},H_{out},W_{out}$) in the formula.</li><li>The data format is consistent with 'x'.</li></ul> | FLOAT16, INT8 | NCDHW                         |

## Constraints

* input_size
    - The available axis sequence is as follows:
        - [batch, in_channels, in_depth, in_height, in_width]
* x
    - The value ranges of the N, C, D, H, and W dimensions must be within [1,2147483647].
* filter
    - The value ranges of the N (out_channels), C (in_channels/groups), and D dimensions must be within [1,2147483647].
    - The lengths of the kernel_height (H) and kernel_width (W) dimensions must be within [1,511].
* strides
    - The N and C dimensions must be 1.
    - The value ranges of the D, H, and W dimensions must be within [1,2147483647].
* pads
    - The padding order is [front, back, top, bottom, left, right].
    - The value ranges of the D, H, and W dimensions must be within [0,2147483647].
* dilations
    - The N and C dimensions must be 1.
    - The value ranges of the D, H, and W dimensions must be within [1,2147483647].
* output_padding
    - The N and C dimensions must be 0; additions are allowed only along the depth, height, and width directions.
    - The value ranges of the D, H, and W dimensions must be within [0,2147483647].
* Due to hardware resource limitations, the operator may fail to execute for some parameter value combinations. Analyze and troubleshoot the problem according to the log information. If the problem cannot be solved, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

This operator has no corresponding aclnn interface. Only when the Conv2DTranspose operator is invoked in graph mode and the input type is INT8 is it converted to ExtendConvTranspose for the quantization-related implementation.
