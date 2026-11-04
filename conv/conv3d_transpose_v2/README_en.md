# Conv3DTransposeV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Compute the gradient of the transpose (deconvolution) of 3D convolution with respect to the input.

- Formula:

  Assume that the input shape is ($N,C_{in},D_{in},H_{in},W_{in}$) and the output shape is ($N,C_{out},D_{out},H_{out},W_{out}$). Their relationships with the convolution stride ($stride$), kernel size ($kernel\_size, kD, kH, kW$), and dilation parameter ($dilation$) are as follows:

  $$
    D_{out}=(D_{in} - 1) * stride[0] - 2*padding[0] + dilation[0] * (kernel\_size[0] - 1) + output\_padding[0] + 1
  $$

  $$
    H_{out}=(H_{in} - 1) * stride[1] - 2*padding[1] + dilation[1] * (kernel\_size[1] - 1) + output\_padding[1] + 1
  $$

  $$
    W_{out}=(W_{in} - 1) * stride[2] - 2*padding[2] + dilation[2] * (kernel\_size[2] - 1) + output\_padding[2] + 1
  $$

## Parameters

| <div style="width:120px">Parameter name</div>  | <div style="width:120px">Input/Output/Attribute</div>  | <div style="width:380px">Description</div> | <div style="width:350px">Data type</div>  | <div style="width:220px">Data format</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------------------- | ----------------- | --------------------- |
| input_size | Input  | <ul><li>1D tensor that indicates the shape of the input tensor.</li></ul> | INT32, INT64 | - |
| x  | Input | <ul><li>Feature map, corresponding to ($N,C_{in},D_{in},H_{in},W_{in}$) in the formula.</li><li>A tensor whose data format is the same as that of 'y'.</li></ul> | FLOAT16, FLOAT32, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN| NDHWC, NCDHW |
| filter | Input | <ul><li>A 5D tensor that represents the convolution kernel size, corresponding to kernelSize[0], kernelSize[1], and kernelSize[2] in the formula.</li></ul> | FLOAT16, BFLOAT16, FLOAT32, HIFLOAT8, FLOAT8_E4M3FN | NCDHW, NDHWC, DHWCN |
| bias | Optional attribute | <ul><li>A 1D tensor. Currently, this parameter is not enabled.</li></ul> | FLOAT16, FLOAT32 | - |
| offset_w | Optional attribute | <ul><li>A 1D tensor used for quantized deconvolution. Currently, this parameter is not enabled.</li></ul> | INT8 | - |
| strides | Required attribute | <ul><li>A tuple or list of 5 integers that specifies the stride of the sliding window along each dimension of input tensor 'x'.</li><li>The axis order is consistent with the feature map format.</li></ul>  | - | - |
| pads | Required attribute | <ul><li>A tuple or list of 6 integers that specifies the padding amount in each direction, corresponding to padding[0], padding[1], and padding[2] in the formula.</li></ul> | - | - |
| dilations | Optional attribute | <ul><li>A tuple or list of 5 integers that indicates the dilation (hole) factor of each dimension of the input, corresponding to dilation[0], dilation[1], and dilation[2] in the formula</li><li>Default value: [1,1,1,1,1].</li><li>The axis order is consistent with the feature map format.</li></ul>| - | - |
| groups | Optional attribute | <ul><li>An integer ranging from [1,65535], with a default value of 1.</li><li>Indicates the number of grouped connections from $c_{in}$ to $c_{out}$.</li><li>$c_{in}$ and $c_{out}$ must be divisible by 'groups'.</li><li>For the supported combinations of different 'groups' values and dtype, see the table below.</li></ul> | INT | - |
| data_format | Optional attribute | <ul><li>A string. The value must be one of ["NDHWC","NCDHW"], with a default value of "NDHWC". The correspondence is batch (N), depth (D), height (H), width (W), and channels (C).</li><li>Specifies the data layout format of 'x' and 'y'.</li></ul> | STRING | - |
| output_padding | Optional attribute | <ul><li>Additional size added to the end of the output shape. Default value: [0,0,0,0,0].</li><li>Corresponding to output_padding[0], output_padding[1], and output_padding[2] in the formula.</li></ul> | - | - |
| offset_x  | Optional attribute | <ul><li>Default value: 0. Reserved field.</li></ul> | INT | - |
| y | Output | <ul><li>Corresponding to ($N,C_{out},D_{out},H_{out},W_{out}$) in the formula.</li><li>The data format is the same as that of 'x'.</li></ul> | FLOAT16, BFLOAT16, FLOAT32, HIFLOAT8, FLOAT8_E4M3FN | NDHWC, NCDHW |

- Description of combinations of different groups values and dtype

    | groups |        dtype           |   x format  | filter format |    y format    |
    |--------|------------------------|-------------|---------------|----------------|
    | =1     |HIFLOAT8/FLOAT8_E4M3FN  |    NCDHW    |      NCDHW    |     NCDHW      |
    | =1     |HIFLOAT8/FLOAT8_E4M3FN  |    NDHWC    |      NCDHW    |     NDHWC      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|    NCDHW    |      NCDHW    |     NCDHW      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|    NCDHW    |      NDHWC    |     NCDHW      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|    NCDHW    |      DHWCN    |     NCDHW      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|    NDHWC    |      NDHWC    |     NDHWC      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|    NDHWC    |      NCDHW    |     NDHWC      |
    | =1     |FLOAT16/BFLOAT16/FLOAT32|    NDHWC    |      DHWCN    |     NDHWC      |
    | \>1    |HIFLOAT8/FLOAT8_E4M3FN  |    NCDHW    |      NCDHW    |     NCDHW      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|    NCDHW    |      NCDHW    |     NCDHW      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|    NCDHW    |      NDHWC    |     NCDHW      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|    NCDHW    |      DHWCN    |     NCDHW      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|    NDHWC    |      NDHWC    |     NDHWC      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|    NDHWC    |      NCDHW    |     NDHWC      |
    | \>1    |FLOAT16/BFLOAT16/FLOAT32|    NDHWC    |      DHWCN    |     NDHWC      |

## Constraints

* input_size
    - The inputtable axis sequences are as follows:
        - [batch, in_depth, in_height, in_width, in_channels]
        - [batch, in_channels, in_depth, in_height, in_width]
* x
    - The values of the N, C, D, H, and W dimensions must be in the range [1,2147483647].
* filter
    - When the data format is DHWCN or NDHWC, HIFLOAT8 and FLOAT8_E4M3FN are not supported.
    - The values of the N (out_channels), C (in_channels), and D dimensions must be in the range [1,2147483647].
    - The values of the H and W dimensions must be in the range [1,511].
* strides
    - The N and C dimensions must be 1.
    - Ascend 950PR&950DT products: The values of the D, H, and W dimensions must be in the range [1,2147483647].
    - Atlas A2 products and Atlas A3 products: The values of the H and W dimensions must be in the range [1,63], and the value of the D dimension must be in the range [1,255].
* pads
    - The padding order is [front, back, top, bottom, left, right].
    - Ascend 950PR&950DT products: The values of the D, H, and W dimensions must be in the range [0,2147483647].
    - Atlas A2 products and Atlas A3 products: The values of the D, H, and W dimensions must be in the range [0,255].
* dilations
    - The N and C dimensions must be 1.
    - Ascend 950PR&950DT products: The values of the D, H, and W dimensions must be in the range [1,2147483647].
    - Atlas A2 products and Atlas A3 products: The values of the D, H, and W dimensions must be in the range [1,255].
* output_padding
    - The N and C dimensions must be 0. Addition is allowed only in the depth, height, and width directions.
    - The values of the D, H, and W dimensions must be in the range [0,2147483647].
* Due to hardware resource limitations, the operator fails to execute for some parameter value combinations. Analyze and troubleshoot the problem based on the log information. If the problem cannot be solved, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

| Invocation method | Sample code | Description |
| -----------  | ------------------- | ---------- |
| aclnn API | [test_aclnn_conv3d_transpose_v2](examples/test_aclnn_conv3d_transpose_v2.cpp)  | Invoke the Conv3DTransposeV2 operator through the [aclnnConvolution](../convolution_forward/docs/aclnnConvolution_en.md) API. |
