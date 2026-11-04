# ConvolutionBackward

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Support |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term>|    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
| <term>Atlas inference products</term>    |     √    |
| <term>Atlas training products</term>    |    √    |

## Function

- Operator function: Backpropagation of convolution. Computes the gradients of the input, weight, and bias according to the output mask settings. This function supports 1D, 2D, and 3D convolutions.

- Formula

 Assume that the shape of the input is ($N,C_{in},D_{in},H_{in},W_{in}$) and the shape of the output is ($N,C_{out},D_{out},H_{out},W_{out}$). The relationship between them and the convolution stride ($stride$), convolution kernel size ($kernelSize，kD,kH,kW$), and dilation parameter ($dilation$) is as follows:

  $$
    D_{out}=\lfloor \frac{D_{in}+2*padding[0]-dilation[0] * (kernelSize[0] - 1) - 1}{stride[0]}+1 \rfloor
  $$

  $$
    H_{out}=\lfloor \frac{H_{in}+2*padding[1]-dilation[1] * (kernelSize[1] - 1) - 1}{stride[1]}+1 \rfloor
  $$

  $$
    W_{out}=\lfloor \frac{W_{in}+2*padding[2]-dilation[2] * (kernelSize[2] -1) -1}{stride[2]}+1 \rfloor
  $$

  Convolution backpropagation needs to compute the gradients of the forward convolution input tensor $x$, convolution kernel weight tensor $w$, and bias $b$.

  - For the gradient $\frac{\partial L}{\partial x}$ of $x$:

    $$
    \frac{\partial L}{\partial x_{n, c_{in}, i, j}} = \sum_{c_{out}=1}^{C_{out}} \sum_{p=1}^{k_H} \sum_{q=1}^{k_W} \frac{\partial L}{\partial y_{n, c_{out}, i-p, j-q}}\cdot w_{c_{out}, c_{in}, p, q}
    $$

    Here, $L$ is the loss function, and $\frac{\partial L}{\partial y}$ is the gradient of the output tensor $y$ with respect to $L$.

  - For the gradient $\frac{\partial L}{\partial w}$ of $w$:

    $$
    \frac{\partial L}{\partial w_{c_{out}, c_{in}, p, q}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, i \cdot s_H + p, j \cdot s_W + q} \cdot \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$

  - For the gradient $\frac{\partial L}{\partial b}$ of $b$:

    $$
    \frac{\partial L}{\partial b_{c_{out}}} = \sum_{n=1}^{N}       \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$

## Parameters

| <div style="width:120px">Parameter name</div>  | <div style="width:120px">Input/Output/Attribute</div>  | <div style="width:350px">Description</div> | <div style="width:350px">Data type</div>  | <div style="width:220px">Data format</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------------- | ----------------- | --------------------- |
| gradOutput | Input | <ul><li>Gradient of the output tensor $y$ with respect to $L$, which corresponds to $\frac{\partial L}{\partial y}$ in the formula.</li><li>The data format must be the same as that of 'input' and 'gradInput'.</li></ul> | FLOAT, FLOAT16, BFLOAT16 | NCL, NCHW, NCDHW |
| input | Input | <ul><li>Input tensor $x$ of the forward convolution in the formula.</li><li>The data format must be the same as that of 'gradOutput' and 'gradInput'.</li></ul> | FLOAT, FLOAT16, BFLOAT16 | NCL, NCHW, NCDHW |
| weight | Input | <ul><li>Convolution kernel weight tensor $w$ in the formula.</li><li>The data format must be the same as that of 'gradWeight'.</li></ul> | FLOAT, FLOAT16, BFLOAT16 | NCL, NCHW, NCDHW |
| biasSizes | Input | <ul><li>Shape of the bias in the forward convolution. The array length is 1.</li><li>In a normal convolution, the value equals [weight.shape[0]]. In a transposed convolution, the value equals [weight.shape[1] * groups].</li><li>In the empty tensor scenario, when outputMask specifies that the gradient of the bias needs to be computed, biasSizes cannot be nullptr.</li></ul> | INT64 | - |
| stride | Input | <ul><li>Stride at which the convolution kernel moves over the input during backpropagation, which corresponds to stride[0], stride[1], and stride[2] in the formula.</li><li>The array length is the dimension of weight minus 2, and the values must be greater than 0.</li></ul> | INT64 | - |
| padding | Input | <ul><li>Padding applied to the input during backpropagation, which corresponds to padding[0], padding[1], and padding[2] in the formula.</li><li>The array length can be the dimension of weight minus 2. In the 2D scenario, the array length can be 4. The values must be greater than or equal to 0.</li></ul> | INT64 | - |
| dilation | Input | <ul><li>Dilation parameter during backpropagation, which corresponds to dilation[0], dilation[1], and dilation[2] in the formula.</li><li>The array length can be the dimension of weight minus 2. The values must be greater than 0.</li></ul> | INT64 | - |
| transposed | Input | <ul><li>Flag for enabling transposed convolution. When the value is True, transposed convolution is enabled.</li></ul> | - | - |
| outputPadding | Input | <ul><li>Padding applied to the output during backpropagation. The array length can be the dimension of weight minus 2, and the value range of each dimension satisfies [0, value of the corresponding stride dimension). When transposed is False, each element value must be 0.</li></ul> | INT64 | - |
| groups | Input | <ul><li>Number of groups into which the input channels are divided during backpropagation.</li><li>groups * the C dimension of weight = the C dimension of input must hold, and the value range of groups is [1,65535].</li><li>The C dimension of gradOutput must be greater than or equal to groups and divisible by groups.</li></ul> | INT32 | - |
| gradInput | Output | <ul><li>Gradient of the input tensor $x$ with respect to $L$, which corresponds to $\frac{\partial L}{\partial x}$ in the formula.</li><li>The data type is the same as that of 'input'.</li><li>The data format must be the same as that of 'input' and 'gradOutput'.</li></ul> | FLOAT, FLOAT16, BFLOAT16 | NCL, NCHW, NCDHW |
| gradWeight | Output | <ul><li>Gradient of the convolution kernel weight tensor $w$ with respect to $L$, which corresponds to $\frac{\partial L}{\partial w}$ in the formula.</li><li>The data format must be the same as that of 'weight'.</li></ul> | FLOAT, FLOAT16, BFLOAT16 | NCL, NCHW, NCDHW |
| gradBias | Output | <ul><li>Gradient of the bias $b$ with respect to $L$, which corresponds to $\frac{\partial L}{\partial b}$ in the formula.</li><li>The data type is the same as that of 'gradOutput'.</li></ul> | FLOAT, FLOAT16, BFLOAT16 | ND |

* <term>Ascend 950PR&950DT products</term>:
    - Only when transposed=true and output_mask[0]=true, the data types HIFLOAT8 and FLOAT8_E4M3FN are supported.
* <term>Atlas inference products</term>, <term>Atlas training products</term>:
    - BFLOAT16, HIFLOAT8, and FLOAT8_E4M3FN are not supported.
    - For the gradOutput and weight parameters, empty tensors are not supported.
* <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - HIFLOAT8 and FLOAT8_E4M3FN are not supported.

## Constraints

* <term>Atlas inference products</term>, <term>Atlas training products</term>: Currently, only the backpropagation of 1D and 2D convolutions is supported. The backpropagation of 3D convolutions is not yet supported.
* For more detailed constraints, see the [aclnnConvolutionBackward](docs/aclnnConvolutionBackward_en.md) API documentation.

## Invocation Description

| Invocation method  | Sample code  | Description                 |
| -----------  | ------------------- | ---------- |
| aclnn API   | [test_aclnn_conv_backward_1d](examples/test_aclnn_conv_backward_1d.cpp)  | Calls the Conv3DBackpropFilterV2 and Conv3DBackpropInputV2 operators through the [aclnnConvolutionBackward](docs/aclnnConvolutionBackward_en.md) API |
| aclnn API   | [test_aclnn_conv_backward_2d](examples/test_aclnn_conv_backward_2d.cpp)  | Calls the Conv3DBackpropFilterV2 and Conv3DBackpropInputV2 operators through the [aclnnConvolutionBackward](docs/aclnnConvolutionBackward_en.md) API |
| aclnn API   | [test_aclnn_conv_backward_3d](examples/test_aclnn_conv_backward_3d.cpp)  | Calls the Conv3DBackpropFilterV2 and Conv3DBackpropInputV2 operators through the [aclnnConvolutionBackward](docs/aclnnConvolutionBackward_en.md) API |
