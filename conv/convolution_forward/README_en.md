# ConvolutionForward

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>      |    √     |
| <term>Atlas A2 products</term>       |    √     |

## Function

- Operator function: implements convolution. It supports 1D convolution, 2D convolution, and 3D convolution, and also supports transposed convolution, dilated convolution, and grouped convolution.
  When the input parameter is `transposed = True`, the transposed convolution or fractional-strided convolution is used. You can regard it as the gradient or the inverse operation of a normal convolution, that is, it restores the input shape from the output shape of the convolution while keeping a connection pattern compatible with the convolution. Its parameters are similar to those of a normal convolution, including the number of input channels, the number of output channels, the kernel size, stride, padding, output padding, groups, bias, dilation, and so on.

- Formula:

  Assume that the shape of input is $(N, C_{\text{in}}, H, W)$, the shape of weight is $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$, and the shape of output is $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$. Then the output is expressed as:

  $$
    \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
  $$

  In this formula, $\star$ denotes the convolution computation, which depends on the dimensions of the convolution input and on the convolution type (dilated convolution or grouped convolution). $N$ denotes the batch size, $C$ denotes the number of channels, and $W$ and $H$ denote the width and the height respectively. The formulas for computing the corresponding output dimensions are as follows:

  $$
    H_{\text{out}}=[(H + 2 * padding[0] - dilation[0] * (K_h - 1) - 1 ) / stride[0]] + 1 \\
    W_{\text{out}}=[(W + 2 * padding[1] - dilation[1] * (K_w - 1) - 1 ) / stride[1]] + 1
  $$

## Parameters

| <div style="width:120px">Parameter name</div>  | <div style="width:120px">Input/Output/Attribute</div>  | <div style="width:350px">Description</div> | <div style="width:350px">Data type</div>  | <div style="width:220px">Data format</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------------- | ----------------- | --------------------- |
| input | Input | <ul><li>input in the formula, which denotes the convolution input.</li><li>The data type must satisfy the data type deduction rules with weight ([mutual deduction relationship](../../docs/en/context/deduction_relationship.md) and <a href="#constraints">Constraints</a>).</li><li>N≥0, C≥1, and the other dimensions≥0.</li></ul> | FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN| NCL, NCHW, NCDHW |
| weight | Input | <ul><li>weight in the formula, which denotes the convolution weight.</li><li>The data type must satisfy the data type deduction rules with input ([mutual deduction relationship](../../docs/en/context/deduction_relationship.md) and <a href="#constraints">Constraints</a>).</li><li>All dimensions are ≥1.</li></ul> | FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN | NCL, NCHW, NCDHW |
| bias | Input | <ul><li>bias in the formula, which denotes the convolution bias.</li><li>When transposed=false, bias is one-dimensional and its value equals the first dimension of weight; when transposed=true, bias is one-dimensional and its value equals weight.shape[1] * groups.</li></ul> | FLOAT, FLOAT16, BFLOAT16 | ND, NCL, NCHW, NCDHW |
| stride | Input | <ul><li>Scanning stride of the convolution.</li><li>The array length must equal the number of dimensions of input minus 2, and the value must be greater than 0.</li></ul> | INT64 | - |
| padding | Input | <ul><li>Padding applied to input.</li><li>Array length: 1 or 2 for non-transposed conv1d; 2 or 4 for conv2d; 3 for conv3d. The value must be greater than or equal to 0.</li></ul> | INT64 | - |
| dilation | Input | <ul><li>Spacing between the elements in the convolution kernel.</li><li>The array length must equal the number of dimensions of input minus 2, and the value must be greater than 0.</li></ul> | INT64 | - |
| transposed | Input | <ul><li>Specifies whether the convolution is transposed.</li></ul> | BOOL | - |
| outputPadding | Input | <ul><li>In the transposed convolution case, padding applied to all sides of the output.</li><li>In the non-transposed convolution case, this configuration is ignored. The value must be greater than or equal to 0 and less than the value of stride or dilation in the corresponding dimension.</li></ul> | INT64 | - |
| groups | Input | <ul><li>Number of block connections from the input channels to the output channels.</li><li>The value must be greater than 0 and must satisfy: the C dimension of weight divided by groups equals the C dimension of input.</li></ul> | INT64 | - |
| output | Output | <ul><li>out in the formula, which denotes the convolution output.</li><li>The data type must be consistent with the data type deduced from input and weight.</li><li>The number of channels equals the first dimension of weight, and the other dimensions are ≥0.</li></ul> | FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN | NCL, NCHW, NCDHW |
| cubeMathType | Input | <ul><li>Used to determine which computation logic the Cube unit should use for the operation.</li><li>0 (KEEP_DTYPE): keeps the input data type for computation.</li><li> 1 (ALLOW_FP32_DOWN_PRECISION): allows FLOAT32 computation with reduced precision to improve performance.</li><li> 2 (USE_FP16): uses FLOAT16 precision for computation.</li><li> 3 (USE_HF32): uses HF32 (mixed precision) for computation.</li></ul> | INT8 | - |

* <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - The data types of input and weight do not support HIFLOAT8 or FLOAT8_E4M3FN.
    - The data type of bias does not support HIFLOAT8 or FLOAT8_E4M3FN. The data type must be consistent with input and weight.
    - In the forward scenarios of conv1d, conv2d, and conv3d, bias is converted to FLOAT for computation.
    - In the conv2d and conv3d transposed=true scenarios, the sizes of weight H and W must be in the range [1,255], and the other dimensions must be greater than or equal to 1.
    - In the conv1d transposed=true scenario, the size of weight L must be in the range [1,255], and the other dimensions must be greater than or equal to 1.
    - In the forward conv3d scenario, the sizes of weight H and W must be in the range [1,511].
    - When cubeMathType is 0 (KEEP_DTYPE) and the input is FLOAT, this option is currently not supported.
    - When cubeMathType is 1 (ALLOW_FP32_DOWN_PRECISION) and the input is FLOAT, conversion to HFLOAT32 for computation is allowed.
    - When cubeMathType is 2 (USE_FP16) and the input is BFLOAT16, this option is not supported.
    - When cubeMathType is 3 (USE_HF32) and the input is FLOAT, the input is converted to HFLOAT32 for computation.
* <term>Ascend 950PR&950DT products</term>:
    - The data types of input and weight support FLOAT, FLOAT16, BFLOAT16, and HIFLOAT8.
    - When transposed=true: the data type of input additionally supports FLOAT8_E4M3FN, the N dimension supports a value greater than or equal to 0, and the sizes of all other dimensions must be greater than or equal to 1.
    - When transposed=false: when the data type of input is HIFLOAT8, the data type of weight must be consistent with input. The N dimension supports a value greater than or equal to 0; the D, H, and W dimensions support values greater than or equal to 0 (a value of 0 is supported only when the deduced D, H, and W dimensions of output are also 0); the C dimension supports a value greater than or equal to 0 (a value of 0 is supported only when one of the deduced N, C, D, H, and W dimensions of output is 0).
    - The data type of bias does not support HIFLOAT8 or FLOAT8_E4M3FN.
    - When transposed=true: when the data types of input and weight are HIFLOAT8 or FLOAT8_E4M3FN, bias is not supported.
    - When transposed=false: when the data types of input and weight are HIFLOAT8, bias is converted to FLOAT for computation.
    - When cubeMathType is 1 (ALLOW_FP32_DOWN_PRECISION) and the input is FLOAT, conversion to HFLOAT32 for computation is allowed.
    - When cubeMathType is 2 (USE_FP16) and the input is BFLOAT16, this option is not supported.
    - When cubeMathType is 3 (USE_HF32) and the input is FLOAT, the input is converted to HFLOAT32 for computation.

## Constraints

* <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: the size of each dimension of each group of tensors in input, weight, and bias must not be greater than 1000000.
* <term>Ascend 950PR&950DT products</term>: in the scenario where transposed is true, 1D, 2D, and 3D convolutions are supported, and empty tensors are supported.
* Due to hardware resource limits, the operator fails to execute for certain combinations of parameter values. Analyze and troubleshoot the problem according to the log information. If the problem cannot be resolved, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

| Invocation method | Sample code | Description |
| -------- | -------- | ---- |
| aclnn invocation | [test_aclnn_convolution_1d_transpose](examples/test_aclnn_convolution_1d_transpose.cpp) | Demonstrates the 1D transposed convolution scenario through the [aclnnConvolution](docs/aclnnConvolution_en.md) API, which invokes the Conv2DTranspose operator |
| aclnn invocation | [test_aclnn_convolution_2d_transpose](examples/test_aclnn_convolution_2d_transpose.cpp) | Demonstrates the 2D transposed convolution scenario through the [aclnnConvolution](docs/aclnnConvolution_en.md) API, which invokes the Conv2DTranspose operator |
| aclnn invocation | [test_aclnn_convolution_3d_transpose](examples/test_aclnn_convolution_3d_transpose.cpp) | Demonstrates the 3D transposed convolution scenario through the [aclnnConvolution](docs/aclnnConvolution_en.md) API, which invokes the Conv3DTransposeV2 operator |
| aclnn invocation | [test_aclnn_conv_depthwise_2d](examples/test_aclnn_conv_depthwise_2d.cpp) | Demonstrates 2D depthwise separable convolution using the [aclnnConvDepthwise2d](docs/aclnnConvDepthwise2d_en.md) API, which invokes the Conv2D operator |
| aclnn invocation | [test_aclnn_conv_tbc](examples/test_aclnn_conv_tbc.cpp) | Demonstrates time-batch-channel convolution using the [aclnnConvTbc](docs/aclnnConvTbc_en.md) API, which invokes the Conv2D operator |
| aclnn invocation | [test_aclnn_quant_convolution](examples/test_aclnn_quant_convolution.cpp) | Demonstrates 3D quantized convolution using the [aclnnQuantConvolution](docs/aclnnQuantConvolution_en.md) API, which invokes the Conv3DV2 operator |
