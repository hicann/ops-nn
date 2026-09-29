# aclnnConvolution

 [📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/conv/convolution_forward)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                      |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>      |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>      |    √     |
|  <term>Atlas 200I/500 A2 inference products</term>                     |     √    |
|  <term>Atlas inference products</term>                             |     √    |
|  <term>Atlas training products</term>                             |     ×    |

## Function

- Description: Implements convolution operations, supporting 1D, 2D, and 3D convolutions, as well as transposed convolution, dilated convolution, and grouped convolution.
  When `transposed = True`, the operator performs transposed convolution (also known as fractionally-strided convolution). This can be viewed as the gradient or inverse operation of regular convolution: it maps the output shape back to the input shape while maintaining a connection pattern compatible with convolution. Its parameters are similar to those of regular convolution, including input channels, output channels, kernel size, stride, padding, output padding, groups, bias, and dilation.

- Formula:

  Assume the input tensor has shape $(N, C_{\text{in}}, D, H, W)$, the weight tensor has shape $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$, and the output tensor has shape $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$, where $N$ denotes the batch size, $C$ denotes the number of channels, $D$, $H$, and $W$ denote the depth, height, and width of the sample respectively, and $K_d$, $K_h$, and $K_w$ denote the depth, height, and width of the kernel respectively. The output is then expressed as:

  $$
  \text{output}(N_i, C_{\text{out}_j}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}) = \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k) + \text{bias}(C_{\text{out}_j})
  $$

  Where $\star$ denotes the convolution operation, whose exact computation depends on the input dimensionality and the convolution type (dilated or grouped). $N$ denotes the batch size, $C$ denotes the number of channels, and $D$, $H$, and $W$ denote the depth, height, and width, respectively. The formulas for computing the corresponding output dimensions are as follows:

  - When `transposed = False`:

    $$
    D_{\text{out}}=[(D + 2 \times padding[0] - dilation[0] \times (K_d - 1) - 1 ) / stride[0]] + 1 \\
    H_{\text{out}}=[(H + 2 \times padding[1] - dilation[1] \times (K_h - 1) - 1 ) / stride[1]] + 1 \\
    W_{\text{out}}=[(W + 2 \times padding[2] - dilation[2] \times (K_w - 1) - 1 ) / stride[2]] + 1
    $$

  - When `transposed = True`:

    $$
    D_{\text{out}}=(D - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
              \times (K_d - 1) + \text {outputPadding}[0] + 1 \\
    H_{\text{out}}=(H - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
              \times (K_h - 1) + \text {outputPadding}[1] + 1 \\
    W_{\text{out}}=(W - 1) \times \text{stride}[2] - 2 \times \text{padding}[2] + \text{dilation}[2]
              \times (K_w - 1) + \text {outputPadding}[2] + 1
    $$

## Prototype

Each operator has <a href="../../../docs/en/context/two_phase_api.md">two-phase API calls</a>. You must call aclnnConvolutionGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnConvolution to perform the computation.

```cpp
aclnnStatus aclnnConvolutionGetWorkspaceSize(
    const aclTensor       *input,
    const aclTensor       *weight,
    const aclTensor       *bias,
    const aclIntArray     *stride,
    const aclIntArray     *padding,
    const aclIntArray     *dilation,
    bool                   transposed,
    const aclIntArray     *outputPadding,
    const int64_t          groups,
    aclTensor             *output,
    int8_t                 cubeMathType,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnConvolution(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnConvolutionGetWorkspaceSize

- **Parameters**

  <table>
  <tr>
  <th style="width:220px">Parameter Name</th>
  <th style="width:120px">Input/Output</th>
  <th style="width:300px">Description</th>
  <th style="width:350px">Usage Instruction</th>
  <th style="width:212px">Data Type</th>
  <th style="width:180px">Data Format</th>
  <th style="width:120px">Dimension (shape) </th>
  <th style="width:145px">Non-contiguous Tensor</th>
  </tr>
  <tr>
  <td>input (aclTensor*) </td>
  <td>Input</td>
  <td>input in the formula, indicating the convolution input.</td>
  <td><ul><li>Empty tensors are supported. </li><li>Its data type and the data type of weight must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN</td>
  <td>NCL, NCHW, NCDHW</td>
  <td>3–5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight (aclTensor*) </td>
  <td>Input</td>
  <td>weight in the formula, indicating the convolution weight.</td>
  <td><ul><li>Empty tensors are supported. </li><li>Its data type and the data type of input must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN</td>
  <td>NCL, NCHW, NCDHW</td>
  <td>3–5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>bias (aclTensor*) </td>
  <td>Input</td>
  <td>bias in the formula, indicating the convolution bias.</td>
  <td><ul><li>If there is no bias, pass nullptr. </li><li>When transposed is set to false, the value is a one-dimensional array and the number of elements is the same as that of the first dimension of weight. When transposed is set to true, the value is a one-dimensional array and the number of elements is the same as that of weight.shape[1] * groups. Only the ND format is supported.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16</td>
  <td>ND, NCL, NCHW, NCDHW</td>
  <td>1–5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>stride (aclIntArray*) </td>
  <td>Input</td>
  <td>Convolution stride.</td>
  <td>The array length must be equal to the number of input dimensions minus 2, and each value must be greater than 0.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>padding (aclIntArray*) </td>
  <td>Input</td>
  <td>Padding added to input.</td>
  <td>Array length: 1 or 2 for non-transposed conv1d; 1 for transposed conv1d; 2 or 4 for conv2d; 3 for conv3d. Values must be greater than or equal to 0.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>dilation (aclIntArray*) </td>
  <td>Input</td>
  <td>Spacing between elements in the convolution kernel.</td>
  <td>The array length must be equal to the number of input dimensions minus 2, and each value must be greater than 0.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>transposed (bool) </td>
  <td>Input</td>
  <td>Whether to perform a transposed convolution.</td>
  <td>-</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>outputPadding (aclIntArray*) </td>
  <td>Input</td>
  <td>Padding added to all sides of the output in transposed convolution mode.</td>
  <td>Ignore this configuration in non-transposed convolution mode. The array length must equal the input rank minus 2. Values must be greater than or equal to 0 and less than the value of the corresponding dimension of stride or dilation.</td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>groups (int64_t) </td>
  <td>Input</td>
  <td>Number of groups that connect input channels to output channels.</td>
  <td>The value must be in the range [1, 65535], and the following condition must be met: groups × C dimension of weight = C dimension of input.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>output (aclTensor*) </td>
  <td>Output</td>
  <td>out in the formula, indicating the convolution output.</td>
  <td><ul><li>Empty tensors are supported. </li><li> The data type must be the same as that of the input and weight derivation. </li><li>The number of channels equals the first dimension of weight, and all other dimensions must be greater than or equal to 0.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN</td>
  <td>NCL, NCHW, NCDHW</td>
  <td>3–5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>cubeMathType (int8_t) </td>
  <td>Input</td>
  <td>Computation logic to be used by the Cube unit.</td>
  <td><ul>
    <li>If the input data types have a <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">mutual inference relationship</a>, this parameter is applied to the inferred data type by default.</li>
    <li>Enumerated values: </li></ul>
    <ul><li> 0 (KEEP_DTYPE): retains the input data type for calculation. </li></ul><ul><li> 1 (ALLOW_FP32_DOWN_PRECISION): allows FLOAT to reduce precision calculation and improve performance. </li></ul><ul><li> 2 (USE_FP16): The FLOAT16 precision is used for calculation. </li></ul><ul><li> 3 (USE_HF32): HFLOAT32 (mixed precision)</li></ul></td>
  <td>INT8</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>workspaceSize (uint64_t*) </td>
  <td>Output</td>
  <td>Size of the workspace required to be allocated on the device.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor (aclOpExecutor**) </td>
  <td>Output</td>
  <td>Operator executor, containing the operator computation flow.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>

- **Return Value**

  aclnnStatus: status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Code</a>.

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:1050px">
    </colgroup>
   <thead>
  <tr>
  <td><strong>Return Value</strong></td>
  <td>Error Code</td>
  <td><strong>Description</strong></td>
  </tr></thead>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">The input pointer is null.</td>
  </tr>
  <tr>
  <td rowspan="11" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="11" align="left">161002</td>
  <td align="left">The data types and formats of input, weight, bias, and output are not supported.</td>
  </tr>
  <tr><td align="left">The data types of input and output are inconsistent. When transposed is set to false, the data types of input and output can be inconsistent, and this error will not be reported.</td></tr>
  <tr><td align="left">The input shape of stride, padding, dilation, or outputPadding is incorrect.</td></tr>
  <tr><td align="left">The groups input is incorrect.</td></tr>
  <tr><td align="left">The shape of output does not meet the inference shape result.</td></tr>
  <tr><td align="left">The value of outputPadding does not meet the requirements.</td></tr>
  <tr><td align="left">Some dimensions of the null tensors passed for input, weight, bias, and output are zero, which does not meet the requirements.</td></tr>
  <tr><td align="left">The spatial size of the input is smaller than that of the weight (after dilation, if dilation > 1 exists) after the padding operation (in non-transpose mode).</td></tr>
  <tr><td align="left">The shape of the bias in transpose mode is not 1.</td></tr>
  <tr><td align="left">The stride or dilation is less than 0.</td></tr>
  <tr><td align="left">The current processor does not support convolution.</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">Internal API verification error. This is usually caused by the fact that the specifications of the input data or attributes are not supported.</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">An exception occurs when the API calls the NPU runtime interface, for example, the SoC version is not supported.</td>
  </tr>
  </table>

## aclnnConvolution

- **Parameters**

  <table>
  <tr>
  <th style="width:120px">Parameter</th>
  <th style="width:80px">Input/Output</th>
  <th>Description</th>
  </tr>
  <tr>
  <td>workspace</td>
  <td>Input</td>
  <td>Address of the workspace to be allocated on the device.</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>Input</td>
  <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnConvolutionGetWorkspaceSize.</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>Input</td>
  <td>Operator executor, containing the operator computation flow.</td>
  </tr>
  <tr>
  <td>stream</td>
  <td>Input</td>
  <td>Stream for executing the task.</td>
  </tr>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Code</a>.

## Constraints

- Deterministic computation:
  - **aclnnConvolution** defaults to a deterministic implementation.

<table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width:150px">
    <col style="width:450px">
    <col style="width:400px">
    <col style="width:400px">
    <col style="width:550px">
    </colgroup>
   <thead>
    <tr>
     <th>Restriction Type</th>
     <th><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></th>
     <th><term>Atlas 200I/500 A2 inference products</term></th>
     <th><term>Atlas inference products</term></th>
     <th>Ascend 950PR/Ascend 950DT</th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">input, weight</th>
     <td>
        <ul><li>The data types of input and weight cannot be HIFLOAT8 or FLOAT8_E4M3FN.</li></ul>
        <ul><li>For transposed 2D and 3D convolutions (transposed=true), the H and W dimensions of weight must be in the range [1, 255], and all other dimensions must be greater than or equal to 1.</li></ul>
        <ul><li>For transposed 1D convolution (transposed=true), the L dimension of weight must be in the range [1, 255], and all other dimensions must be greater than or equal to 1.</li></ul>
        <ul><li>For forward 3D convolution, the H and W dimensions of weight must be in the range [1, 511].</li></ul>
     </td>
     <td>
        The input and weight data types do not support HIFLOAT8 and FLOAT8_E4M3FN.
     </td>
     <td>
        <ul><li>The data types of input and weight cannot be BFLOAT16, HIFLOAT8, or FLOAT8_E4M3FN.</li></ul>
        <ul><li>The height and width of the input must be less than or equal to 4096, and the D, height, and width of the weight must be less than or equal to 255.</li></ul>
     </td>
     <td>
           The supported data types of the input and weight are FLOAT, FLOAT16, BFLOAT16, and HIFLOAT8.
           <ul>
              <li>When transposed is true, the supported data type of the input is FLOAT8_E4M3FN. When the input data type is HIFLOAT8 or FLOAT8_E4M3FN, the data types of the output and weight must be the same as that of the input. The size of the N dimension must be greater than or equal to 0, and the size of other dimensions must be greater than or equal to 1. The supported data type of the weight is FLOAT8_E4M3FN. The size of each dimension must be greater than or equal to 0. (If the D, H, or W dimension is 0, the derived output dimension must also be 0.)</li>
              <li>When transposed is false, the data type of the weight must be the same as that of the input if the input data type is HIFLOAT8. The size of the N dimension must be greater than or equal to 0, the size of the D, H, and W dimensions must be greater than or equal to 0 (the size of these dimensions can be 0 only when the derived D, H, and W dimensions of the output are also 0), and the size of the C dimension must be greater than or equal to 0 (the size of this dimension can be 0 only when the derived N, C, D, H, or W dimension of the output is 0). The height and width of the weight must be in the range of [1, 511]. The size of the N dimension must be greater than or equal to 0 (the size of this dimension can be 0 only when the N dimension of the bias and output is also 0). The size of the C dimension is the same as that of the input C dimension.</li>
            </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">bias</th>
     <td>
        <ul>
          <li>The data type of bias cannot be HIFLOAT8 or FLOAT8_E4M3FN. The data type is the same as that of the input and weight.</li>
          <li>For forward 1D, 2D, and 3D convolutions, bias is converted to FLOAT for computation.</li>
        </ul>
     </td>
     <td>
        <ul>
          <li>The data type of bias cannot be HIFLOAT8 or FLOAT8_E4M3FN. The data type is the same as that of input and weight.</li>
          <li>For forward 1D, 2D, and 3D convolutions, bias is converted to FLOAT for computation.</li>
        </ul>
     </td>
     <td>
        The data type of bias cannot be BFLOAT16, HIFLOAT8, or FLOAT8_E4M3FN. For forward 3D convolution, only FLOAT16 is supported.
     </td>
     <td>
          The data type of bias cannot be HIFLOAT8 or FLOAT8_E4M3FN.
          <ul>
            <li>When transposed is set to true, bias is not supported when the data types of input and weight are HIFLOAT8 and FLOAT8_E4M3FN.</li>
            <li>When transposed is set to false, the bias data type is converted into FLOAT for computation when the data types of input and weight are HIFLOAT8.</li>
          </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">stride</th>
     <td colspan="3">
        For transposed 3D convolution (transposed=true), strideD must be greater than or equal to 1, and strideH and strideW must be in the range [1, 63]. For transposed 1D and 2D convolutions (transposed=true), values must be greater than or equal to 1. For forward 3D convolution, strideH and strideW must be in the range [1, 63].
     </td>
     <td>
        In the conv3d scenario where transposed is set to true, strideD, strideH, and strideW must be greater than or equal to 1. In the conv3d scenario where transposed is set to false, strideH and strideW must be within the range of [1, 63], and strideD must be within the range of [1, 1000000].
     </td>
   </tr>
   <tr>
     <th scope="row">padding</th>
     <td colspan="3">
        For transposed 3D convolution (transposed=true), paddingD must be greater than or equal to 0, and paddingH and paddingW must be in the range [0, 255]. For non-transposed 1D and 2D convolutions (transposed=false), all values must be in the range [0, 255]. For forward 3D convolution, paddingH and paddingW must be in the range [0, 255].
     </td>
     <td>
        In the conv3d transposed=true scenario, paddingD, paddingH, and paddingW must be greater than or equal to 0. In the conv3d transposed=false scenario, paddingH and paddingW must be within the range of [0, 255], and paddingD must be within the range of [0, 1000000].
     </td>
   </tr>
   <tr>
     <th scope="row">dilation</th>
     <td colspan="3">
        For transposed 1D, 2D, and 3D convolutions (transposed=true), all values must be in the range [1, 255]. For forward 3D convolution, dilationH and dilationW must be in the range [1, 255].
     </td>
     <td>
        In the conv3d transposed=true scenario, dilationD, dilationH, and dilationW must be greater than or equal to 1. In the conv3d transposed=false scenario, dilationH and dilationW must be within the range of [1, 255], and dilationD must be within the range of [1, 1000000].
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType</th>
     <td>
        <ul>
          <li>If the value is 0 (KEEP_DTYPE), the input is of type FLOAT, which is not supported currently.</li>
          <li>If the value is 1 (ALLOW_FP32_DOWN_PRECISION), the input can be converted to HFLOAT32 for computation.</li>
          <li>2 (USE_FP16): This option is not supported when the input is BFLOAT16.</li>
          <li>3 (USE_HF32): The input FLOAT is converted to HFLOAT32 for computation.</li>
        </ul>
     </td>
     <td>
        <ul>
          <li>0 (KEEP_DTYPE): This option is not supported for FLOAT inputs.</li>
          <li>1 (ALLOW_FP32_DOWN_PRECISION): The FLOAT input is allowed to be converted to HFLOAT32 for computation.</li>
          <li>2 (USE_FP16): This option is not supported for BFLOAT16 inputs.</li>
          <li>3 (USE_HF32): The FLOAT input is converted to HFLOAT32 for computation.</li>
        </ul>
     </td>
     <td>
        <ul>
          <li>0 (KEEP_DTYPE): This option is not supported for FLOAT inputs.</li>
          <li>1 (ALLOW_FP32_DOWN_PRECISION): The FLOAT input is allowed to be converted to FLOAT16 for computation.</li>
          <li>2 (USE_FP16): This option is not supported for BFLOAT16 inputs.</li>
          <li>3 (USE_HF32): This option is not supported currently.</li>
      </ul>
     </td>
     <td>
        <ul>
          <li>1 (ALLOW_FP32_DOWN_PRECISION): The input FLOAT can be converted to HFLOAT32 for computation.</li>
          <li>2 (USE_FP16): This option is not supported when the input is BFLOAT16.</li>
          <li>3 (USE_HF32): The input FLOAT is converted to HFLOAT32 for computation.</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">Other constraints</th>
     <td>
        For tensors in input, weight, and bias, the size of each dimension must not exceed 1,000,000.
     </td>
     <td>
        <ul>
          <li>Currently, only 2D convolution is supported, and the value of transposed must be false. 1D and 3D convolutions are not supported.</li>
          <li>The size of each dimension of each group of tensors in input, weight, and bias must be less than or equal to 1000000.</li>
        </ul>
     </td>
     <td>
        <ul>
          <li>1D and 2D convolutions are supported. In the forward scenario of 3D convolution, only FLOAT16 is supported.</li>
          <li>The size of each dimension of each group of tensors in input, weight, and bias must be less than or equal to 1000000.</li>
        </ul>
     </td>
     <td>
        When transposed is set to true, 1D, 2D, and 3D convolutions are supported, and empty tensors are supported. In this case, the gradient calculation behavior in the padding area depends on the input shape. Depending on the operator optimization policy, the gradient in the padding area may be directly set to 0.
     </td>
   </tr>
   </tbody>
  </table>

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                    \
    if (!(cond)) {                        \
      Finalize(deviceId, stream);         \
      return_expr;                        \
    }                                     \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i: shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Boilerplate) Initialize resources.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnConvolutionTest(int32_t deviceId, aclrtStream& stream)
{
  auto ret = Init(deviceId, &stream);
  // Customize error handling based on your requirements.
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> shapeInput = {2, 2, 2, 2};
  std::vector<int64_t> shapeWeight = {1, 2, 1, 1};
  std::vector<int64_t> shapeResult = {2, 1, 2, 2};
  std::vector<int64_t> convStrides;
  std::vector<int64_t> convPads;
  std::vector<int64_t> convOutPads;
  std::vector<int64_t> convDilations;

  void* deviceDataA = nullptr;
  void* deviceDataB = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* result = nullptr;
  std::vector<float> inputData(GetShapeSize(shapeInput), 1);
  std::vector<float> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> outputData(GetShapeSize(shapeResult), 1);
  convStrides = {1, 1, 1, 1};
  convPads = {0, 0, 0, 0};
  convOutPads = {0, 0, 0, 0};
  convDilations = {1, 1, 1, 1};

  // Create an input aclTensor.
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, aclDataType::ACL_FLOAT, &input);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a weight aclTensor.
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, aclDataType::ACL_FLOAT, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult, aclDataType::ACL_FLOAT, &result);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
  CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outPadsPtr(outPads, aclDestroyIntArray);
  CHECK_FREE_RET(outPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
  CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnConvolution.
  ret = aclnnConvolutionGetWorkspaceSize(input, weight, nullptr, strides, pads, dilations, false, outPads, 1, result, 1,
                                         &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnConvolution.
  ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolution failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(shapeResult);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnConvolutionTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
