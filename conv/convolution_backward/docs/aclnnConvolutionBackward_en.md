# aclnnConvolutionBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T16:47:01.376Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Backward propagation of convolution. Computes the gradients of the input, weight, and bias based on the output mask settings. This function supports 1D, 2D, and 3D convolution.

- Formula:

  The relationship between the input ($N,C_{in},D_{in},H_{in},W_{in}$), output ($N,C_{out},D_{out},H_{out},W_{out}$), convolution stride ($stride$), convolution kernel size ($kernelSize$, $kD$, $kH$, $kW$), and dilation parameter ($dilation$) is:

  $$
    D_{out}=\lfloor \frac{D_{in}+2*padding[0]-dilation[0] * (kernelSize[0] - 1) - 1}{stride[0]}+1 \rfloor
  $$

  $$
    H_{out}=\lfloor \frac{H_{in}+2*padding[1]-dilation[1] * (kernelSize[1] - 1) - 1}{stride[1]}+1 \rfloor
  $$

  $$
    W_{out}=\lfloor \frac{W_{in}+2*padding[2]-dilation[2] * (kernelSize[2] -1) -1}{stride[2]}+1 \rfloor
  $$

  Convolution backward propagation computes the gradients of the input tensor $x$ (input in the function prototype), kernel weight tensor $w$ (weight in the function prototype), and bias $b$ for the convolution forward operation.

  - For the gradient of $x$, $\frac{\partial L}{\partial x}$ (corresponding to the gradInput parameter in the function prototype):

    $$
    \frac{\partial L}{\partial x_{n, c_{in}, i, j}} = \sum_{c_{out}=1}^{C_{out}} \sum_{p=1}^{k_H} \sum_{q=1}^{k_W} \frac{\partial L}{\partial y_{n, c_{out}, i-p, j-q}}\cdot w_{c_{out}, c_{in}, p, q}
    $$

    Where $L$ is the loss function, and $\frac{\partial L}{\partial y}$ is the gradient of the output tensor $y$ with respect to $L$ (corresponding to the gradOutput parameter in the function prototype).

  - For the gradient of $w$, $\frac{\partial L}{\partial w}$ (corresponding to the gradWeight parameter in the function prototype):

    $$
    \frac{\partial L}{\partial w_{c_{out}, c_{in}, p, q}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, i \cdot s_H + p, j \cdot s_W + q} \cdot \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$

  - For the gradient of $b$, $\frac{\partial L}{\partial b}$ (corresponding to the gradBias parameter in the function prototype):

    $$
    \frac{\partial L}{\partial b_{c_{out}}} = \sum_{n=1}^{N}       \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnConvolutionBackwardGetWorkspaceSize" to obtain the workspace size required for computation and the executor containing the operator computation process, and then call "aclnnConvolutionBackward" to perform the computation.

```cpp
aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(
    const aclTensor     *gradOutput,
    const aclTensor     *input,
    const aclTensor     *weight,
    const aclIntArray   *biasSizes,
    const aclIntArray   *stride,
    const aclIntArray   *padding,
    const aclIntArray   *dilation,
    bool                 transposed,
    const aclIntArray   *outputPadding,
    int                  groups,
    const aclBoolArray  *outputMask,
    int8_t               cubeMathType,
    aclTensor           *gradInput,
    aclTensor           *gradWeight,
    aclTensor           *gradBias,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```cpp
aclnnStatus aclnnConvolutionBackward(
    void                *workspace,
    uint64_t             workspaceSize,
    aclOpExecutor       *executor,
    const aclrtStream    stream)
```

## aclnnConvolutionBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width:170px">
    <col style="width:120px">
    <col style="width:300px">
    <col style="width:330px">
    <col style="width:212px">
    <col style="width:100px">
    <col style="width:190px">
    <col style="width:145px">
    </colgroup>
    <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient of output tensor y with respect to L.</td>
      <td>       <ul><li>Supports empty tensors.</li>
       <li>The data type must follow the data type deduction rules with input and weight (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a> and <a href="#constraints" target="_blank">Constraints</a>).</li>
       <li>The shape does not support broadcast. It must follow the deduction relationship of the convolution Input/Output shape with input and weight.</li>
       <li>The data format must be consistent with input and gradInput.</li></ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCL, NCHW, NCDHW</td>
      <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>x in the formula.</td>
      <td>       <ul><li>Supports empty tensors.</li>
       <li>The data type must comply with the data type inference rules with gradOutput and weight (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a> and <a href="#constraints" target="_blank">Constraints</a>).</li>
       <li>The shape does not support broadcast. The shape must satisfy the derivation relationship of the convolution input/output shapes with gradOutput and weight.</li>
       <li>The data format must be the same as that of gradOutput and gradInput.</li></ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCL, NCHW, NCDHW</td>
      <td>See <a href="#constraints" target="_blank">Constraints</a>.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>w in the formula.</td>
      <td>       <ul><li>Supports empty tensors.</li>
       <li>The data type must comply with the data type inference rules with gradOutput and input (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a> and <a href="#constraints" target="_blank">Constraints</a>).</li>
       <li>The shape does not support broadcast. The shape must satisfy the derivation relationship of the convolution input/output shapes with gradOutput and input.</li>
       <li>The data format must be the same as that of gradWeight.</li></ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCL, NCHW, NCDHW</td>
      <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasSizes</td>
      <td>Input</td>
      <td>Shape of the bias in forward convolution.</td>
      <td>       <ul><li>The array length is 1.</li>
       <li>For regular convolution, the value equals [weight.shape[0]]. For transposed convolution, the value equals [weight.shape[1] * groups].</li>
       <li>In the empty tensor use case, if outputMask specifies that the bias gradient needs to be computed, biasSizes cannot be nullptr.</li></ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Stride of the convolution kernel moving across the input during backward propagation.</td>
      <td>       <ul><li>For 1D convolution backward, the array length must be 1.</li>
       <li>The array length equals the weight dimension minus 2, and each value must be greater than 0.</li></ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
      <td>×</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Padding for the input during backward propagation.</td>
      <td>       <ul><li>For 1D convolution backward, the array length must be 1.</li>
       <li>The array length can be the weight dimension minus 2. In 2D use cases, the array length can be 4.</li>
       <li>The value must be greater than or equal to 0.</li></ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>See <a href="#Constraints" target="_blank">Constraints.</a></td>
      <td>×</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Input</td>
      <td>Dilation parameter during backward propagation.</td>
      <td>       <ul><li>For 1D Convolution Backward, the array length must be 1.</li>
       <li>The array length can be the weight dimension minus 2.</li>
       <li>The value must be greater than 0.</li></ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
      <td>×</td>
    </tr>
    <tr>
      <td>transposed</td>
      <td>Input</td>
      <td>Flag that indicates whether transposed convolution is enabled. Transposed convolution is enabled when the value is True.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>outputPadding</td>
      <td>Input</td>
      <td>Output padding during backward propagation.</td>
      <td>       <ul><li>The array length can be the weight dimension minus 2, and the value of each dimension must be in the range [0, max(the corresponding value of stride, the corresponding value of dilation)).</li>
       <li>If transposed is False, each element value must be 0.</li></ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>Input</td>
      <td>Number of groups that input channels are divided into during backward propagation.</td>
      <td>       The C dimension of groups*weight must be equal to the C dimension of input, and the value of groups ranges from [1, 65535].
      </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>outputMask</td>
      <td>Input</td>
      <td>       <ul><li>Output mask parameter. Specifies whether the output contains the gradients of the input, weight, and bias.</li>
       <li>During backward propagation, the gradients at the positions where the mask parameter is True are output.</li></ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Determines which calculation logic the Cube unit uses for computation.</td>
      <td>       <ul><li>If the input data types have a <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">deduction relationship</a>, this parameter applies to the deduced data type by default.</li>
       <li>The supported enumeration values are as follows:</li>
       <ul><li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
       <li>1: ALLOW_FP32_DOWN_PRECISION, allows computing the input data at reduced precision.</li>
       <li>2: USE_FP16, allows converting to the FLOAT16 data type for computation. If the input data type is FLOAT, it is converted to FLOAT16 for computation.</li>
       <li>3: USE_HF32, allows converting to the HFLOAT32 data type for computation. If the input is FLOAT16, FLOAT16 is still used for computation.</li></ul></ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Gradient of input tensor x with respect to L.</td>
      <td>       <ul><li>Supports empty Tensor.</li>
       <li>The data type is the same as that of input.</li>
       <li>The data format must be the same as those of input and gradOutput.</li></ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCL, NCHW, NCDHW</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradWeight</td>
      <td>Output</td>
      <td>Gradient of the convolution kernel weight tensor w with respect to L.</td>
      <td>       <ul><li>Supports empty Tensor.</li>
       <li>The data format must be the same as that of weight.</li></ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCL, NCHW, NCDHW</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradBias</td>
      <td>Output</td>
      <td>Gradient of bias b with respect to L.</td>
      <td>       <ul><li>Supports empty tensors.</li>
       <li>The data type is the same as that of gradOutput.</li></ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
   </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:1050px">
    </colgroup>
   <thead>
    <tr>
     <th>Return Value</th>
     <th>Error Code</th>
     <th>Description</th>
   </tr>
   </thead>
   <tbody>

   <tr>
     <td>ACLNN_ERR_PARAM_NULLPTR</td>
     <td>161001</td>
     <td>The passed-in gradOutput, input, weight, biasSizes, stride, padding, dilation, outputPadding, or outputMask is a null pointer, or the pointer for computing gradInput, gradWeight, or gradBias is a null pointer.</td>
   </tr>
   <tr>
   </tr>
   <tr>
     <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
     <td rowspan="7">161002</td>
   </tr>
   <tr>
     <td>The data types of gradOutput, input, and weight are not within the supported range.</td>
   </tr>
   <tr>
     <td>The data formats of gradOutput, input, and weight are not within the supported range.</td>
   </tr>
   <tr>
     <td>The shapes of gradOutput, input, and weight do not meet the constraints.</td>
   </tr>
   <tr>
     <td>The shapes of biasSizes, stride, padding, dilation, and outputPadding do not meet the constraints.</td>
   </tr>
   <tr>
     <td>The condition groups * C dimension of weight = C dimension of input is not met.</td>
   </tr>
   <tr>
     <td>The current processor does not support convolution backward propagation.</td>
   </tr>
   <tr>
     <td>ACLNN_ERR_INNER_NULLPTR</td>
     <td>561103</td>
     <td>Internal API verification error, usually caused by unsupported specifications of the input data or attributes.</td>
   </tr>
   </tbody>

   </table>

## aclnnConvolutionBackward

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
       <col style="width:100px">
       <col style="width:100px">
       <col style="width:950px">
       </colgroup>
    <thead>
       <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
    </thead>
    <tbody>
       <tr>
       <td>workspace</td>
       <td>Input</td>
       <td>The workspace memory address allocated on the Device side.</td>
       </tr>
       <tr>
       <td>workspaceSize</td>
       <td>Input</td>
       <td>The workspace size allocated on the Device side, obtained by the first-phase API aclnnConvolutionBackwardGetWorkspaceSize.</td>
       </tr>
       <tr>
       <td>executor</td>
       <td>Input</td>
       <td>Op executor that contains the operator computation flow.</td>
       </tr>
       <tr>
       <td>stream</td>
       <td>Input</td>
       <td>Specifies the stream for executing tasks.</td>
       </tr>
    </tbody>
  </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation

  <!-- npu="A3,910b,910,310b" id7 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas training products</term>: aclnnConvolutionBackward defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnConvolutionBackward defaults to a deterministic implementation.
  <!-- end id8 -->

<table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
  <col style="width: 150px">
  <col style="width: 430px">
  <col style="width: 430px">
  <col style="width: 390px">
  </colgroup>
   <thead>
    <tr>
     <th>Constraint Type</th>

     <th><term>Ascend 950PR&950DT products</term></th>


     <th><term>Atlas A2 products</term>, <term>Atlas A3 products</term></th>


     <th><term>Atlas inference products</term>, <term>Atlas training products</term></th>

   </tr>
   </thead>
   <tbody>

   <tr>
     <th scope="row">Empty Tensor Constraints</th>
     <td colspan="2">        <ul><li>The dimensions of input must be 2 or greater. The dimensions of stride, padding, dilation, and outputPadding must be equal to the dimensions of input minus 2. padding can be 4D in the 2D case.</li>
        <li>When transposed=false, the constraints on the D, H, and W dimensions comply with Formula 1 (see the description below the table).</li>
        <li>When transposed=true:</li>
          <ul><li>The N axis of weight must be greater than 0.</li>
          <li>The N axis of weight must be equal to the C axis of input.</li></ul>
        </ul>
     </td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">gradOutput Constraints</th>
     <td>-</td>
     <td>        For the 1d, 2d, and 3d transposed=false use cases, the size of each dimension must be greater than or equal to 1. When input is an empty tensor, the N, C, D, H, and W dimensions can be 0.
     </td>
     <td>        Does not support empty tensors.
     </td>
   </tr>
   <tr>
     <th scope="row">input constraints</th>
     <td>        <ul><li>transposed=true: Supports empty tensors with the N dimension of 0. When the N dimension is 0 and the empty tensor constraints are met, the DHW dimensions can also be 0.</li>
        <li>transposed=false: Supports empty tensors with the N or C dimension of 0 (when the C dimension is 0, the C dimension of weight must also be 0). When the N or C dimension is 0 and the empty tensor constraints are met, the DHW dimensions can also be 0.</li></ul>
     </td>
     <td>        For 1d, 2d, and 3d transposed=false use cases, the size of each dimension must be greater than or equal to 1.
          <ul><li>transposed=true: Supports empty tensors with the N dimension of 0. When the N dimension is 0 and the empty tensor constraints are met, the DHW dimensions can also be 0.</li>
          <li>transposed=false: Supports empty tensors with the N or C dimension of 0 (when the C dimension is 0, the C dimension of weight must also be 0). When the N or C dimension is 0 and the empty tensor constraints are met, the DHW dimensions can also be 0.</li></ul>
     </td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">weight constraints</th>
     <td>        <ul><li>transposed=true: The sizes of H and W must be in the range of [1,255], and the sizes of other dimensions must be greater than or equal to 1. When input is an empty tensor and the empty tensor constraints are met, the C, D, H, and W axes can be 0.</li>
        <li>transposed=false: The C dimension can be 0 (when C is 0, the C dimension of input must also be 0), and the sizes of all dimensions must be greater than or equal to 1. When input is an empty tensor and the empty tensor constraints are met, the D, H, and W axes can be 0.</li></ul>
     </td>
     <td>        For 2d and 3d transposed=false use cases, the sizes of H and W must be in the range of [1,255], and the sizes of other dimensions must be greater than or equal to 1. For the 1d transposed=false use case, the size of L must be in the range of [1,255], and the sizes of other dimensions must be greater than or equal to 1.
           <ul><li>transposed=true: When input is an empty tensor and the empty tensor constraints are met, the C, D, H, and W axes can be 0.</li>
           <li>transposed=false: The C dimension can be 0 (when C is 0, the C dimension of input must also be 0). When input is an empty tensor and the empty tensor constraints are met, the D, H, and W axes can be 0.</li></ul>
     </td>
     <td>        Does not support empty tensors.
     </td>
   </tr>
   <tr>
     <th scope="row">stride constraints</th>
     <td>For 1d, 2d, and 3d transposed=false use cases, each value must be greater than or equal to 1. For 1d, 2d, and 3d transposed=true use cases, strideD must be in the range of [1,1000000], and strideH and strideW must be in the range of [1,2147483646].</td>
     <td>For 3d transposed=false use cases, strideD must be greater than or equal to 1, and strideH and strideW must be in the range of [1,63]. For 1d and 2d transposed=false use cases, each value must be greater than or equal to 1.</td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">padding constraints</th>
     <td>For 1d, 2d, and 3d transposed=false use cases, each value must be greater than or equal to 0. For 1d, 2d, and 3d transposed=true use cases, paddingD must be in the range of [0,1000000], and paddingH and paddingW must be in the range of [0,2147483646].</td>
     <td>For 3d transposed=false use cases, paddingD must be greater than or equal to 0, and paddingH and paddingW must be in the range of [0,255]. For 1d and 2d transposed=false use cases, each value must be in the range of [0,255].</td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">dilation constraints</th>
     <td>For 1d, 2d, and 3d transposed=false use cases, each value must be greater than or equal to 1. For 1d, 2d, and 3d transposed=true use cases, dilationD must be in the range of [1,1000000], and dilationH and dilationW must be in the range of [1,2147483646].</td>
     <td>For 1d, 2d, and 3d transposed=false use cases, each value must be in the range of [1,255].</td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">dtype constraint</th>
     <td>        The data types support HIFLOAT8 and FLOAT8_E4M3FN only when transposed=true and output_mask[0]=true.
     </td>
     <td>        HIFLOAT8 and FLOAT8_E4M3FN are not supported.
     </td>
     <td>        BFLOAT16, HIFLOAT8, and FLOAT8_E4M3FN are not supported.
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType description</th>
     <td colspan="2">        <ul><li>Enumeration value 0: no constraints.</li>
        <li>Enumeration value 1: When the input is FLOAT, it is converted to HFLOAT32 for computation. No processing is performed for other data types.</li>
        <li>Enumeration value 2: This option is not supported when the input is BFLOAT16. No processing is performed for other data types.</li>
        <li>Enumeration value 3: When the input is FLOAT, it is converted to HFLOAT32 for computation. No processing is performed for other data types.</li>
        </ul>
     </td>
     <td>        <ul><li>Enumeration value 0: When the input is FLOAT, the Cube computing unit does not support it yet. An error is reported when the value is 0.</li>
        <li>Enumeration value 1: When the input is FLOAT, it is converted to FLOAT16 for computation. No processing is performed for other data types.</li>
        <li>Enumeration value 2: no constraints.</li>
        <li>Enumeration value 3: When the input is FLOAT, the Cube computing unit does not support it yet.</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">Other constraints</th>
     <td>        The computation behavior of the gradients in the padding area depends on the input shape. Depending on the operator optimization strategy, the gradients in the padding area may be directly set to 0.
     </td>
     <td>-</td>
     <td>        Currently, only backward propagation of 1D and 2D convolutions is supported. Backward propagation of 3D convolutions is not supported.
     </td>
   </tr>
   </tbody>
  </table>

  - Formula 1:
    $$
        (input_{dim} + pad_{dim} \times 2) \ge ((weight_{dim}  - 1) \times dilation_{dim} + 1)
    $$

Due to hardware resource limits, the operator may fail for some parameter value combinations. Analyze and troubleshoot the problem based on the log information. If the problem cannot be resolved, visit [here](https://www.hiascend.com/support) to obtain technical support.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_convolution_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)        \
    do {                                         \
        if (!(cond)) {                           \
            Finalize(deviceId, stream); \
            return_expr;                         \
        }                                        \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // Boilerplate for resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **DeviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for device memory.
    auto ret = aclrtMalloc(DeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to the device memory.
    ret = aclrtMemcpy(*DeviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    if (shape.size() == 4) {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                  shape.data(), shape.size(), *DeviceAddr);
    } else {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *DeviceAddr);
    }

    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnConvolutionBackwardTest(int32_t deviceId, aclrtStream &stream)
{
    // 1. Initialize.
    auto ret = Init(deviceId, &stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs. Build them as required by the API.
    std::vector<int64_t> gradOutputShape = {2, 2, 7, 7};
    std::vector<int64_t> inputShape = {2, 2, 7, 7};
    std::vector<int64_t> weightShape = {2, 2, 1, 1};
    std::vector<int64_t> biasSize = {2};
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    bool transposed = false;
    std::vector<int64_t> outputPadding = {0, 0};
    int groups = 1;
    bool outputMask[3] = {true, true, true};
    int8_t cubeMathType = 1;

    std::vector<int64_t> gradInputShape = {2, 2, 7, 7};
    std::vector<int64_t> gradWeightShape = {2, 2, 1, 1};
    std::vector<int64_t> gradBiasShape = {2};

    // Create the gradOutput aclTensor.
    std::vector<float> gradOutputData(GetShapeSize(gradOutputShape), 1);
    aclTensor *gradOutput = nullptr;
    void *gradOutputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradOutputData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradOutputTensorPtr(gradOutput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradOutputDeviceAddrPtr(gradOutputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the input aclTensor.
    std::vector<float> inputData(GetShapeSize(inputShape), 1);
    aclTensor *input = nullptr;
    void *inputDeviceAddr = nullptr;
    ret = CreateAclTensor(inputData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> inputDeviceAddrPtr(inputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the weight aclTensor.
    std::vector<float> weightData(GetShapeSize(weightShape), 1);
    aclTensor *weight = nullptr;
    void *weightDeviceAddr = nullptr;
    ret = CreateAclTensor(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the gradInput aclTensor.
    std::vector<float> gradInputData(GetShapeSize(inputShape), 1);
    aclTensor *gradInput = nullptr;
    void *gradInputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradInputData, inputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradInputTensorPtr(gradInput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradInputDeviceAddrPtr(gradInputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the gradWeight aclTensor.
    std::vector<float> gradWeightData(GetShapeSize(weightShape), 1);
    aclTensor *gradWeight = nullptr;
    void *gradWeightDeviceAddr = nullptr;
    ret = CreateAclTensor(gradWeightData, weightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradWeightTensorPtr(gradWeight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradWeightDeviceAddrPtr(gradWeightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the gradBias aclTensor.
    std::vector<float> gradBiasData(GetShapeSize(biasSize), 1);
    aclTensor *gradBias = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    ret = CreateAclTensor(gradBiasData, biasSize, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradBiasTensorPtr(gradBias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradBiasDeviceAddrPtr(gradBiasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create the biasSizes aclIntArray.
    aclIntArray *biasSizes = aclCreateIntArray(biasSize.data(), 1);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> biasSizesPtr(biasSizes, aclDestroyIntArray);
    CHECK_FREE_RET(biasSizes != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // Create the strides aclIntArray.
    aclIntArray *strides = aclCreateIntArray(stride.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
    CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // Create the pads aclIntArray.
    aclIntArray *pads = aclCreateIntArray(padding.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
    CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // Create the dilations aclIntArray.
    aclIntArray *dilations = aclCreateIntArray(dilation.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
    CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // Create the outputPads aclIntArray.
    aclIntArray *outputPads = aclCreateIntArray(outputPadding.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outputPadsPtr(outputPads, aclDestroyIntArray);
    CHECK_FREE_RET(outputPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // Create the outMask aclBoolArray.
    aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);
    std::unique_ptr<aclBoolArray, aclnnStatus (*)(const aclBoolArray *)> outMaskPtr(outMask, aclDestroyBoolArray);
    CHECK_FREE_RET(outMask != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API aclnnConvolutionBackwardGetWorkspaceSize.
    ret = aclnnConvolutionBackwardGetWorkspaceSize(gradOutput, input, weight, biasSizes, strides, pads, dilations,
                                                   transposed, outputPads, groups, outMask, cubeMathType, gradInput,
                                                   gradWeight, gradBias, &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
                   return ret);
    // Request device memory based on the workspaceSize calculated by the first-phase API.
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API aclnnConvolutionBackward.
    ret = aclnnConvolutionBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackward failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value, copy the result from device memory to host memory, and modify it based on the specific API definition
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> gradInputResult(size, 0);
    ret = aclrtMemcpy(gradInputResult.data(), gradInputResult.size() * sizeof(gradInputResult[0]), gradInputDeviceAddr,
                      size * sizeof(gradInputResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradInputResult[%ld] is: %f\n", i, gradInputResult[i]);
    }

    size = GetShapeSize(gradWeightShape);
    std::vector<float> gradWeightResult(size, 0);
    ret = aclrtMemcpy(gradWeightResult.data(), gradWeightResult.size() * sizeof(gradWeightResult[0]), gradWeightDeviceAddr,
                      size * sizeof(gradWeightResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradWeightResult[%ld] is: %f\n", i, gradWeightResult[i]);
    }

    size = GetShapeSize(gradBiasShape);
    std::vector<float> gradBiasResult(size, 0);
    ret = aclrtMemcpy(gradBiasResult.data(), gradBiasResult.size() * sizeof(gradBiasResult[0]), gradBiasDeviceAddr,
                      size * sizeof(gradBiasResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradBiasResult[%ld] is: %f\n", i, gradBiasResult[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. (boilerplate) device/stream initialization
    // Enter the deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnConvolutionBackwardTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackwardTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
