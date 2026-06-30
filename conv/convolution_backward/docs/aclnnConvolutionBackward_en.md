# aclnnConvolutionBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/conv/convolution_backward)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Computes the backward pass of a convolution operation. Based on the output mask settings, it selectively computes gradients for the input, weight, and bias. This function supports 1D, 2D, and 3D convolutions.  

- Formula:

  The relationship between the input tensor (**input($N,C_{in},D_{in},H_{in},W_{in}$)**), output tensor (**out($N,C_{out},D_{out},H_{out},W_{out}$)**), stride (**$stride$**), kernel size (**$kernelSize,kD,kH,kW$**), and dilation (**$dilation$**) is:

  $$
    D_{out}=\lfloor \frac{D_{in}+2*padding[0]-dilation[0] * (kernelSize[0] - 1) - 1}{stride[0]}+1 \rfloor
  $$

  $$
    H_{out}=\lfloor \frac{H_{in}+2*padding[1]-dilation[1] * (kernelSize[1] - 1) - 1}{stride[1]}+1 \rfloor
  $$

  $$
    W_{out}=\lfloor \frac{W_{in}+2*padding[2]-dilation[2] * (kernelSize[2] -1) -1}{stride[2]}+1 \rfloor
  $$
  
  The backward pass computes gradients with respect to the forward-pass tensors: the input tensor $x$ (corresponding to **input** in the function prototype), the convolution kernel weights $w$ (corresponding to **weight** in the function prototype), and the bias $b$. 
  - Gradient with respect to $x$, $\frac{\partial L}{\partial x}$ (corresponding to the **gradInput** parameter in the function prototype):
  
    $$
    \frac{\partial L}{\partial x_{n, c_{in}, i, j}} = \sum_{c_{out}=1}^{C_{out}} \sum_{p=1}^{k_H} \sum_{q=1}^{k_W} \frac{\partial L}{\partial y_{n, c_{out}, i-p, j-q}}\cdot w_{c_{out}, c_{in}, p, q}
    $$
  
    Where $L$ is the loss function and $\frac{\partial L}{\partial y}$ is the gradient of the loss $L$ with respect to the output tensor $y$ (corresponding to the **gradOutput** parameter in the function prototype). 
  
  - Gradient with respect to $w$, $\frac{\partial L}{\partial w}$ (corresponding to the **gradWeight** parameter in the function prototype):
  
    $$
    \frac{\partial L}{\partial w_{c_{out}, c_{in}, p, q}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, i \cdot s_H + p, j \cdot s_W + q} \cdot \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$
  
  - Gradient with respect to $b$, $\frac{\partial L}{\partial b}$ (corresponding to the **gradBias** parameter in the function prototype):
  
    $$
    \frac{\partial L}{\partial b_{c_{out}}} = \sum_{n=1}^{N}       \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnConvolutionBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnConvolutionBackward** is called to perform computation.

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

- **Parameters**

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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient of the loss L with respect to the output tensor y.</a></td>
      <td>  
       <ul><li>Empty tensors are supported.</li>
       <li>The data type must be mutually inferable with those of input and weight (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a> and <a href="#constraints" target="_blank">Constraints</a>).</li>
       <li>The shape does not support broadcasting; must satisfy convolution shape rules.</li>
       <li>The data format must be the same as those of input and gradInput.</li>
       </ul>
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
      <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The data type must be mutually inferable with those of gradOutput and weight (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a> and <a href="#constraints" target="_blank">Constraints</a>).</li>
       <li>The shape does not support broadcasting; must satisfy convolution shape rules.</li>
       <li>The data format must be the same as those of gradOutput and gradInput.</li>
       </ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCL, NCHW, NCDHW</td>
      <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>w in the formula.</td>
      <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The data type must be mutually inferable with with those of gradOutput and input (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a> and <a href="#constraints" target="_blank">Constraints</a>).</li>
       <li>The shape does not support broadcasting; must satisfy convolution shape rules.</li>
       <li>The data format must be the same as that of gradWeight.</li>
       </ul>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCL, NCHW, NCDHW</td>
      <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasSizes</td>
      <td>Input</td>
      <td>Shape of the bias tensor used in forward convolution.</td>
      <td>
       <ul><li>The array length is 1.</li>
       <li>It is equivalent to [weight.shape[0]] in a regular convolution and equivalent to [weight.shape[1] * groups] in a transposed convolution.</li>
       <li>In the empty tensor scenario, biasSizes must not be nullptr when outputMask specifies that the gradient of the bias requires computation.</li>
       </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Stride of the convolution kernel on the input during backward pass.</td>
      <td>
       <ul><li>For 1D convolution backward pass, the array length must be 1.</li>
       <li>The array length must equal the number of dimensions of weight minus 2, and each element in the array must be greater than 0.</li>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
      <td>×</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Input padding during backward pass.</td>
      <td>
       <ul><li>For 1D convolution backward pass, the array length must be 1.</li>
       <li>The array length can be the number of dimensions of weight minus 2. In 2D scenarios, the array length can be 4.</li>
       <li>Each element in the array must be greater than or equal to 0.</li>
       </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
      <td>×</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Input</td>
      <td>Dilation during backward pass.</td>
      <td>
       <ul><li>For 1D convolution backward pass, the array length must be 1.</li>
       <li>The array length can be the number of dimensions of weight minus 2.</li>
       <li>Each element in the array must be greater than 0.</li>
       </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
      <td>×</td>
    </tr>
    <tr>
      <td>transposed</td>
      <td>Input</td>
      <td>Transposed convolution enable flag (True: enabled).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>outputPadding</td>
      <td>Input</td>
      <td>Output padding during backward pass.</td>
      <td>
       <ul><li>The array length can be the number of dimensions of weight minus 2. Each element in the array must be in the range [0, stride value at the corresponding dimension).</li>
       <li>When transposed is set to False, the value of each element must be 0.</li>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>Input</td>
      <td>Number of groups in the input channels during backward pass.</td>
      <td>
       The product of groups and the C dimension of weight must equal to the C dimension of input. groups must be in the range [1, 65535].
      </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>outputMask</td>
      <td>Input</td>
      <td>
       <ul><li>Output mask, which specifies whether the output contains the gradients of the input, weight, and bias.</li>
       <li>During backward pass, only gradients at positions marked True in the mask are returned.</li>
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
      <td>Computation logic to be used by the Cube unit.</td>
      <td>
       <ul><li>If the input data types have a <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">mutual inference relationship</a>, this parameter is applied to the inferred data type by default.</li>
       <li>Supported enumerations:</li>
       <ul><li>0: KEEP_DTYPE. The input data type is retained for computation.</li>
       <li>1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with a reduced precision.</li>
       <li>2: USE_FP16. The input data type can be converted to FLOAT16 for computation. When the input data type is FLOAT, it is converted to FLOAT16 for computation.</li>
       <li>3: USE_HF32. The input data type can be converted to HFLOAT32 for computation. When the input data type is FLOAT16, FLOAT16 is still used for computation.</li>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Gradient of the loss L with respect to the input tensor x.</td>
      <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The data type must be the same as that of input.</li>
       <li>The data format must be the same as those of input and gradOutput.</li>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCL, NCHW, NCDHW</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradWeight</td>
      <td>Output</td>
      <td>Gradient of the loss L with respect to the convolution kernel weight tensor w.</td>
      <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The data format must be the same as that of weight.</li>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCL, NCHW, NCDHW</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradBias</td>
      <td>Output</td>
      <td>Gradient of the loss L with respect to the bias b.</td>
      <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The data type must be the same as that of gradOutput.</li>
      </td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
   </tbody>
  </table>  

- **Returns**

    **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md). 

    The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 123px">
    <col style="width: 744px">
    </colgroup>
   <thead>
    <tr>
     <th>Return</th>
     <th>Error Code</th>
     <th>Description</th>
   </tr>
   </thead>
   <tbody>

   <tr>
     <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
     <td rowspan="2">161001</td>
     <td>The passed gradOutput, input, weight, biasSizes, stride, padding, dilation, outputPadding, outputMask, gradInput, or gradWeight is a null pointer.</td>
   </tr>
   <tr>
     <td>The input gradBias is a null pointer when the output includes the bias gradient.</td>
   </tr>
   <tr>
     <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
     <td rowspan="7">161002</td>
   </tr>
   <tr>
     <td>The data type of gradOutput, input, or weight is not supported.</td>
   </tr>
   <tr>
     <td>The data format of gradOutput, input, or weight is not supported.</td>
   </tr>
   <tr>
     <td>The shape of gradOutput, input, or weight does not meet the constraints.</td>
   </tr>
   <tr>
     <td>The shape of biasSizes, stride, padding, dilation, or outputPadding does not meet the constraints.</td>
   </tr>
   <tr>
     <td>The product of groups and the C dimension of weight is not equal to the C dimension of input.</td>
   </tr>
   <tr>
     <td>Convolution backward is not supported by the current processor.</td>
   </tr>
   <tr>
     <td>ACLNN_ERR_INNER_NULLPTR</td>
     <td>561103</td>
     <td>Internal API verification error, usually caused by unsupported input data or attribute specifications.</td>
   </tr>
   </tbody>
   </table>


## aclnnConvolutionBackward

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 123px">
  <col style="width: 855px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnConvolutionBackwardGetWorkspaceSize.</td>
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
  </tbody>
  </table>


- **Returns**

    **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
  
- Deterministic computation:
  - **aclnnConvolutionBackward** is non-deterministic by default. Deterministic mode can be enabled via **aclrtCtxSetSysParamOpt**.

  <table style="undefined;table-layout: fixed; width: 1396px"><colgroup>
    <col style="width: 170px">
    <col style="width: 700px">
    <col style="width: 395px">
    </colgroup>
   <thead>
    <tr>
     <th>Constraint Type</th>
     <th><term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term></th>
     <th><term>Atlas inference series products</term> and <term>Atlas training series products</term></th>
   </tr>
   </thead>
   <tbody>

   <tr>
     <th scope="row">Empty tensor constraints</th>
     <td>
        <ul><li>input must have at least 2 dimensions. The dimensions of stride, padding, dilation, and outputPadding must equal to the dimensions of input minus 2 (for 2D scenarios, padding can be 4-dimensional).</li>
        <li>When transposed is set to false, the constraints on the D, H, and W dimensions must satisfy formula 1 (see the notes below the table).</li>
        <li>When transposed is set to true:</li>
          <ul><li>The N axis of weight must be greater than 0.</li>
          <li>The N axis of weight must be equal to the C axis of input.</li></ul>
        </ul>
     </td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">gradOutput constraints</th>
     <td>For non-transposed 1D, 2D, and 3D convolutions (transposed=false), all dimensions must be greater than or equal to 1. When input is an empty tensor, the N, C, D, H, W dimensions can be 0.</td>
     <td>Empty tensors are not supported.</td>
   </tr>
   <tr>
     <th scope="row">input constraints</th>
     <td>
        For non-transposed 1D, 2D, and 3D convolutions (transposed=false), all dimensions must be greater than or equal to 1.
          <ul><li>transposed=true: Empty tensors with N dimension 0 are supported. When N dimension is 0 and empty tensor constraints are met, D, H, and W dimensions can also be 0.</li>
          <li>transposed=false: Empty tensors with N or C dimension 0 are supported (if C dimension is 0, the C dimension of weight must also be 0). When N or C dimension is 0 and empty tensor constraints are met, D, H, and W dimensions can also be 0.</li></ul>
     </td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">weight constraints</th>
     <td>
        For non-transposed 2D and 3D convolutions (transposed=false), the H and W dimensions must be in the range [1,255], and all other dimensions must be greater than or equal to 1. For non-transposed 1D convolution (transposed=false),the L dimension must be in the range [1,255], and all other dimensions must be greater than or equal to 1.
           <ul><li>transposed=true: If input is an empty tensor and empty tensor constraints are met, C, D, H, and W axes can be 0.</li>
           <li>transposed=false: The C dimension can be 0 (which requires the C dimension of input to also be 0). If input is an empty tensor and empty tensor constraints are met, the D, H, and W axes can be 0.</li></ul>
     </td>
     <td>
        <ul>Empty tensors are not supported.</ul>
     </td>
   </tr>
   <tr>
     <th scope="row">stride constraints</th>
     <td>For non-transposed 3D convolution (transposed=false), strideD must be greater than or equal to 1, and strideH and strideW must be in the range [1, 63]. For non-transposed 1D and 2D convolutions (transposed=false), values must be greater than or equal to 1.</td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">padding constraints</th>
     <td>For non-transposed 3D convolution (transposed=false), paddingD must be greater than or equal to 0, and paddingH and paddingW must be in the range [0, 255]. For non-transposed 1D and 2D convolutions (transposed=false), all values must be in the range [0, 255].</td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">dilation constraints</th>
     <td>For non-transposed 1D, 2D, and 3D convolutions (transposed=false), all values must be in the range [1, 255].</td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">dtype constraints</th>
     <td>HIFLOAT8 and FLOAT8_E4M3FN are not supported.</td>
     <td>BFLOAT16, HIFLOAT8, and FLOAT8_E4M3FN are not supported.</td>
   </tr>
   <tr>
     <th scope="row">cubeMathType description</th>
     <td>
        <ul><li>0: No description available.</li>
        <li>1: When the input data type is FLOAT, it is converted to HFLOAT32 for computation. When the input is of other data types, it is not processed.</li>
        <li>2: This option is not supported when the input is BFLOAT16. When the input is of other data types, it is not processed.</li>
        <li>3: When the input data type is FLOAT, it is converted to HFLOAT32 for computation. When the input is of other data types, it is not processed.</li>
        </ul>
     </td>
     <td>
        <ul><li>0: When the input data type is FLOAT, the Cube unit does not currently support this mode. Selecting 0 will result in an error.</li>
        <li>1: When the input data type is FLOAT, it is converted to FLOAT16 for computation. When the input is of other data types, it is not processed.</li>
        <li>2: No description available.</li>
        <li>3: When the input data type is FLOAT, the Cube unit does not currently support this mode.</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">Other constraints</th>
     <td>-</td>
     <td>Currently, only 1D and 2D convolution backward passes are supported; 3D convolution backward is not supported.</td>
   </tr>
   </tbody>
  </table>
    
  - Formula 1:
    $$
        (input_{dim} + pad_{dim} \times 2) \ge ((weight_{dim}  - 1) \times dilation_{dim} + 1)
    $$

Due to hardware resource limitations, the operator may fail for certain parameter combinations. Analyze the error logs to diagnose the issue. If the error persists, click [Link](https://www.hiascend.com/support) to obtain technical support.


## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **DeviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(DeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*DeviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
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
    // 1. Perform initialization.
    auto ret = Init(deviceId, &stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on API definitions.
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

    // Create a gradOutput aclTensor.
    std::vector<float> gradOutputData(GetShapeSize(gradOutputShape), 1);
    aclTensor *gradOutput = nullptr;
    void *gradOutputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradOutputData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradOutputTensorPtr(gradOutput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradOutputDeviceAddrPtr(gradOutputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create an input aclTensor.
    std::vector<float> inputData(GetShapeSize(inputShape), 1);
    aclTensor *input = nullptr;
    void *inputDeviceAddr = nullptr;
    ret = CreateAclTensor(inputData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> inputDeviceAddrPtr(inputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create a weight aclTensor.
    std::vector<float> weightData(GetShapeSize(weightShape), 1);
    aclTensor *weight = nullptr;
    void *weightDeviceAddr = nullptr;
    ret = CreateAclTensor(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create a gradInput aclTensor.
    std::vector<float> gradInputData(GetShapeSize(inputShape), 1);
    aclTensor *gradInput = nullptr;
    void *gradInputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradInputData, inputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradInputTensorPtr(gradInput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradInputDeviceAddrPtr(gradInputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create a gradWeight aclTensor.
    std::vector<float> gradWeightData(GetShapeSize(weightShape), 1);
    aclTensor *gradWeight = nullptr;
    void *gradWeightDeviceAddr = nullptr;
    ret = CreateAclTensor(gradWeightData, weightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradWeightTensorPtr(gradWeight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradWeightDeviceAddrPtr(gradWeightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create a gradBias aclTensor.
    std::vector<float> gradBiasData(GetShapeSize(biasSize), 1);
    aclTensor *gradBias = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    ret = CreateAclTensor(gradBiasData, biasSize, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradBiasTensorPtr(gradBias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradBiasDeviceAddrPtr(gradBiasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create a biasSizes aclIntArray.
    aclIntArray *biasSizes = aclCreateIntArray(biasSize.data(), 1);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> biasSizesPtr(biasSizes, aclDestroyIntArray);
    CHECK_FREE_RET(biasSizes != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // Create a strides aclIntArray.
    aclIntArray *strides = aclCreateIntArray(stride.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
    CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // Create a pads aclIntArray.
    aclIntArray *pads = aclCreateIntArray(padding.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
    CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // Create a dilations aclIntArray.
    aclIntArray *dilations = aclCreateIntArray(dilation.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
    CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // Create an outputPads aclIntArray.
    aclIntArray *outputPads = aclCreateIntArray(outputPadding.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outputPadsPtr(outputPads, aclDestroyIntArray);
    CHECK_FREE_RET(outputPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // Create an outMask aclBoolArray.
    aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);
    std::unique_ptr<aclBoolArray, aclnnStatus (*)(const aclBoolArray *)> outMaskPtr(outMask, aclDestroyBoolArray);
    CHECK_FREE_RET(outMask != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the CANN operator library API. Modify the API as required.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API aclnnConvolutionBackwardGetWorkspaceSize.
    ret = aclnnConvolutionBackwardGetWorkspaceSize(gradOutput, input, weight, biasSizes, strides, pads, dilations,
                                                   transposed, outputPads, groups, outMask, cubeMathType, gradInput,
                                                   gradWeight, gradBias, &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
                   return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
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
    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
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
    // 1. (Boilerplate) Initialize the device and stream.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnConvolutionBackwardTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackwardTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
