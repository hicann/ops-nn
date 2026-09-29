# aclnnConvTbcBackward

## Product Support

| Product                                                    | Supported|
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                  |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>|    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
| <term>Atlas inference products</term>   |     ×    |
| <term>Atlas training products</term>   |     √    |

## Function

- This API is used to implement the backpropagation of 1D convolution with the input and output dimensions being **T**** (time or spatial dimension), **B** (batch), and **C** (channel).

- The calculation formula is as follows: Assume that the shape of the input $input$ of the Conv_tbc forward propagation is $(H_{\text{in}},N,C_{\text{in}})$, the shape of the output gradient $gradOutput$
  is $(H_{\text{out}},N,C_{\text{out}})$, the shape of the convolution kernel $weight$ is $(K,C_{\text{in}},C_{\text{out}})$, and the shape of the bias $bias$
  is $(C_{\text{out}})$. During backpropagation, the input padding is $pad$. The relationship between the preceding parameters is as follows:

  $$
  H_{out} = {H_{in} + 2 \cdot pad - K} + 1
  $$

  The backpropagation of convolution needs to calculate the gradients of the forward input tensor $x$ (corresponding to the input in the function prototype), convolution kernel weight tensor $w$
  (corresponding to the weight in the function prototype), and bias $b$ (corresponding to the bias in the function prototype).

    - Gradient with respect to $x$, $\frac{\partial L}{\partial x}$ (corresponding to the **gradInput** parameter in the function prototype):

      $$
      \frac{\partial L}{\partial x_{t,b,c_{in}}} = \sum_{k=0}^{K-1} \sum_{c_{out}=0}^{C_{out}-1} \frac{\partial L}{\partial y_{t-k,b,c_{out}}} \cdot w_{k,c_{in},c_{out}}
      $$

      $N$ indicates the batch size, $C$ indicates the number of channels, $H$ indicates the time or spatial dimension, and $L$
      indicates the loss function. $\frac{\partial L}{\partial y}$ indicates the gradient of the output tensor $y$ to $L$ (corresponding to the self parameter in the function prototype).

    - Gradient with respect to $w$, $\frac{\partial L}{\partial w}$ (corresponding to the **gradWeight** parameter in the function prototype):

      $$
      \frac{\partial L}{\partial w_{k,c_{in},c_{out}}} = \sum_{b=0}^{N-1} \sum_{t=0}^{H_{out}-1} x_{t+k,b,c_{in}} \cdot \frac{\partial L}{\partial y_{t,b,c_{out}}}
      $$

    - Gradient with respect to $b$, $\frac{\partial L}{\partial b}$ (corresponding to the **gradBias** parameter in the function prototype):

      $$
      \frac{\partial L}{\partial b_{c_{out}}} = \sum_{b=0}^{N-1}\sum_{t=0}^{H_{\text{out}}-1} \frac{\partial L}{\partial y_{t,b,c_{out}}}
      $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls.
You must call aclnnConvTbcBackwardGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnConvTbcBackward to perform the computation.

```cpp
aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(
    const aclTensor *self, 
    const aclTensor *input, 
    const aclTensor *weight, 
    const aclTensor *bias, 
    int64_t          pad, 
    int8_t           cubeMathType, 
    aclTensor       *gradInput, 
    aclTensor       *gradWeight, 
    aclTensor       *gradBias, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnConvTbcBackward(
    void                *workspace, 
    uint64_t             workspaceSize, 
    aclOpExecutor       *executor, 
    const aclrtStream    stream)
```

## aclnnConvTbcBackwardGetWorkspaceSize

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
    </tr>
    </thead>
    <tbody>
    <tr>
     <td>self</td>
     <td>Input</td>
     <td>Gradient of the output tensor y with respect to L in the formula, indicating the input of the backward convolution.</td>
     <td>
       <ul><li>Empty tensors are supported. </li><li>The shape is (N,C<sub>out</sub>,H<sub>out</sub>). </li><li>Its data type and the data type of weight must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a>).</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>3</td>
     <td>√</td>
    </tr>
    <tr>
     <td>input</td>
     <td>Input</td>
     <td>X in the formula, indicating the forward convolution input.</td>
     <td>
       <ul>
        <li>Empty tensors are supported.</li>
        <li>The shape is (N,C<sub>in</sub>,H<sub>in</sub>).</li>
        <li>Its data type and the data type of weight must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a>).</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>3</td>
     <td>√</td>
    </tr>
    <tr>
     <td>weight</td>
     <td>Input</td>
     <td>w in the formula indicates the convolution weight.</td>
     <td>
       <ul>
        <li>Empty tensors are supported.</li>
        <li>The shape is (C<sub>out</sub>,C<sub>in</sub>,K).</li>
        <li>Its data type and the data type of weight must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a>).</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>3</td>
     <td>√</td>
    </tr>
    <tr>
     <td>bias</td>
     <td>Input</td>
     <td>b in the formula indicates the convolution bias.</td>
     <td>
       <ul>
        <li>The shape is (C<sub>out</sub>).</li>
        <li>The value is a one-dimensional array and must have the same dimension as the first dimension of weight. A null pointer cannot be passed.</li>
        <li>The data type is the same as that of self and weight.</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>1</td>
     <td>√</td>
    </tr>
    <tr>
     <td>pad</td>
     <td>Input</td>
     <td>Number of padding elements on the left and right sides of the input in the H dimension during backpropagation.</td>
     <td>
       <ul><li>The value must be within the range of [0, 255].</li></ul>
     </td>
     <td>INT64</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>cubeMathType</td>
     <td>Input</td>
     <td>Computation logic to be used by the Cube unit.</td>
     <td>
       Supported enumerations:
       <ul>
       <li>0: KEEP_DTYPE. The input data type is retained for computation.</li>
       <li>1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with a reduced precision.</li>
       <li>2: USE_FP16. The input data type can be converted to FLOAT16 for computation.</li>
       <li>3: USE_HF32. The input data type can be converted to HFLOAT32 for computation.</li>
       </ul>
     </td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradInput</td>
     <td>Output</td>
     <td>Gradient of the input tensor x with respect to L in the formula.</td>
     <td>
       <ul>
        <li>Empty tensors are supported.</li>
        <li>The data type must be the same as that of input.</li>
        <li>shape is (N,C<sub>in</sub>,H<sub>in</sub>).</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradWeight</td>
     <td>Output</td>
     <td>Gradient of the loss L with respect to the convolution kernel weight tensor w.</td>
     <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The data type must be the same as that of weight.</li>
       <li>The shape is (C<sub>out</sub>,C<sub>in</sub>,K).</li></ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN</td>
     <td>ND, NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradBias</td>
     <td>Output</td>
     <td>Gradient of the loss L with respect to the bias b.</td>
     <td>
      <ul><li>Empty tensors are supported.</li></ul>
      <li>The data type must be the same as that of bias.</li>
      <li>The shape is (C<sub>out</sub>).</li>
    </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
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

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:1050px">
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>The data type or format of self, input, weight, bias, gradInput, gradWeight, or gradBias is not supported.</td>
    </tr>
    <tr>
      <td>The data types of self, input, weight, and bias do not match.</td>
    </tr>
    <tr>
      <td>The shape of gradInput, gradWeight, or gradBias does not match the inferred shape (infershape).</td>
    </tr>
    <tr>
      <td>The shape of gradInput, gradWeight, or gradBias contains a value less than 0.</td>
    </tr>
    <tr>
      <td>self, input, or weight is not a 3D tensor.</td>
    </tr>
    <tr>
      <td>bias is not a 1D tensor.</td>
    </tr>
    <tr>
      <td>The third dimension value of input is not equal to the second dimension value of weight.</td>
    </tr>
    <tr>
      <td>The value of bias is not equal to the third dimension value of weight.</td>
    </tr>
    <tr>
      <td>The value of pad does not meet the requirements.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>Internal API verification error, usually caused by unsupported input data or attribute specifications.</td>
      </tr>
      </tbody>
  </table>

## aclnnConvTbcBackward

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
       <col style="width:100px">
       <col style="width:100px">
       <col style="width:950px">
       </colgroup>
    <thead>
     <tr>
       <th>Name</th>
       <th>Input/Output</th>
       <th>Description</th>
     </tr>
    </thead>
    <tbody>
     <tr>
       <td>workspace</td>
       <td>Input</td>
       <td>Memory address of the workspace to be allocated on the device.</td>
     </tr>
     <tr>
       <td>workspaceSize</td>
       <td>Input</td>
       <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnConvTbcBackwardGetWorkspaceSize.</td>
     </tr>
     <tr>
       <td>executor</td>
       <td>Input</td>
       <td>Operator executor, containing the operator computation process.</td>
     </tr>
     <tr>
       <td>stream</td>
       <td>Input</td>
       <td>Stream for executing the task.</td>
     </tr>
    </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - **aclnnConvTbcBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computation.

<table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
  <col style="width: 150px">
  <col style="width: 440px">
  <col style="width: 410px">
  <col style="width: 400px">
    </colgroup>
   <thead>
    <tr>
     <th>Constraint Type</th>
     <th>Ascend 950PR/Ascend 950DT</th>
     <th><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></th>
     <th><term>Atlas training products</term></th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">Self constraint</th>
     <td>
      <ul>
        <li>The N dimension is greater than or equal to 0, and the C dimension is greater than or equal to 0. (The C dimension can be 0 only when the N dimension of the weight is 0.)</li>
        <li>The L dimension is greater than or equal to 0. (The L dimension can be 0 only when the L dimension of the self is 0.)</li>
      </ul>
     </td>   
     <td colspan="2">
        The N, C, and L dimensions are greater than or equal to 0. (The dimensions can be 0 only when the N, C, or L dimension of the self is 0.)
     </td>
   </tr>
   <tr>
     <th scope="row">input constraints</th>
     <td>
      The N and C dimensions of the input must be greater than or equal to 0, and the L dimension must be greater than or equal to 0. (The scenario where the L dimension is 0 is supported only when the L dimension derived from out is also 0.)
      </td>   
     <td>
        The input data type does not support HIFLOAT8. The N, C, and L dimensions must be greater than or equal to 0.
     </td>
     <td>
        The input data type does not support BFLOAT16 or HIFLOAT8. The N, C, and L dimensions must be greater than or equal to 0.
     </td>
   </tr>
   <tr>
     <th scope="row">weight constraints</th>
     <td>
        <ul>
          <li>The weight supports the N and C dimensions greater than or equal to 0, and the L dimension greater than or equal to 0. (The scenario where the L dimension is 0 is supported only when the L dimension derived from out is also 0.)</li>
          <li>The weight supports the N dimension greater than or equal to 0 (the scenario where the N dimension is 0 is supported only when the N dimension of bias and the C dimension of out are also 0). The size of the C dimension is the same as that of the C dimension of self. The size of the L dimension must be within the range of [1, 255].</li>
        </ul>
     </td>   
     <td>
          The weight data type does not support HIFLOAT8. The N, C, and L dimensions must be greater than or equal to 0.
     </td>
     <td>
          The weight data type does not support BFLOAT16 or HIFLOAT8. The N, C, and L dimensions must be greater than or equal to 0.
     </td>
   </tr>
   <tr>
     <th scope="row">dtype constraints</th>
     <td>
        HIFLOAT8 and FLOAT8_E4M3FN are supported only in the gradWeight parameter.
     </td>   
     <td>
        HIFLOAT8 and FLOAT8_E4M3FN are not supported.
     </td>
     <td>
        BFLOAT16, HIFLOAT8, and FLOAT8_E4M3FN are not supported.
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType description</th>
     <td>
        <ul>
        <li>1: When the input data type is FLOAT, it is converted to HFLOAT32 for computation. When the input is of other data types, it is not processed.</li>
        <li>2: This option is not supported when the input is BFLOAT16.</li>
        <li>3: When the input data type is FLOAT, it is converted to HFLOAT32 for computation. When the input is of other data types, this option is not supported.</li>
        </ul>
     </td>
     <td>
        <ul><li>0: When the input data type is FLOAT, the Cube unit does not currently support this mode. Selecting 0 will result in an error.</li>
        <li>1: If the input is of type FLOAT, the processor converts it to HFLOAT32 for computation. When the input is of other data types, it is not processed.</li>
        <li>2: This option is not supported when the input is of type BFLOAT16.</li>
        <li>3: When the input data type is FLOAT, the Cube unit does not currently support this mode. Selecting 3 will result in an error.</li>
        </ul>
     </td>
     <td>
        <ul><li>0: When the input data type is FLOAT, the Cube unit does not currently support this mode. Selecting 0 will result in an error.</li>
        <li>1: When the input data type is FLOAT, it is converted to FLOAT16 for computation. When the input is of other data types, it is not processed.</li>
        <li>2: This option is not supported when the input is of type BFLOAT16.</li>
        <li>3: This option is not supported currently.</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">Other constraints</th>
     <td>
        <ul>The gradient calculation behavior in the padding area depends on the input shape. Depending on the operator optimization policy, the gradient in the padding area may be directly set to 0.</ul>
     </td>
     <td>-</td>
     <td>-</td>
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
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
    if (shape.size() == 4) {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                  shape.data(), shape.size(), *deviceAddr);
    } else {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *deviceAddr);
    }

    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnConvTbcBackwardTest(int32_t deviceId, aclrtStream &stream)
{
    // 1. Perform initialization.
    auto ret = Init(deviceId, &stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on API definitions.
    std::vector<int64_t> selfShape = {5, 1, 2};
    std::vector<int64_t> inputShape = {5, 1, 2};
    std::vector<int64_t> weightShape = {1, 2, 2};
    std::vector<int64_t> biasShape = {2};
    const int64_t pad = 0;
    int8_t cubeMathType = 1;

    std::vector<int64_t> gradInputShape = {5, 1, 2};
    std::vector<int64_t> gradWeightShape = {1, 2, 2};
    std::vector<int64_t> gradBiasShape = {2};

    // Create a self aclTensor.
    std::vector<float> selfData(GetShapeSize(selfShape), 1);
    aclTensor *self = nullptr;
    void *selfdeviceAddr = nullptr;
    ret = CreateAclTensor(selfData, selfShape, &selfdeviceAddr, aclDataType::ACL_FLOAT, &self);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> selfdeviceAddrPtr(selfdeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create an input aclTensor.
    std::vector<float> inputData(GetShapeSize(inputShape), 1);
    aclTensor *input = nullptr;
    void *inputdeviceAddr = nullptr;
    ret = CreateAclTensor(inputData, inputShape, &inputdeviceAddr, aclDataType::ACL_FLOAT, &input);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> inputDeviceAddrPtr(inputdeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create a weight aclTensor.
    std::vector<float> weightData(GetShapeSize(weightShape), 1);
    aclTensor *weight = nullptr;
    void *weightDeviceAddr = nullptr;
    ret = CreateAclTensor(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create a bias aclTensor.
    std::vector<float> biasData(GetShapeSize(biasShape), 1);
    aclTensor *bias = nullptr;
    void *biasDeviceAddr = nullptr;
    ret = CreateAclTensor(biasData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
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
    std::vector<float> gradBiasData(GetShapeSize(gradBiasShape), 1);
    aclTensor *gradBias = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    ret = CreateAclTensor(gradBiasData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradBiasTensorPtr(gradBias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradBiasDeviceAddrPtr(gradBiasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Modify the API as required.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnConvTbcBackward.
    ret = aclnnConvTbcBackwardGetWorkspaceSize(self, input, weight, bias, pad, cubeMathType, gradInput, gradWeight,
                                               gradBias, &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
                   return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnConvTbcBackward.
    ret = aclnnConvTbcBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackward failed. ERROR: %d\n", ret); return ret);
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
    auto ret = aclnnConvTbcBackwardTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackwardTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
