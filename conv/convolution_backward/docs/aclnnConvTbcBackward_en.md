# aclnnConvTbcBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/conv/convolution_backward)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Computes the backward pass of a temporal convolution.

- Formula:

  Assume the forward input $input$ of Conv_tbc has shape $(H_{\text{in}},N,C_{\text{in}})$, the output gradient $gradOutput$ has shape $(H_{\text{out}},N,C_{\text{out}})$, the convolution kernel $weight$ has shape $(K,C_{\text{in}},C_{\text{out}})$, and the bias $bias$ has shape $(C_{\text{out}})$.

  $$
  H_{out} = \lfloor \frac{H_{in} + 2 \cdot pad - K}{S} \rfloor + 1
  $$
  The backward pass computes gradients with respect to the forward-pass tensors: the input tensor $x$ (corresponding to **input** in the function prototype), the convolution kernel weights $w$ (corresponding to **weight** in the function prototype), and the bias $b$.

  - Gradient with respect to $x$, $\frac{\partial L}{\partial x}$ (corresponding to the **gradInput** parameter in the function prototype):

    $$
    \frac{\partial L}{\partial x_{t,b,c_{in}}} = \sum_{k=0}^{K-1} \sum_{c_{out}=0}^{C_{out}-1} \frac{\partial L}{\partial y_{t-k,b,c_{out}}} \cdot w_{k,c_{in},c_{out}}
    $$

    Where $L$ is the loss function and $\frac{\partial L}{\partial y}$ is the gradient of the loss $L$ with respect to the output tensor $y$ (corresponding to the **gradOutput** parameter in the function prototype).

  - Gradient with respect to $w$, $\frac{\partial L}{\partial w}$ (corresponding to the **gradWeight** parameter in the function prototype):

    $$
    \frac{\partial L}{\partial w_{k,c_{in},c_{out}}} = \sum_{b=0}^{N-1} \sum_{t=0}^{H_{out}-1} x_{t \cdot S+k,b,c_{in}} \cdot \frac{\partial L}{\partial y_{t,b,c_{out}}}
    $$

  - Gradient with respect to $b$, $\frac{\partial L}{\partial b}$ (corresponding to the **gradBias** parameter in the function prototype):

    $$
    \frac{\partial L}{\partial b_{c_{out}}} = \sum_{b=0}^{N-1}\sum_{t=0}^{H_{\text{out}}-1} \frac{\partial L}{\partial y_{t,b,c_{out}}}
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnConvTbcBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnConvTbcBackward** is called to perform computation.

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

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width:180px">
    <col style="width:120px">
    <col style="width:280px">
    <col style="width:320px">
    <col style="width:250px">
    <col style="width:120px">
    <col style="width:140px">
    <col style="width:140px">
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
     <td>gradOutput in the formula.</td>
     <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The shape can be represented by Formula 1 (see the description below the table).</li>
       </ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
     <td>√</td>
    </tr>
    <tr>
     <td>input</td>
     <td>Input</td>
     <td>input in the formula.</td>
     <td>
     
       <ul><li>Empty tensors are supported.</li>
       <li>The shape can be represented by Formula 2 (see the description below the table).</li>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
     <td>√</td>
    </tr>
    <tr>
     <td>weight</td>
     <td>Input</td>
     <td>weight in the formula.</td>
     <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The shape can be represented by Formula 3 (see the description below the table).</li>
       </ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>See <a href="#constraints" target="_blank">Constraints.</a></td>
     <td>√</td>
    </tr>
    <tr>
     <td>bias</td>
     <td>Input</td>
     <td>bias in the formula.</td>
     <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The shape can be represented by Formula 4 (see the description below the table).</li>
       </ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>-</td>
     <td>√</td>
    </tr>
    <tr>
     <td>pad</td>
     <td>Input</td>
     <td>Number of padding elements to add on both sides of the T dimension.</td>
     <td>
       The value must be in the range [0, 255].
     </td>
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
     <td>gradInput in the formula.</td>
     <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The data type must be the same as that of input.</li>
       <li>The shape can be represented by Formula 5 (see the description below the table).</li>
       </ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16</td>
     <td>ND, NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradWeight</td>
     <td>Output</td>
     <td>gradWeight in the formula.</td>
     <td>
       <ul><li>Empty tensors are supported.</li>
       <li>The data type must be the same as that of weight.</li>
       <li>The shape can be represented by Formula 6 (see the description below the table).</li>
       </ul>
     </td>
     <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN</td>
     <td>ND, NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradBias</td>
     <td>Output</td>
     <td>gradBias in the formula.</td>
     <td>
      <ul><li>Empty tensors are supported.</li>
      <li>The data type must be the same as that of bias.</li>
      <li>The shape can be represented by Formula 7 (see the description below the table).</li>
      </ul>
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

  - Formula 1: $(H_{\text{out}},N,C_{\text{out}})$
  - Formula 2: $(H_{\text{in}},N,C_{\text{in}})$
  - Formula 3: $(K,C_{\text{in}},C_{\text{out}})$
  - Formula 4: $(C_{\text{out}})$
  - Formula 5: $(H_{\text{in}},N,C_{\text{in}})$
  - Formula 6: $(K,C_{\text{in}},C_{\text{out}})$
  - Formula 7: $(C_{\text{out}})$

- **Returns**

    **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md). 

    The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 650px">
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

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnConvTbcBackwardGetWorkspaceSize.</td>
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
  - **aclnnConvTbcBackward** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computation.

  <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
    <col style="width: 168px">
    <col style="width: 600px">
    <col style="width: 395px">
    </colgroup>
   <thead>
    <tr>
     <th>Constraint Type</th>
     <th><term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term></th>
     <th><term>Atlas training series products</term></th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">gradOutput constraints</th>
     <td>All dimension sizes must be in the range [1, 2147483646].</td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">input constraints</th>
     <td>All dimension sizes must be in the range [1, 2147483646].</td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">weight constraints</th>
     <td>The size of the L dimension must be in the range [1, 255], and the sizes of all other dimensions must be in the range [1, 2147483646].</td>
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
        <li>2: This option is not supported when the input is BFLOAT16.</li>
        <li>3: When the input data type is FLOAT, it is converted to HFLOAT32 for computation. When the input is of other data types, this option is not supported.</li>
        </ul>
     </td>
     <td>
        <ul><li>0: When the input data type is FLOAT, the Cube unit does not currently support this mode. Selecting 0 will result in an error.</li>
        <li>1: When the input data type is FLOAT, it is converted to FLOAT16 for computation. When the input is of other data types, it is not processed.</li>
        <li>2: No description available.</li>
        <li>3: When the input data type is FLOAT, the Cube unit does not currently support this mode. Selecting 3 will result in an error.</li>
        </ul>
     </td>
   </tr>
   </tbody>
  </table>

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
