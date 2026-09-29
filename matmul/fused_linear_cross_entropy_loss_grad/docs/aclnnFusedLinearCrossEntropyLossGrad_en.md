# aclnnFusedLinearCrossEntropyLossGrad

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/fused_linear_cross_entropy_loss_grad)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Description: This operator is a part of the cross entropy computation module in the vocabulary parallelism scenario. It solves the video memory and computing efficiency problems in the case of ultra-large vocabulary. This part involves the gradient computation implementation, which is used to compute the gradients of leaf nodes `input` and `weight`.
  The outputs of `aclnnFusedLinearOnlineMaxSum` and `aclnnFusedCrossEntropyLossWithMaxSum`, along with the global communication result related to `logits`, need to be obtained as the input of this API.
- Formula:

&emsp;&emsp; High-performance mode, where the value of **softmaxOptional** is not **nullptr**:

$$
\text{softmax} \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = \mathbf{1} - \text{target\_mask}.view(-1) \in \mathbb{R}^{BT}
$$

$$
\text{softmax}[\text{arange\_1d}, \text{masked\_target}] \leftarrow \text{softmax}[\text{arange\_1d}, \text{masked\_target}] - \text{softmax\_update}
$$

$$
\text{softmax} \leftarrow \text{softmax} \odot \text{grad}.unsqueeze(-1) \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_input} = \text{softmax} \cdot \text{weight}^T \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{softmax}^T \cdot \text{input} \in \mathbb{R}^{V \times H}
$$

<br>
&emsp;&emsp; Video memory–saving mode, where the value of **softmaxOptional** is **nullptr**:

$$
\text{vocab\_parallel\_logits} = \text{input} \cdot \text{weight}^T \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{logits\_sub} = \text{vocab\_parallel\_logits} - \text{logits\_max}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} = \exp(\text{logits\_sub}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} \gets \frac{\text{exp\_logits}}{\text{sum\_exp\_logits}.unsqueeze(-1)} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_logits} = \text{exp\_logits} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_2d} = \text{grad\_logits}.view(-1, \text{partition\_vocab\_size}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \quad \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = 1 - \text{target\_mask}.view(-1) \quad \in \mathbb{R}^{BT}
$$

$$
\text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] \gets \text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] - \text{softmax\_update}
$$

$$
\text{grad\_logits} \gets \text{grad\_logits} \odot \text{grad}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_input} = \text{grad\_logits} \cdot \text{weight} \quad \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{grad\_logits}^T \cdot \text{input} \quad \in \mathbb{R}^{V \times H}
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnFusedLinearCrossEntropyLossGrad` is called to perform computation.

  ```Cpp
  aclnnStatus aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(
    const aclTensor   *grad,
    const aclTensor   *input,
    const aclTensor   *weight,
    const aclTensor   *targetMask,
    const aclTensor   *maskedTarget,
    float              labelSmoothing,
    const aclTensor   *logitsMaxOptional,
    const aclTensor   *sumExpLogitsOptional,
    const aclTensor   *softmaxOptional,
    aclTensor         *inputGradOut,
    aclTensor         *weightGradOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
  ```

  ```Cpp
  aclnnStatus aclnnFusedLinearCrossEntropyLossGrad(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
  ```

## aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
    <col style="width: 169px">
    <col style="width: 121px">
    <col style="width: 264px">
    <col style="width: 253px">
    <col style="width: 242px">
    <col style="width: 148px">
    <col style="width: 135px">
    <col style="width: 146px">
    </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>grad</td>
        <td>Input</td>
        <td>Gradient of the current node, which is the input grad in the formula.</td>
        <td>Empty tensors are supported.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>input</td>
        <td>Input</td>
        <td>Input matrix for matrix multiplication, which is the input in the formula.</td>
        <td>The length of the first dimension is the same as that of the input grad. Empty tensors are supported.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Weight matrix for matrix multiplication, which is the weight in the formula.</td>
        <td>The data type is the same as that of the input. The length of the first dimension of the shape cannot be less than 128, and the length of the second dimension is the same as that of the second dimension of the input.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>targetMask</td>
        <td>Input</td>
        <td>Intermediate variable, indicating whether the corresponding word ID is within the target range, which is the target_mask in the formula.</td>
        <td><ul><li>Each bit of data represents a Boolean value. 0 indicates false, and 1 indicates true. </li><li>The length of the shape multiplied by 8 must be greater than or equal to the length of the input grad.</li></ul></td>
        <td>UINT8</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maskedTarget</td>
        <td>Input</td>
        <td>Intermediate variable, indicating the local index of the corresponding word ID mapped to the vocabulary shard of the current device. Invalid targets are processed by the targetMask, which is the masked_target in the formula.</td>
        <td>The shape length is the same as that of the input grad.</td>
        <td>INT64, INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>labelSmoothing</td>
        <td>Input</td>
        <td>Label smoothing coefficient, which is used to alleviate overfitting.</td>
        <td>Currently, only the value 0 is supported.</td>
        <td>FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logitsMaxOptional</td>
        <td>Optional input</td>
        <td>Intermediate variable, which is the maximum value of the global logits, that is, logits_max in the formula.</td>
        <td><ul><li>The input nullptr is supported. In the scenario where nullptr is input, a valid softmaxOptional input must be provided. </li><li>The shape length is the same as that of the input grad.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sumExpLogitsOptional</td>
        <td>Optional input</td>
        <td>Intermediate variable, which is the processed logits, that is, sum_exp_logits in the formula.</td>
        <td><ul><li>The input nullptr is supported. In the scenario where nullptr is input, a valid softmaxOptional input must be provided. </li><li>The shape length is the same as that of the input grad.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxOptional</td>
        <td>Optional input</td>
        <td>Intermediate variable, which is the result of matrix multiplication and softmax in the formula.</td>
        <td><ul><li>The nullptr value is supported. When nullptr is input, valid logitsMaxOptional and sumExpLogitsOptional inputs must be provided. When a non-null value is input, logitsMaxOptional and sumExpLogitsOptional inputs are invalid. </li><li>The length of the first dimension of the shape is the same as that of the input grad, and the length of the second dimension is the same as that of the first dimension of the input weight.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>inputGradOut</td>
        <td>Output</td>
        <td>Gradient of the corresponding leaf node input, which is grad_input in the formula.</td>
        <td><ul><li>The data type is the same as that of the input. </li><li>The length of the first dimension of the shape is the same as that of the input grad, and the length of the second dimension is the same as that of the second dimension of the input weight.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightGradOut</td>
        <td>Output</td>
        <td>Gradient of the weight leaf node, that is, grad_weight in the formula.</td>
        <td><ul><li>The data type is the same as that of the input. </li><li>The length of the first dimension of the shape is the same as that of the first dimension of the input weight, and the length of the second dimension is the same as that of the input grad.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Returns**
  
  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>The input non-optional parameter is a null pointer.</td>
    </tr>
    <tr>
      <td>When the input softmaxOptional is a null pointer, logitsMaxOptional or sumExpLogitsOptional is also a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The value of labelSmoothing is not supported.</td>
    </tr>
    <tr>
      <td>The input data type is not supported.</td>
    </tr>
    <tr>
      <td>The input data format is not supported.</td>
    </tr>
    <tr>
      <td>The input shape is not supported and does not meet the length requirement.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedLinearCrossEntropyLossGrad

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize.</td>
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
  
  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic description:
  - The aclnnFusedLinearCrossEntropyLossGrad is implemented in a deterministic manner by default.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_linear_cross_entropy_loss_grad.h"

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
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
    // (Fixed writing) Initialize resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
std::vector<T> GenZeroVector(const std::vector<int64_t>& shape) {
    // 1. Calculate the total number of elements.
    size_t total = 1;
    for (auto dim : shape) {
        total *= dim;
    }

    // 2. Fill in 0.
    std::vector<T> vec(total);
    for (auto& elem : vec) {
        elem = 0;
    }
    return vec;
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

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateEmptyAclTensor(const std::vector<int64_t> &shape, void **deviceAddr,
                         aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external ACL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API definition.
    int64_t BT = 1024;
    int64_t V = 1024;
    int64_t H = 1024;
    std::vector<int64_t> gradShape = {BT};
    std::vector<int64_t> inputShape = {BT, H};
    std::vector<int64_t> weightShape = {V, H};
    std::vector<int64_t> targetMaskShape = {BT};
    std::vector<int64_t> maskedTargetShape = {BT};
    std::vector<int64_t> softmaxOptionalShape = {BT, V};
    std::vector<int64_t> inputGradOutShape = {BT, H};
    std::vector<int64_t> weightGradOutShape = {V, H};
    void *gradDeviceAddr = nullptr;
    void *inputDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *targetMaskDeviceAddr = nullptr;
    void *maskedTargetDeviceAddr = nullptr;
    void *softmaxOptionalDeviceAddr = nullptr;
    void *inputGradOutDeviceAddr = nullptr;
    void *weightGradOutDeviceAddr = nullptr;
    aclTensor *grad = nullptr;
    aclTensor *input = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *targetMask = nullptr;
    aclTensor *maskedTarget = nullptr;
    float labelSmoothing = 0.0;
    aclTensor *logitsMaxOptional = nullptr;
    aclTensor *sumExpLogitsOptional = nullptr;
    aclTensor *softmaxOptional = nullptr;
    aclTensor *inputGradOut = nullptr;
    aclTensor *weightGradOut = nullptr;
    // Create an aclTensor.
    auto gradData = GenZeroVector<int32_t>(gradShape);
    ret = CreateAclTensor<int32_t>(gradData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto inputData = GenZeroVector<int16_t>(inputShape);
    ret = CreateAclTensor<int16_t>(inputData, inputShape, &inputDeviceAddr, aclDataType::ACL_BF16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto weightData = GenZeroVector<int16_t>(weightShape);
    ret = CreateAclTensor<int16_t>(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_BF16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto targetMaskData = GenZeroVector<int8_t>(targetMaskShape);
    ret = CreateAclTensor<int8_t>(targetMaskData, targetMaskShape, &targetMaskDeviceAddr, aclDataType::ACL_UINT8, &targetMask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto maskedTargetData = GenZeroVector<int32_t>(maskedTargetShape);
    ret = CreateAclTensor<int32_t>(maskedTargetData, maskedTargetShape, &maskedTargetDeviceAddr, aclDataType::ACL_INT32, &maskedTarget);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto softmaxOptionalData = GenZeroVector<int32_t>(softmaxOptionalShape);
    ret = CreateAclTensor<int32_t>(softmaxOptionalData, softmaxOptionalShape, &softmaxOptionalDeviceAddr, aclDataType::ACL_FLOAT, &softmaxOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto inputGradOutData = GenZeroVector<int16_t>(inputGradOutShape);
    ret = CreateAclTensor<int16_t>(inputGradOutData, inputGradOutShape, &inputGradOutDeviceAddr, aclDataType::ACL_BF16, &inputGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto weightGradOutData = GenZeroVector<int16_t>(weightGradOutShape);
    ret = CreateAclTensor<int16_t>(weightGradOutData, weightGradOutShape, &weightGradOutDeviceAddr, aclDataType::ACL_BF16, &weightGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnFusedLinearCrossEntropyLossGrad.
    ret = aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(grad, input, weight, targetMask, maskedTarget, labelSmoothing, logitsMaxOptional, sumExpLogitsOptional, softmaxOptional, inputGradOut, weightGradOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnFusedLinearCrossEntropyLossGrad.
    ret = aclnnFusedLinearCrossEntropyLossGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearCrossEntropyLossGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    // inputGradOut
    auto size = GetShapeSize(inputGradOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), inputGradOutDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < 16; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor. Modify the code based on the API definition.
    aclDestroyTensor(grad);
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(targetMask);
    aclDestroyTensor(maskedTarget);
    aclDestroyTensor(softmaxOptional);
    aclDestroyTensor(inputGradOut);
    aclDestroyTensor(weightGradOut);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(gradDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(targetMaskDeviceAddr);
    aclrtFree(maskedTargetDeviceAddr);
    aclrtFree(softmaxOptionalDeviceAddr);
    aclrtFree(inputGradOutDeviceAddr);
    aclrtFree(weightGradOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
