# aclnnFusedLinearCrossEntropyLossGrad

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/fused_linear_cross_entropy_loss_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

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

</br>
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

- `aclnnStatus aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(const aclTensor *grad, const aclTensor *input, const aclTensor *weight, const aclTensor *targetMask, const aclTensor *maskedTarget, float labelSmoothing, const aclTensor *logitsMaxOptional, const aclTensor *sumExpLogitsOptional, const aclTensor *softmaxOptional, aclTensor *inputGradOut, aclTensor *weightGradOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnFusedLinearCrossEntropyLossGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize

- **Parameters:**
  
  - **grad** (aclTensor*, computation input): gradient of the current node, grad in the formula, and aclTensor on the device. The data type can be FLOAT32, and the shape can be 1D. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) and empty tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **input** (aclTensor*, computation input): input matrix of matrix multiplication, input in the formula, and aclTensor on the device. The data type can be FLOAT16 or BFLOAT16. The shape can be 2D, and the length of the first dimension is the same as that of **grad**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) and empty tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **weight** (aclTensor*, computation input): weight matrix of matrix multiplication, weight in the formula, and aclTensor on the device. The data type can be FLOAT16 or BFLOAT16, which is the same as that of **input**. The shape can be 2D. The length of the first dimension cannot be less than 128, and the length of the second dimension is the same as that of the second dimension of **input**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) and empty tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **targetMask** (aclTensor*, computation input): intermediate variable, target_mask in the formula, and aclTensor on the device. This parameter indicates whether the corresponding word ID is within the target range. The data type can be UINT8. Each bit represents a Boolean value, where **0** indicates false and **1** indicates true. The shape can be 1D, and the length multiplied by 8 must be greater than or equal to the length of **grad**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) and empty tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **maskedTarget** (aclTensor*, computation input): intermediate variable, masked_target in the formula, and aclTensor on the device. This parameter indicates the local index of the corresponding word ID mapped to the vocabulary shard of the current device. Invalid targets are processed by **targetMask**. The data type can be INT64 or INT32. The shape can be 1D, and the length is the same as that of **grad**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) and empty tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **labelSmoothing** (float, computation input): label smoothing coefficient, which is used to alleviate overfitting. Currently, the value can only be **0**. This is an attribute parameter.
  - **logitsMaxOptional** (aclTensor*, optional input): intermediate variable, maximum value of global logits, logits_max in the formula, and aclTensor on the device. This parameter is optional, with value **nullptr** supported. If this parameter is set to **nullptr**, a valid value of **softmaxOptional** must be provided. The data type can be FLOAT32. The shape can be 1D, and the length is the same as that of **grad**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) and empty tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **sumExpLogitsOptional** (aclTensor*, optional input): intermediate variable, processed logits, sum_exp_logits in the formula, and aclTensor on the device. This parameter is optional, with value **nullptr** supported. If this parameter is set to **nullptr**, a valid value of **softmaxOptional** must be provided. The data type can be FLOAT32. The shape can be 1D, and the length is the same as that of **grad**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) and empty tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **softmaxOptional** (aclTensor*, computation input): intermediate variable, result of matrix multiplication, softmax in the formula, and aclTensor on the device. This parameter is optional, with value **nullptr** supported. If this parameter is set to **nullptr**, valid values of **logitsMaxOptional** and **sumExpLogitsOptional** must be provided. If this parameter is set to any other value, the values of **logitsMaxOptional** and **sumExpLogitsOptional** are invalid. The data type can be FLOAT32, and the shape can be 2D. The length of the first dimension is the same as that of **grad**, and the length of the second dimension is the same as that of the first dimension of **weight**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) and empty tensors are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **inputGradOut** (aclTensor*, computation output): gradient of the leaf node **input**, grad_input in the formula, and aclTensor on the device. The data type can be FLOAT16 or BFLOAT16, which is the same as that of **input**. The shape can be 2D. The length of the first dimension is the same as that of **grad**, and the length of the second dimension is the same as that of the second dimension of **weight**. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **weightGradOut** (aclTensor*, computation output): gradient of the leaf node **weight**, grad_weight in the formula, and aclTensor on the device. The data type can be FLOAT16 or BFLOAT16, which is the same as that of **input**. The shape can be 2D. The length of the first dimension is the same as that of the first dimension of **weight**, and the length of the second dimension is the same as that of **grad**. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.

- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed non-optional parameter is a null pointer.
                                        2. If the passed softmaxOptional is a null pointer, logitsMaxOptional or sumExpLogitsOptional is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The input value of labelSmoothing is not supported.
                                        2. The input data type is not supported.
                                        3. The input data format is not supported.
                                        4. The input shape is not supported and does not meet the length requirements.
  361001 (ACLNN_ERR_RUNTIME_ERROR): 1. The current platform is not supported.
  ```

## aclnnFusedLinearCrossEntropyLossGrad

- **Parameters:**
  
  - **workspace** (void*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize**.
  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.
  - **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic description:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnFusedLinearCrossEntropyLossGrad** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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

    // 6. Release aclTensor. Modify the code based on the API definition.
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
