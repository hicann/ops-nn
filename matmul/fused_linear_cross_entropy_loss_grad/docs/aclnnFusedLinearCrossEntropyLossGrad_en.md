# aclnnFusedLinearCrossEntropyLossGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:02:46.981Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: This operator is part of the cross-entropy loss computation module in vocabulary parallelism scenarios, addressing memory and computational efficiency issues for extremely large vocabularies. The current part implements gradient computation, used to compute the gradients of the leaf nodes `input` and `weight`.
  The relevant outputs of `aclnnFusedLinearOnlineMaxSum` and `aclnnFusedCrossEntropyLossWithMaxSum`, as well as the global communication results related to `logits`, are required as inputs of this API.
- Formula:

&emsp;&emsp;High-performance mode, softmaxOptional is not nullptr:

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
\text{grad\_input} = \text{softmax} \cdot \text{weight} \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{softmax}^T \cdot \text{input} \in \mathbb{R}^{V \times H}
$$

<br>
&emsp;&emsp;Memory-saving mode, where softmaxOptional is nullptr:

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

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize` API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the `aclnnFusedLinearCrossEntropyLossGrad` API to perform the computation.

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
        <td>grad</td>
        <td>Input</td>
        <td>Gradient of the current node, corresponding to input grad in the formula.</td>
        <td>Supports empty tensors.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>input</td>
        <td>Input</td>
        <td>Input matrix of matrix multiplication, corresponding to input input in the formula.</td>
        <td>The first dimension has the same length as input grad. Empty tensors are supported.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Weight matrix for matrix multiplication, corresponding to weight in the formula.</td>
        <td>Data type is the same as input. Shape is [V, H], with its second dimension equal to that of input. Only high-performance mode (softmaxOptional is not nullptr) requires V >= 128; memory-saving mode has no such lower bound.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>targetMask</td>
        <td>Input</td>
        <td>Intermediate variable, target_mask in the formula. A value of 1 means the row's target is outside the current vocabulary shard and skips the one-hot subtraction; 0 subtracts 1 at the local index in maskedTarget. Masking does not skip subsequent gradient scaling or matrix multiplication.</td>
        <td><ul><li>BOOL: one element per row; length must equal BT (the length of grad).</li><li>UINT8: bit-packed; length multiplied by 8 must be at least BT. Row i uses bit (i % 8) of targetMask[i / 8], unpacked least significant bit (LSB) first.</li></ul></td>
        <td>BOOL, UINT8</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maskedTarget</td>
        <td>Input</td>
        <td>Intermediate variable representing the local index of the corresponding token ID in the vocabulary shard on the current device, masked_target in the formula.</td>
        <td><ul><li>Length must equal the length of grad.</li><li>For each row whose unpacked targetMask is 0, 0 &lt;= maskedTarget[i] &lt; V is required. Rows with mask 1 skip the one-hot subtraction without using their target index.</li><li>Out-of-range indices are skipped defensively by the kernel, never clamped. This does not make them normal targets: callers must still supply valid indices for unmasked rows.</li></ul></td>
        <td>INT64, INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>labelSmoothing</td>
        <td>Input</td>
        <td>Label smoothing coefficient, used to mitigate overfitting.</td>
        <td>Currently, only the value 0 is supported.</td>
        <td>FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logitsMaxOptional</td>
        <td>Optional input</td>
        <td>Intermediate variable, maximum value of global logits, corresponding to logits_max in the formula.</td>
        <td><ul><li>Supports nullptr input. When nullptr is used, a valid softmaxOptional input must be provided.</li><li>The shape length is the same as that of the input grad.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sumExpLogitsOptional</td>
        <td>Optional input</td>
        <td>Intermediate variable, processed logits, namely sum_exp_logits in the formula.</td>
        <td><ul><li>Supports nullptr input. When nullptr is used, a valid softmaxOptional input must be provided.</li><li>The shape length is the same as that of the input grad.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxOptional</td>
        <td>Optional input</td>
        <td>Intermediate variable, the result of matrix multiplication, that is, softmax in the formula.</td>
        <td><ul><li>Supports input of nullptr. If nullptr is input, valid logitsMaxOptional and sumExpLogitsOptional inputs must be provided; if the input is not nullptr, the logitsMaxOptional and sumExpLogitsOptional inputs are invalid.</li><li>The length of the first dimension of the shape is the same as the length of the input grad, and the length of the second dimension is the same as the length of the first dimension of the input weight.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>inputGradOut</td>
        <td>Output</td>
        <td>Gradient of the leaf node input, corresponding to grad_input in the formula.</td>
        <td><ul><li>Data type: same as the input input.</li><li>Shape: the length of the first dimension is the same as the length of the input grad, and the length of the second dimension is the same as the second dimension of the input weight.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightGradOut</td>
        <td>Output</td>
        <td>Gradient corresponding to the leaf node weight, that is, grad_weight in the formula.</td>
        <td><ul><li>The data type is the same as input.</li><li>Shape is [V, H]: its first dimension equals the first dimension of weight, and its second dimension equals the second dimension of input.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>The passed-in non-optional input is a null pointer.</td>
    </tr>
    <tr>
      <td>In the use case where the input softmaxOptional is a null pointer, logitsMaxOptional or sumExpLogitsOptional is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The input labelSmoothing value is not supported.</td>
    </tr>
    <tr>
      <td>The input data type is out of the supported range.</td>
    </tr>
    <tr>
      <td>The input data format is out of the supported range.</td>
    </tr>
    <tr>
      <td>The input shape is out of the supported range and does not meet the length requirements.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is out of the supported platform range.</td>
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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Let BT be the length of grad, H the second dimension of input, and V the first dimension of weight. The output shapes are [BT, H] for inputGradOut and [V, H] for weightGradOut.
- For non-empty computation, both modes require H <= 65535.
- High-performance mode (softmaxOptional is not nullptr) additionally requires BT <= 65536, 128 <= V <= 65280, and sufficient UB capacity: `2 * align_up(4 * V, 64) <= UBBytes - 32`. UBBytes is the per-core device UB capacity in bytes; align_up rounds up to a multiple of 64 bytes. Thus, the effective V upper bound also depends on UB capacity. Memory-saving mode is not subject to these high-performance BT/V bounds or this UB formula.
- These dimension and UB conditions do not guarantee successful tiling; Matmul tiling and platform resource limitations still apply.
- Subject to parameter validation, if input or weight is empty, non-empty outputs are zero-filled and empty outputs are not written. No computation is scheduled when both outputs are empty. Use the actual workspaceSize returned by the first-phase API; an empty input does not necessarily imply zero workspace.
- In the formulas, target_mask denotes the unpacked per-row mask. The one-hot subtraction applies only to rows with targetMask=0 and a valid index; targetMask=1 takes precedence and skips the target update.
- Deterministic computation:
  - aclnnFusedLinearCrossEntropyLossGrad defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
std::vector<T> GenZeroVector(const std::vector<int64_t>& shape) {
    // 1. Calculate the total number of elements.
    size_t total = 1;
    for (auto dim : shape) {
        total *= dim;
    }

    // 2. Fill with 0.
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
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateEmptyAclTensor(const std::vector<int64_t> &shape, void **deviceAddr,
                         aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream. For details, see the list of external ACL APIs.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs, and customize the construction as required by the API.
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
    // Create the aclTensor.
    auto gradData = GenZeroVector<int32_t>(gradShape);
    ret = CreateAclTensor<int32_t>(gradData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto inputData = GenZeroVector<int16_t>(inputShape);
    ret = CreateAclTensor<int16_t>(inputData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto weightData = GenZeroVector<int16_t>(weightShape);
    ret = CreateAclTensor<int16_t>(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
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
    ret = CreateAclTensor<int16_t>(inputGradOutData, inputGradOutShape, &inputGradOutDeviceAddr, aclDataType::ACL_FLOAT16, &inputGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto weightGradOutData = GenZeroVector<int16_t>(weightGradOutShape);
    ret = CreateAclTensor<int16_t>(weightGradOutData, weightGradOutShape, &weightGradOutDeviceAddr, aclDataType::ACL_FLOAT16, &weightGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first-phase API of aclnnFusedLinearCrossEntropyLossGrad.
    ret = aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(grad, input, weight, targetMask, maskedTarget, labelSmoothing, logitsMaxOptional, sumExpLogitsOptional, softmaxOptional, inputGradOut, weightGradOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnFusedLinearCrossEntropyLossGrad.
    ret = aclnnFusedLinearCrossEntropyLossGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearCrossEntropyLossGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait for the task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, and copy the result from device memory to host memory. Modify it based on the specific API definition.
    // inputGradOut
    auto size = GetShapeSize(inputGradOutShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), inputGradOutDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < 16; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }

    // 6. Release the aclTensor. Modify it based on the specific API definition.
    aclDestroyTensor(grad);
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(targetMask);
    aclDestroyTensor(maskedTarget);
    aclDestroyTensor(softmaxOptional);
    aclDestroyTensor(inputGradOut);
    aclDestroyTensor(weightGradOut);

    // 7. Release Device resources. Modify the code based on the interface definition of the specific API.
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
