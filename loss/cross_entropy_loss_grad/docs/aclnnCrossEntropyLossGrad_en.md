# aclnnCrossEntropyLossGrad

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/cross_entropy_loss_grad)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- Description: Computes the backward pass of **aclnnCrossEntropyLoss**.
- Formula:

$$
ignoreMask_{target(t)}=\begin{cases}
1, &target(t) ≠ ignoreIndex \\
0, &target(t) = ignoreIndex
\end{cases}
$$

$$
smoothLossGrad=\begin{cases}
grad / sum(weight_{target}* ignoreMask) * labelSmoothing / C, &reduction = mean \\
grad * labelSmoothing / C, &reduction = sum \\
grad * labelSmoothing / C, &reduction = none
\end{cases}
$$

$$
lossOutGrad=\begin{cases}
grad * (1-labelSmoothing) / sum(weight_{target}* ignoreMask) * ignoreMask, &reduction = mean \\
grad * (1-labelSmoothing) * ignoreMask, &reduction = sum \\
grad * (1-labelSmoothing) * ignoreMask, &reduction = none
\end{cases}
$$

$$
nllLossGrad = lossOutGrad * weight_{target}
$$

$$
logSoftmaxGradLossOutSubPart = exp(logProb) * nllLossGrad
$$

$$
predictionsGradLossOut_{ij}=\begin{cases}
nllLossGrad_i, & j=target(i)  \\
0, & j ≠ target(i) 
\end{cases}
$$

$$
predictionsGradLossOut = logSoftmaxGradLossOutSubPart - predictionsGradLossOut
$$

$$
smoothLossGrad = smoothLossGrad * ignoreMask
$$

$$
logSoftmaxGradSmoothLoss = smoothLossGrad * weight
$$

$$
predictionsGradSmoothLoss = exp(logProb) * sum(logSoftmaxGradSmoothLoss) - logSoftmaxGradSmoothLoss
$$

Non-**zloss** scenario:

$$
xGrad_{out} = predictionsGradLossOut + predictionsGradSmoothLoss
$$

**zloss** scenario:

$$
gradZ=\begin{cases}
grad + gradZloss, & if gradZloss is provided  \\
grad, & if gradZloss is not provided
\end{cases}
$$

$$
zlossGrad=\begin{cases}
gradZ / sum(ignoreMask), & &reduction = mean  \\
gradZ, & &reduction = sum \\
gradZ, & &reduction = none
\end{cases}
$$

$$
lseGrad = 2 * lseSquareScaleForZloss * lseForZloss * ignoreMask * zlossGrad
$$

$$
zlossOutputGrad = exp(logProb) * lseGrad
$$

With **zloss** enabled, the output gradient is accumulated:

$$
xGrad_{out} = xGrad_{out} + zlossOutputGrad
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnCrossEntropyLossGradGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnCrossEntropyLossGrad** is called to perform computation.

```Cpp
aclnnStatus aclnnCrossEntropyLossGradGetWorkspaceSize(
 const aclTensor *gradLoss,
 const aclTensor *logProb,
 const aclTensor *target,
 const aclTensor *weightOptional,
 const aclTensor *gradZlossOptional,
 const aclTensor *lseForZlossOptional,
 char            *reductionOptional,
 int64_t          ignoreIndex,
 double           labelSmoothing,
 double           lseSquareScaleForZloss,
 const aclTensor *out,
 uint64_t        *workspaceSize,
 aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCrossEntropyLossGrad(
 void          *workspace,
 uint64_t       workspaceSize,
 aclOpExecutor *executor,
 aclrtStream    stream)
```

## aclnnCrossEntropyLossGradGetWorkspaceSize

- **Parameters**
  
    <table style="undefined;table-layout: fixed; width: 1546px"><colgroup>
    <col style="width: 214px">
    <col style="width: 123px">
    <col style="width: 307px">
    <col style="width: 276px">
    <col style="width: 217px">
    <col style="width: 120px">
    <col style="width: 140px">
    <col style="width: 149px">
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
        <td>gradLoss</td>
        <td>Input</td>
        <td>Gradient of the forward output loss, corresponding to grad in the formula.</td>
        <td><ul><li>When reductionOptional is none, this parameter must be a 1D tensor. </li><li>When reductionOptional is mean or sum, this parameter must be a 0D tensor.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N,)<br>N indicates the batch size.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logProb</td>
        <td>Input</td>
        <td>Log-softmax result from the forward pass, which must be a 2D tensor.</td>
        <td>-</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N, C)<br>C indicates the number of labels. The value must be greater than 0.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>target</td>
        <td>Input</td>
        <td>Class index, which must be a 1D tensor.</td>
        <td>The value range is [0, C).</td>
        <td>INT64</td>
        <td>ND</td>
        <td>(N,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightOptional</td>
        <td>Input</td>
        <td>Optional input, whose shape must be a 1D tensor.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>(C,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>gradZlossOptional</td>
        <td>Input</td>
        <td>Optional input, corresponding to gradZloss in the formula. This zloss-related input is used in the backward pass only when the forward pass generates an additional zloss output.</td>
        <td><ul><li>When reductionOptional is none, this parameter must be a 1D tensor. </li><li>When reductionOptional is mean or sum, this parameter must be a 0D tensor. </li><li>This parameter is currently not supported.</li></ul></td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>lseForZlossOptional</td>
        <td>Input</td>
        <td>Optional zloss-related input. The intermediate lse_for_zloss tensor, additionally output by the forward pass when lse_square_scale_for_zloss is non-zero, is used for LSE computation in the backward pass.</td>
        <td><ul><li>This parameter must be a 1D tensor. </li><li>This parameter is currently not supported.</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(N,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>reductionOptional</td>
        <td>Input</td>
        <td>Reduction to be applied to the output.</td>
        <td><ul><li>none: No reduction is applied. </li><li>mean: The weighted average of the output is computed. </li><li>sum: The output is summed.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignoreIndex</td>
        <td>Input</td>
        <td>Index of a target label to be ignored, which does not contribute to the input gradient.</td>
        <td>The value must be less than C. Values less than 0 indicate no label is ignored.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>labelSmoothing</td>
        <td>Input</td>
        <td>Smoothing factor for loss computation, which must be a floating-point number in the range [0.0, 1.0]. 0.0 means no smoothing is applied.</td>
        <td>Currently, only 0.0 is supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>lseSquareScaleForZloss</td>
        <td>Input</td>
        <td>zloss-related attribute. If the value is 0.0, the native PyTorch branch is used. If the value is not 0.0, the new zloss branch is used.</td>
        <td>This parameter is not supported currently.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Gradient computation result, which must be a 2D tensor.</td>
        <td>-</td>
        <td>Same as gradLoss.</td>
        <td>ND</td>
        <td>(N,C)</td>
        <td>-</td>
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
        <td>Operator executor, containing the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Returns**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 287px">
  <col style="width: 125px">
  <col style="width: 740px">
  </colgroup>
  <thead>
    <tr>
    <th>Return</th>
    <th>Error Code</th>
    <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>The passed gradLoss, logProb, target, or out is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type of gradLoss, logProb, target, weightOptional, gradZlossOptional, or lseForZlossOptional is not supported.</td>
    <tr>
    <td>ACLNN_ERR_INNER_TILING_ERROR</td>
    <td>561002</td>
    <td>The shape of logProb, target, or weightOptional is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnCrossEntropyLossGrad

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
  <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnCrossEntropyLossGradGetWorkspaceSize.</td>
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

  - **target** only supports class label indices; probabilistic inputs are not supported.
  - The data types of **gradLoss**, **logProb**, **gradZlossOptional**, **lseForZlossOptional**, and **xGradOut** must be the same.
  - The **zloss**-related features are currently not supported. Parameters **gradZlossOptional**, **lseForZlossOptional**, and **lseSquareScaleForZloss** are ignored even if provided.

  - Deterministic computation:
    - **aclnnCrossEntropyLossGrad** defaults to a deterministic implementation.
    
## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cross_entropy_loss_grad.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Customize error handling based on your requirements.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> gradLossShape = {};
  std::vector<int64_t> logProbShape = {2, 3};
  std::vector<int64_t> targetShape = {2,};
  std::vector<int64_t> weightShape = {3,};
  std::vector<int64_t> xGradShape = {2, 3};
  void* gradLossDeviceAddr = nullptr;
  void* logProbDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* xGradOutDeviceAddr = nullptr;
  aclTensor* gradLoss = nullptr;
  aclTensor* logProb = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* gradZloss = nullptr;
  aclTensor* lseForZloss = nullptr;
  aclTensor* xGradOut = nullptr;
  std::vector<float> gradLossHostData = {0.1};
  std::vector<float> logProbHostData = {-0.2, -0.2, -0.2, -0.2, -0.2, -0.2};
  std::vector<float> targetHostData = {0, 0};
  std::vector<float> weightHostData = {1.0, 1.0, 1.0};
  std::vector<float> xGradOutHostData = {-0.0091, 0.0409, 0.0409, -0.0091, 0.0409, 0.0409};
  int64_t ignoreIndex = -100;
  float labelSmoothing = 0.0;
  float lseSquareScaleForZloss = 0.0;

  // Create a gradLoss aclTensor.
  ret = CreateAclTensor(gradLossHostData, gradLossShape, &gradLossDeviceAddr, aclDataType::ACL_BF16, &gradLoss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a logProb aclTensor.
  ret = CreateAclTensor(logProbHostData, logProbShape, &logProbDeviceAddr, aclDataType::ACL_BF16, &logProb);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an xGradOut aclTensor.
  ret = CreateAclTensor(xGradOutHostData, xGradShape, &xGradOutDeviceAddr, aclDataType::ACL_BF16, &xGradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 3. Call the CANN operator library API. Modify the API as required.
  // Call the first-phase API of aclnnCrossEntropyLossGrad.
  ret = aclnnCrossEntropyLossGradGetWorkspaceSize(gradLoss, logProb, target, weight, gradZloss, lseForZloss, "mean", ignoreIndex, labelSmoothing, lseSquareScaleForZloss, xGradOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCrossEntropyLossGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnCrossEntropyLossGrad.
  ret = aclnnCrossEntropyLossGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCrossEntropyLossGrad failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(xGradShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), xGradOutDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(gradLoss);
  aclDestroyTensor(logProb);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(xGradOut);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(gradLossDeviceAddr);
  aclrtFree(logProbDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(xGradOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
