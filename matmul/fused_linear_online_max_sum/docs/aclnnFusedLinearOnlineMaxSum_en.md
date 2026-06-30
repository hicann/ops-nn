# aclnnFusedLinearOnlineMaxSum

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/fused_linear_online_max_sum)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description:

  The function is equivalent to the implementation of Megatron's matmul and fused\_vocab\_parallel\_cross\_entropy. It supports splitting and fusion of matmul and celoss in the vocabulary\_size dimension. Based on the communication, the intermediate operations require [aclnnFusedLinearOnlineMaxSum](./aclnnFusedLinearOnlineMaxSum_en.md) and [aclnnFusedCrossEntropyLossWithMaxSum](../../../loss/fused_cross_entropy_loss_with_max_sum/docs/aclnnFusedCrossEntropyLossWithMaxSum_en.md), which need to be called in sequence for complete function implemention.

- Formula:

  1. Obtain the result of the matrix multiplication of $input$ and $weight^T$.

     $$
     vocabParallelLogitsOutOptional = input @ weight^T
     $$
     
  2. Calculate the maximum value of each row in $vocabParallelLogitsOutOptional$.

     $$
     logitsMaxLocalOut = max(vocabParallelLogitsOutOptional, dim=-1)
     $$
     
  3. Calculate the difference between $vocabParallelLogitsOutOptional$ and $logitsMaxLocalOut$.

     $$
     subRes[b][n] = vocabParallelLogitsOutOptional[b][n] - logitsMaxLocalOut[b]
     $$

  4. Calculate the sum of each row after the exponential operation on $subRes$.

     $$
     sumExpLogitsLocalOut = sum(exp(subRes), dim=-1)
     $$

  5. Calculate the mask where $target$ is less than $vocabStartIndex$ or $target$ is greater than $vocabEndIndex$.

     $$
     targetMask = (target < vocabStartIndex) | (target > vocabEndIndex)
     $$

  6. Calculate $maskedTargetOut$.

     $$
     maskedTargetOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     target[b] - vocabStartIndex & \text{targetMask[b]=false}
     \end{cases}
     $$

  7. Calculate $predictedLogitsLocalOut$.

     $$
     predictedLogitsLocalOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     subRes[b][maskedTargetOut[b]] & \text{targetMask[b]=false}
     \end{cases}
     $$

  8. Calculate $targetMaskOut$.
  
     $$
     alignNum = (input.size(0) + 7) / 8 * 8\\
     maskBit[p] = \begin{cases}
     uint8(targetMask[p]) & \text{p < input.size(0)}\\
     1 & \text{input.size(0) <= p < alignNum}
     \end{cases} \\
     targetMaskOut[k] = 0b(maskBit[8*k:8*k+8])
     $$

  In the preceding information, $0 \le b \lt input.size(0), 0 \le n \lt weight.size(0), 0 \le p \lt alignNum, and 0 \le k \lt alignNum / 8$.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnFusedLinearOnlineMaxSumGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnFusedLinearOnlineMaxSum** is called to perform computation.

```Cpp
aclnnStatus aclnnFusedLinearOnlineMaxSumGetWorkspaceSize(
  const aclTensor* input,
  const aclTensor* weight,
  const aclTensor* target,
  int64_t          vocabStartIndex,
  int64_t          vocabEndIndex,
  aclTensor*       logitsMaxLocalOut,
  aclTensor*       sumExpLogitsLocalOut,
  aclTensor*       predictedLogitsLocalOut,
  aclTensor*       targetMaskOut,
  aclTensor*       maskedTargetOut,
  aclTensor*       vocabParallelLogitsOutOptional,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnFusedLinearOnlineMaxSum(
  void*           workspace,
  uint64_t        workspaceSize,
  aclOpExecutor*  executor,
  aclrtStream     stream)
```

## aclnnFusedLinearOnlineMaxSumGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1515px"><colgroup>
  <col style="width: 258px">
  <col style="width: 123px">
  <col style="width: 282px">
  <col style="width: 283px">
  <col style="width: 159px">
  <col style="width: 122px">
  <col style="width: 141px">
  <col style="width: 147px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
   <tbody>
      <tr>
      <td>input</td>
      <td>Input</td>
      <td>Left matrix of matmul computation, corresponding to input in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The value of input.size(1) must be less than or equal to 65534.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Right matrix of matmul computation, corresponding to weight in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of input. </li><li>The value of weight.size(0) must be greater than 0. </li><li>The value of weight.size(1) must be the same as that of input.size(1).</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>target</td>
      <td>Input</td>
      <td>Target index, corresponding to target in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The value of target.size(0) must be the same as that of input.size(0).</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>vocabStartIndex</td>
      <td>Input</td>
      <td>Start index allocated to the current card, corresponding to vocabStartIndex in the formula.</td>
      <td><ul><li>The value range is [0, max(target) – 1].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>vocabEndIndex</td>
      <td>Input</td>
      <td>End index allocated to the current card, corresponding to vocabEndIndex in the formula.</td>
      <td><ul><li>The value range is [vocabStartIndex, min(vocabStartIndex + weight.size(0) – 1, max(target) – 1)].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>logitsMaxLocalOut</td>
      <td>Output</td>
      <td>Maximum value of each row after matmul computation, corresponding to logitsMaxLocalOut in the formula.</td>
      <td><ul><li>The value of logitsMaxLocalOut.size(0) must be the same as that of input.size(0).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>sumExpLogitsLocalOut</td>
      <td>Output</td>
      <td>Sum of the exp result of the difference between the matmul computation result and the maximum value of each row, corresponding to sumExpLogitsLocalOut in the formula.</td>
      <td><ul><li>The value of sumExpLogitsLocalOut.size(0) must be the same as that of input.size(0).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>predictedLogitsLocalOut</td>
      <td>Output</td>
      <td>Result filtered by maskedTargetOut after the difference between the matmul computation result and the maximum value of each row is calculated, corresponding to predictedLogitsLocalOut in the formula.</td>
      <td><ul><li>The value of predictedLogitsLocalOut.size(0) must be the same as that of input.size(0).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>targetMaskOut</td>
      <td>Output</td>
      <td>Mask used to filter the vocabulary, corresponding to targetMaskOut in the formula.</td>
      <td><ul><li>The shape is [(input.size (0) + 7)/8].</li></ul></td>
      <td>UINT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>maskedTargetOut</td>
      <td>Output</td>
      <td>Result after target is filtered by targetMaskOut, corresponding to maskedTargetOut in the formula.</td>
      <td><ul><li>The data type must be the same as that of target. </li><li>The value of maskedTargetOut.size(0) must be the same as that of input.size(0).</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>vocabParallelLogitsOutOptional</td>
      <td>Output</td>
      <td>(Optional) matmul computation result, corresponding to vocabParallelLogitsOutOptional in the formula.</td>
      <td><ul><li>The data type must be the same as that of input. </li><li>The shape is [input.size (0), weight.size (0)]. </li><li>When vocabParallelLogitsOutOptional is set to nullptr, the selection for saving video memory is used. Otherwise, the high-performance selection is used.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
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
  </tbody>
  </table>

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 283px">
  <col style="width: 120px">
  <col style="width: 747px">
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
      <td>The passed input, weight, target, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut, or maskedTargetOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data type of input, weight, target, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut or maskedTargetOut is not supported.</td>
    </tr>
    <tr>
      <td>vocabParallelLogitsOutOptional is not empty and the data type of vocabParallelLogitsOutOptional is not supported.</td>
    </tr>
    <tr>
      <td>The data format of input, weight, target, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut or maskedTargetOut is not supported.</td>
    </tr>
    <tr>
      <td>vocabParallelLogitsOutOptional is not empty and the data format of vocabParallelLogitsOutOptional is not supported.</td>
    </tr>
    <tr>
      <td>The shape of input, weight, target, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut, or maskedTargetOut does not meet the constraints.</td>
    </tr>
    <tr>
      <td>vocabParallelLogitsOutOptional is not empty and the shape of vocabParallelLogitsOutOptional does not meet the constraints.</td>
    </tr>
    <tr>
      <td>The data types of input and weight are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of target and maskedTargetOut are inconsistent.</td>
    </tr>
    <tr>
      <td>vocabParallelLogitsOutOptional is not a null pointer, and the data type of vocabParallelLogitsOutOptional is inconsistent with that of input.</td>
    </tr>
    <tr>
      <td>The value of vocabStartIndex is less than 0.</td>
    </tr>
    <tr>
      <td>The value of vocabEndIndex is less than that of vocabStartIndex.</td>
    </tr>
  </tbody></table>

## aclnnFusedLinearOnlineMaxSum

- **Parameters:**

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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnFusedLinearOnlineMaxSumGetWorkspaceSize.</td>
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

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic description:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnFusedLinearOnlineMaxSum** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_linear_online_max_sum.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. Construct the input and output based on the API definition.
  int64_t mSize = 128;
  int64_t kSize = 64;
  int64_t nSize = 256;
  std::vector<int64_t> inputShape = {mSize, kSize};
  std::vector<int64_t> weightShape = {nSize, kSize};
  std::vector<int64_t> targetShape = {mSize};
  std::vector<int64_t> logitsMaxLocalOutShape = {mSize};
  std::vector<int64_t> sumExpLogitsLocalOutShape = {mSize};
  std::vector<int64_t> predictedLogitsLocalOutShape = {mSize};
  std::vector<int64_t> targetMaskOutShape = {(mSize + 7) / 8};
  std::vector<int64_t> maskedTargetOutShape = {mSize};
  std::vector<int64_t> vocabParallelLogitsOutOptionalShape = {mSize, nSize};
  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* tatgetDeviceAddr = nullptr;
  void* logitsMaxLocalOutDeviceAddr = nullptr;
  void* sumExpLogitsLocalOutDeviceAddr = nullptr;
  void* predictedLogitsLocalOutDeviceAddr = nullptr;
  void* targetMaskOutDeviceAddr = nullptr;
  void* maskedTargetOutDeviceAddr = nullptr;
  void* vocabParallelLogitsOutOptionalDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* target = nullptr;
  aclTensor* logitsMaxLocalOut = nullptr;
  aclTensor* sumExpLogitsLocalOut = nullptr;
  aclTensor* predictedLogitsLocalOut = nullptr;
  aclTensor* targetMaskOut = nullptr;
  aclTensor* maskedTargetOut = nullptr;
  aclTensor* vocabParallelLogitsOutOptional = nullptr;
  std::vector<op::fp16_t> inputHostData(mSize * kSize, 1.0);
  std::vector<op::fp16_t> weightHostData(nSize * kSize, 1.0);
  std::vector<int32_t> targetHostData(mSize, 1);
  std::vector<float> logitsMaxLocalOutHostData(mSize, 0);
  std::vector<float> sumExpLogitsLocalOutHostData(mSize, 0);
  std::vector<float> predictedLogitsLocalOutHostData(mSize, 0);
  std::vector<uint8_t> targetMaskOutHostData((mSize + 7) / 8, 0);
  std::vector<int32_t> maskedTargetOutHostData(mSize, 0);
  std::vector<op::fp16_t> vocabParallelLogitsOutOptionalHostData(mSize * nSize, 0);
  int64_t vocabStartIndex = 0;
  int64_t vocabEndIndex = 64;
  // Create an input aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &tatgetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create logitsMaxLocalOut aclTensor.
  ret = CreateAclTensor(logitsMaxLocalOutHostData, logitsMaxLocalOutShape, &logitsMaxLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &logitsMaxLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a sumExpLogitsLocalOut aclTensor.
  ret = CreateAclTensor(sumExpLogitsLocalOutHostData, sumExpLogitsLocalOutShape, &sumExpLogitsLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &sumExpLogitsLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a predictedLogitsLocalOut aclTensor.
  ret = CreateAclTensor(predictedLogitsLocalOutHostData, predictedLogitsLocalOutShape, &predictedLogitsLocalOutDeviceAddr, aclDataType::ACL_FLOAT, &predictedLogitsLocalOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a targetMaskOut aclTensor.
  ret = CreateAclTensor(targetMaskOutHostData, targetMaskOutShape, &targetMaskOutDeviceAddr, aclDataType::ACL_UINT8, &targetMaskOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a maskedTargetOut aclTensor.
  ret = CreateAclTensor(maskedTargetOutHostData, maskedTargetOutShape, &maskedTargetOutDeviceAddr, aclDataType::ACL_INT32, &maskedTargetOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a vocabParallelLogitsOutOptional aclTensor.
  ret = CreateAclTensor(vocabParallelLogitsOutOptionalHostData, vocabParallelLogitsOutOptionalShape, &vocabParallelLogitsOutOptionalDeviceAddr, aclDataType::ACL_FLOAT16, &vocabParallelLogitsOutOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnFusedLinearOnlineMaxSum.
  ret = aclnnFusedLinearOnlineMaxSumGetWorkspaceSize(input, weight, target, vocabStartIndex, vocabEndIndex, logitsMaxLocalOut, sumExpLogitsLocalOut, predictedLogitsLocalOut, targetMaskOut, maskedTargetOut, vocabParallelLogitsOutOptional, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearOnlineMaxSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnFusedLinearOnlineMaxSum.
  ret = aclnnFusedLinearOnlineMaxSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearOnlineMaxSum failed. ERROR: %d\n", ret); return ret);
  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(logitsMaxLocalOutShape);
  std::vector<float> logitsMaxLocalOutResultData(size, 0);
  ret = aclrtMemcpy(logitsMaxLocalOutResultData.data(), logitsMaxLocalOutResultData.size() * sizeof(logitsMaxLocalOutResultData[0]), logitsMaxLocalOutDeviceAddr, size * sizeof(logitsMaxLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("logitsMaxLocalOut[%ld] is: %f\n", i, logitsMaxLocalOutResultData[i]);
  }

  size = GetShapeSize(sumExpLogitsLocalOutShape);
  std::vector<float> sumExpLogitsLocalOutResultData(size, 0);
  ret = aclrtMemcpy(sumExpLogitsLocalOutResultData.data(), sumExpLogitsLocalOutResultData.size() * sizeof(sumExpLogitsLocalOutResultData[0]), sumExpLogitsLocalOutDeviceAddr, size * sizeof(sumExpLogitsLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("sumExpLogitsLocalOut[%ld] is: %f\n", i, sumExpLogitsLocalOutResultData[i]);
  }

  size = GetShapeSize(predictedLogitsLocalOutShape);
  std::vector<float> predictedLogitsLocalOutResultData(size, 0);
  ret = aclrtMemcpy(predictedLogitsLocalOutResultData.data(), predictedLogitsLocalOutResultData.size() * sizeof(predictedLogitsLocalOutResultData[0]), predictedLogitsLocalOutDeviceAddr, size * sizeof(predictedLogitsLocalOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("predictedLogitsLocalOut[%ld] is: %f\n", i, predictedLogitsLocalOutResultData[i]);
  }

  size = GetShapeSize(targetMaskOutShape);
  std::vector<uint8_t> targetMaskOutResultData(size, 0);
  ret = aclrtMemcpy(targetMaskOutResultData.data(), targetMaskOutResultData.size() * sizeof(targetMaskOutResultData[0]), targetMaskOutDeviceAddr, size * sizeof(targetMaskOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("targetMaskOut[%ld] is: %hhu\n", i, targetMaskOutResultData[i]);
  }

  size = GetShapeSize(maskedTargetOutShape);
  std::vector<int32_t> maskedTargetOutResultData(size, 0);
  ret = aclrtMemcpy(maskedTargetOutResultData.data(), maskedTargetOutResultData.size() * sizeof(maskedTargetOutResultData[0]), maskedTargetOutDeviceAddr, size * sizeof(maskedTargetOutResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("maskedTargetOut[%ld] is: %d\n", i, maskedTargetOutResultData[i]);
  }

  size = GetShapeSize(vocabParallelLogitsOutOptionalShape);
  std::vector<op::fp16_t> vocabParallelLogitsOutOptionalResultData(size, 0);
  ret = aclrtMemcpy(vocabParallelLogitsOutOptionalResultData.data(), vocabParallelLogitsOutOptionalResultData.size() * sizeof(vocabParallelLogitsOutOptionalResultData[0]), vocabParallelLogitsOutOptionalDeviceAddr, size * sizeof(vocabParallelLogitsOutOptionalResultData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("vocabParallelLogitsOutOptional[%ld] is: %f\n", i, static_cast<float>(vocabParallelLogitsOutOptionalResultData[i]));
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(weight);
  aclDestroyTensor(target);
  aclDestroyTensor(logitsMaxLocalOut);
  aclDestroyTensor(sumExpLogitsLocalOut);
  aclDestroyTensor(predictedLogitsLocalOut);
  aclDestroyTensor(targetMaskOut);
  aclDestroyTensor(maskedTargetOut);
  aclDestroyTensor(vocabParallelLogitsOutOptional);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(inputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(tatgetDeviceAddr);
  aclrtFree(logitsMaxLocalOutDeviceAddr);
  aclrtFree(sumExpLogitsLocalOutDeviceAddr);
  aclrtFree(predictedLogitsLocalOutDeviceAddr);
  aclrtFree(targetMaskOutDeviceAddr);
  aclrtFree(maskedTargetOutDeviceAddr);
  aclrtFree(vocabParallelLogitsOutOptionalDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
