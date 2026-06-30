# aclnnTopKTopPSample

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/top_k_top_p_sample)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- Description:
  Performs topK-topP sampling computation based on the input **logits**, topK/topP sampling parameters, and random sampling weight distribution **q**, and outputs the maximum word frequency **logitsSelectIdx** of each batch and the word frequency distribution **logitsTopKPSelect** after topK-topP sampling.

  The operator contains three sampling algorithms that can be enabled separately with the upstream and downstream processing relationships remain unchanged (from the original input to the final output): TopK sampling, TopP sampling, and exponential sampling (referred to as Sample in this document). They can form eight compute scenarios, as shown in the following table.
  | Compute Scenario| TopK Sampling| TopP Sampling| Exponential Distribution Sampling|Remarks|
  | :-------:| :------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax Sampling|×|×|×|For each batch of input logits, obtains the maximum result after SoftMax is performed.|
  |TopK sampling|√|×|×|For each batch of input logits, obtains the topK[batch] maximum results.|
  |TopP sampling|×|√|×|For each batch of input logits, sorts the input logits in descending order, and samples the first n results whose accumulated value is greater than or equal to topP[batch].|
  |Sample sampling|×|×|√|For each batch of input logits, performs Softmax, divides the result by q, and obtains the maximum result.|
  |TopK-TopP sampling|√|√|×|For each batch of input logits, performs topK sampling and then topP sampling, and obtains the maximum result.|
  |TopK-Sample sampling|√|×|√|For each batch of input logits, performs topK sampling and then Sample sampling, and obtains the maximum result.|
  |TopP-Sample sampling|×|√|√|For each batch of input logits, performs topP sampling and then Sample sampling, and obtains the maximum result.|
  |TopK-TopP-Sample sampling|√|√|√|For each batch of input logits, performs topK sampling, topP sampling, and then Sample sampling, and obtains the maximum result.|

- Formula:
The input logits are a word frequency table with a size of [batch, voc_size], where each batch corresponds to one input sequence, and voc_size is the uniform length of each batch.<br>
Based on topK[batch], topP[batch], and q[batch,:], different computations are executed for each logits[batch][:] row in logits.<br>
In the following formulas, b and v are used to represent the indices in the batch and voc_size directions, respectively.

  TopK sampling

  1. Computes the topK for each segment based on the segment length v, applies merge sort to the topK results, pre-filters the input of the current {s} segments using the topK of the previous {s-1} segments, and gradually updates the topK of a single batch to reduce data redundancy and computation.
  2. topK[batch] indicates the k value of the current batch. The valid range is 1 ≤ topK[batch] ≤ min(voc_size[batch], 1024). If top[k] is out of the valid range, the topK sampling phase of the current batch is skipped, so does the sorting of the current batch. The input logits[batch] is directly passed to the next module.

  * Divides the current batch into several sub-segments and calculates topKValue[b] in rolling mode.

  $$
  topKValue[b] = {Max(topK[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topKValue[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge topKMin[b][s-1] \right \} \right \}\\
  Card(topKValue[b])=topK[b]
  $$

  Where:

  $$
  topKMin[b][s] = Min(topKValue[b]\left \{  s \right \})
  $$

  v indicates the preset fixed segment length during the rolling topK operation.

  $$
  v=8*1024
  $$

  * Generates the mask to be filtered.

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask[b] = sortedValue[b]<Min(topKValue[b])
  $$

  * Sets the part that is less than the threshold to -Inf using mask.

  $$
  sortedValue[b][v]=
  \begin{cases}
  -Inf & \text{topKMask[b][v]=true} \\
  sortedValue[b][v] & \text{topKMask[b][v]=false} &
  \end{cases}
  $$

  * Converts the logits filtered by topK into probability distribution along the last axis using softmax.

  $$
  probsValue[b]=sortedValue[b].softmax (dim=-1)
  $$

  * Computes the cumulative probability along the last axis (starting cumulation from the smallest probability).

  $$
  probsSum[b]=probsValue[b].cumsum (dim=-1)
  $$

  TopP sampling

  * If the previous topK sampling has a sorted output result, computes the cumulative word frequency based on the topK sampling output, and performs truncated sampling based on the topP:
    $$
    topPMask[b] = probsSum[b][*] < topP[b]
    $$

  * If topK sampling is skipped, performs softmax on the input logits[b] first:

  $$
  logitsValue[b] = logits[b].softmax(dim=-1)
  $$

  * Attempts to use topKGuess to perform rolling sorting on logits to obtain the mask for computing topP:

  $$
  topPValue[b] = {Max(topKGuess)}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topPValue[b]\left \{s-1 \right \}  \cup \left \{ logitsValue[b][v] \ge topKMin[b][s-1] \right \} \right \}
  $$

  * If the following condition is met before the 1e4th element of logitsValue[b] is accessed, topKGuess is considered successful:
  $$
  \sum^{topKGuess}(topPValue[b]) \ge topP[b]\\
  topPMask[b][Index(topPValue[b])] = false
  $$

  * If topKGuess fails, performs full sorting and cumsum on the current logitsValue[b], and performs truncated sampling based on topP[b]:

  $$
  sortedLogits[b] = sort(logitsValue[b], descendant) \\
  probsSum[b]=sortedLogits[b].cumsum (dim=-1) \\
  topPMask[b] = (probsSum[b] - sortedLogits[b])>topP[b] 
  $$

  * Sets the positions to be filtered to -Inf to obtain sortedValue[b][v]:

    $$
    sortedValue[b][v] = \begin{cases} -Inf& \text{topPMask[b][v]=true}\\sortedValue[b][v]& \text{topPMask[b][v]=false}\end{cases}
    $$

    Obtains the first topK elements in each row of the filtered sortedValue[b][v], searches for the original indices of these elements in the input, and integrates them into `logitsIdx`:

    $$
    logitsIdx[b][v] = Index(sortedValue[b][v] \in logits)
    $$

  Exponential sampling (Sample)

  * If `isNeedLogits=true`, selects the sampling result based on `logitsIdx` and outputs it to `logitsTopKPSelect`:

  $$
  logitsTopKPSelect[b][logitsIdx[b][v]]=sortedValue[b][v]
  $$

  * Performs exponential distribution sampling on `logitsSort`:

    $$
    probs = softmax(logitsSort)
    $$

    $$
    probsOpt = \frac{probs}{q + eps}
    $$
  * Obtains the maximum element of each batch from `probsOpt`, performs gather on the input indices of the corresponding elements from `logitsIdx`, and uses the gathered result as the output `logitsSelectIdx`:
    
    $$
    logitsSelectIdx[b] = logitsIdx[b][argmax(probsOpt[b][:])]
    $$

  where, 0 ≤ b < sortedValue.size(0) and 0 ≤ v < sortedValue.size(1)

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnTopKTopPSampleGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnTopKTopPSample` is called to perform computation.

```Cpp
aclnnStatus aclnnTopKTopPSampleGetWorkspaceSize(
  const aclTensor *logits, 
  const aclTensor *topK, 
  const aclTensor *topP, 
  const aclTensor *q, 
  double           eps, 
  bool             isNeedLogits, 
  int64_t          topKGuess, 
  const aclTensor *logitsSelectIdx, 
  const aclTensor *logitsTopKPSelect, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)

```

```Cpp
aclnnStatus aclnnTopKTopPSample(
  void           *workspace, 
  uint64_t        workspaceSize, 
  aclOpExecutor  *executor, 
  aclrtStream     stream)

```

## aclnnTopKTopPSampleGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
      <col style="width: 146px">
      <col style="width: 120px">
      <col style="width: 271px">
      <col style="width: 392px">
      <col style="width: 228px">
      <col style="width: 101px">
      <col style="width: 100px">
      <col style="width: 145px">
      </colgroup>
      <thead>
        <tr>
          <th>Name</th>
          <th>Input/Output</th>
          <th>Description</th>
          <th>Usage Notes</th>
          <th>Data Type</th>
          <th>Data Format</th>
          <th>Dimension (Shape)</th>
          <th>Non-contiguous Tensor</th>
        </tr></thead>
      <tbody>
      <tr>
        <td>logits</td>
        <td>Input</td>
        <td>Input word frequency to be sampled. The word frequency index is fixed to the last dimension, corresponding `logits` in the formula.</td>
        <td><ul><li>Empty tensors are not supported.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topK</td>
        <td>Input</td>
        <td>k value sampled for each batch. It corresponds to `topK[b]` in the formula.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as the first n–1 dimensions of `logits`.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topP</td>
        <td>Input</td>
        <td>p value sampled for each batch. It corresponds to `topP[b]` in the formula.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as the first n–1 dimensions of `logits`, and the data type must be the same as that of `logits`.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>q</td>
        <td>Input</td>
        <td>Exponential sampling matrix output by topK-topP sampling. It corresponds to `q` in the formula.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as that of `logits`.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>Input</td>
        <td>Coefficient for preventing division by zero in softmax and weight sampling. The recommended value is 1e-8.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
        <td>isNeedLogits</td>
        <td>Input</td>
        <td>Output condition of logitsTopKPselect. The recommended value is 0.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tr>
        <td>topKGuess</td>
        <td>Input</td>
        <td>Size of candidate logits applied when part of each batch of logits are traversed for topP sampling. The value must be a positive integer.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tr>
        <td>logitsSelectIdx</td>
        <td>Output</td>
        <td>Position index of the element with the maximum word frequency max(probsOpt[batch, :]) in each batch in the input logits after the topK-topP-sample computation process.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as the first n–1 dimensions of `logits`.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      </tr>
        <td>logitsTopKPSelect</td>
        <td>Output</td>
        <td>Remaining logits that are not filtered out from the input logits after the topK-topP computation process.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as the first n–1 dimensions of `logits`.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
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

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>The input logits, topK, or topP is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of logits, topK, topP, or q is not supported.</td>
    </tr>
    <tr>
      <td>The dimensions or sizes of logits and q are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimensions of topK or topP are inconsistent with the first n–1 dimensions of logits.</td>
    </tr>
    <tr>
      <td>The data types of logits and topP are inconsistent.</td>
    </tr>
  </tbody></table>
  
## aclnnTopKTopPSample

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnTopKTopPSampleGetWorkspaceSize.</td>
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

- Deterministic compute:
  - **aclnnTopKTopPSample** defaults to a deterministic implementation.
- For all sampling parameters, their sizes must meet the following requirements: batch > 0 and 0 < vocSize <= 2^20.
- Only non-negative values are valid inputs for topK and topP. Passing 0 or negative values will not cause the sampling to be skipped for the corresponding batch; instead, it causes unexpected errors.
- The sizes of **logits**, **q**, and **logitsTopKPselect** must be the same, so does their dimensions.
- All dimensions except the last dimension of **logits**, **topK**, **topP**, and **logitsSelectIdx** must be in the same order and have the same size. Currently, **logits** can only be two-dimensional, and **topK**, **topP**, and **logitsSelectIdx** must be one-dimensional non-empty tensors. Empty tensors cannot be used as the input of **logits**, **topK**, or **topP**. If the corresponding module needs to be skipped, set the input as required.
- To skip the topK module separately, pass a tensor of size [batch, 1] and set each element to an invalid value.
- If 1024 < topK[batch] < vocSize[batch], all valid elements in the current batch are selected and topK sampling is skipped.
- To skip the topP module separately, pass a tensor of size [batch, 1] and set each element to a value greater than or equal to 1.
- To skip the sample module separately, pass `q=nullptr`. To use the sample module, pass a tensor of size [batch, vocSize].

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_top_k_top_p_sample.h"

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
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> logitsShape = {48, 131072};
      std::vector<int64_t> topKPShape = {48};
      long long vocShapeSize = GetShapeSize(logitsShape);
      long long batchShapeSize = GetShapeSize(topKPShape);

      void* logitsDeviceAddr = nullptr;
      void* topKDeviceAddr = nullptr;
      void* topPDeviceAddr = nullptr;
      void* qDeviceAddr = nullptr;
      void* logitsSelectedIdxDeviceAddr = nullptr;
      void* logitsTopKPSelectDeviceAddr = nullptr;

      aclTensor* logits = nullptr;
      aclTensor* topK = nullptr;
      aclTensor* topP = nullptr;
      aclTensor* q = nullptr;
      aclTensor* logitsSelectedIdx = nullptr;
      aclTensor* logitsTopKPSelect = nullptr;
      std::vector<int16_t> logitsHostData(48 * 131072, 1);
      std::vector<int32_t> topKHostData(48, 128);
      std::vector<int16_t> topPHostData(48, 1);
      std::vector<float> qHostData(48 * 131072, 1.0f);

      std::vector<int64_t> logitsSelectedIdxHostData(48, 0);
      std::vector<float> logitsTopKPSelectHostData(48 * 131072, 0);

      float eps=1e-8;
      int64_t isNeedLogits=0;
      int32_t topKGuess=32;
      // Create a logitsaclTensor.
      ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr, aclDataType::ACL_BF16, &logits);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a topKaclTensor.
      ret = CreateAclTensor(topKHostData, topKPShape, &topKDeviceAddr, aclDataType::ACL_INT32, &topK);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a topPaclTensor.
      ret = CreateAclTensor(topPHostData, topKPShape, &topPDeviceAddr, aclDataType::ACL_BF16, &topP);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a q aclTensor.
      ret = CreateAclTensor(qHostData, logitsShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a logtisSelected aclTensor.
      ret = CreateAclTensor(logitsSelectedIdxHostData, topKPShape, &logitsSelectedIdxDeviceAddr, aclDataType::ACL_INT64, &logitsSelectedIdx);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a logitsTopKPSelect aclTensor.
      ret = CreateAclTensor(logitsTopKPSelectHostData, logitsShape, &logitsTopKPSelectDeviceAddr, aclDataType::ACL_FLOAT, &logitsTopKPSelect);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // Call the first-phase API of aclnnTopKTopPSample.
      ret = aclnnTopKTopPSampleGetWorkspaceSize(logits, topK, topP, q, eps, isNeedLogits, topKGuess, logitsSelectedIdx, logitsTopKPSelect, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSampleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnTopKTopPSample.
      ret = aclnnTopKTopPSample(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSample failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(topKPShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), logitsSelectedIdxDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }

      // 6. Release aclTensor. Modify the configuration based on the API definition.
      aclDestroyTensor(logits);
      aclDestroyTensor(topK);
      aclDestroyTensor(topP);
      aclDestroyTensor(q);
      aclDestroyTensor(logitsSelectedIdx);
      aclDestroyTensor(logitsTopKPSelect);
      // 7. Release device resources. Modify the configuration based on the API definition.
      aclrtFree(logitsDeviceAddr);
      aclrtFree(topKDeviceAddr);
      aclrtFree(topPDeviceAddr);
      aclrtFree(qDeviceAddr);
      aclrtFree(logitsSelectedIdxDeviceAddr);
      aclrtFree(logitsTopKPSelectDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
      }
  ```
