# aclnnEmbeddingBag

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/embedding_bag)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Retrieves a set of values from **weight** based on **indices**, then aggregates them using the specified mode (max, sum, or mean) according to the offsets defined by **offsets**. Other parameters provide finer control over the computation process.
- Shape derivation:
  Assume:

  ```
  weight shape: (numWeight, embeddingDim)
  indices shape: (bagIndices)
  offsets shape: (bagOffsets)
  ```
  
  - When **mode** is **sum**:
  
    ```
    output shape: includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    offset2bag shape: (bagIndices,)
    bagSize shape: includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    maxIndices shape: includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    ```

  - When **mode** is **mean**:

    ```
    output shape: includeLastOffset? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    offset2bag shape: (bagIndices,)
    bagSize shape: includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    maxIndices shape: includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    ```

  - When **mode** is **max**:

    ```
    output shape: includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    offset2bag shape: (bagIndices,)
    bagSize shape: includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    maxIndices shape: includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    ```

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnEmbeddingBagGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnEmbeddingBag** is called to perform computation.

```Cpp
aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(
 const aclTensor* weight,
 const aclTensor* indices,
 const aclTensor* offsets,
 bool             scaleGradByFreq,
 int64_t          mode,
 bool             sparse,
 const aclTensor* perSampleWeights,
 bool             includeLastOffset,
 int64_t          paddingIdx,
 aclTensor*       output,
 aclTensor*       offset2bag,
 aclTensor*       bagSize,
 aclTensor*       maxIndices,
 uint64_t*        workspaceSize,
 aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnEmbeddingBag(
 void*            workspace,
 uint64_t         workspaceSize,
 aclOpExecutor*   executor,
 aclrtStream      stream)`
```

## aclnnEmbeddingBagGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1391px"><colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 274px">
    <col style="width: 172px">
    <col style="width: 116px">
    <col style="width: 114px">
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
        <th>Shape</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Word embedding matrix containing embedding vectors for all words.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>Input</td>
        <td>Tensor containing indices specifying which word embedding vectors to extract from weight.</td>
        <td>-</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0–1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>offsets</td>
        <td>Input</td>
        <td>Offset tensor used to split indices into multiple bags.</td>
        <td>-</td>
        <td>INT32, INT64</td>
        <td>-</td>
        <td>0–1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleGradByFreq</td>
        <td>Input</td>
        <td>Whether to scale gradients by word frequency.</td>
        <td>true: Gradients are scaled by word frequency. false: Gradients are not scaled by word frequency.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>mode</td>
        <td>Input</td>
        <td>Aggregation mode.</td>
        <td>0: sum aggregation; 1: mean aggregation; other values: max aggregation.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparse</td>
        <td>Input</td>
        <td>Whether to enable sparse mode.</td>
        <td><ul><li>false: weight is a non-sparse matrix.</li><li>true: weight is a sparse matrix.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>perSampleWeights</td>
        <td>Input</td>
        <td>Sample-wise weights.</td>
        <td>The value can be non-nullptr only in sum mode, and must be nullptr in other modes.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>includeLastOffset</td>
        <td>Input</td>
        <td>Whether to include the last offset.</td>
        <td><ul><li>false: The last offset is not included.</li><li>true: The last offset is included.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>paddingIdx</td>
        <td>Input</td>
        <td>Indices that are ignored in the computation.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>output</td>
        <td>Output</td>
        <td>Result of the embedding aggregation.</td>
        <td>-</td>
        <td>Same as weight.</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>offset2bag</td>
        <td>Output</td>
        <td>Starting offset of each bag.</td>
        <td>The data type must be the same as the higher-precision data type between indices and offsets.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0–1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bagSize</td>
        <td>Output</td>
        <td>Size of each bag.</td>
        <td>The data type must be the same as the higher-precision data type between indices and offsets.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0–1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maxIndices</td>
        <td>Output</td>
        <td>Indices of the rows containing the maximum values in the word embedding vectors when mode is max.</td>
        <td>The data type must be the same as the higher-precision data type between indices and offsets.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>1</td>
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
        <td>Operator executor, containing the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

    - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: The data type cannot be BFLOAT16.

- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>The passed weight, indices, offsets, output, offset2bag, bagSize, or maxIndices is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type of weight is not supported, or weight does not have exactly two dimensions.</td>
      </tr>
      <tr>
      <td>The data type of indices is not supported, or indices does not have exactly one dimension.</td>
      </tr>
      <tr>
      <td>The data type of offsets is not supported, or offsets does not have exactly one dimension.</td>
      </tr>
      <tr>
      <td>The data types of indices and offsets are neither INT32 nor INT64.</td>
      </tr>
      <tr>
      <td>When perSampleWeights is not nullptr: its data type does not match weight, it does not have exactly one dimension, its element count does not match indices, or it is not nullptr in non-sum modes.</td>
      </tr>
      <tr>
      <td>The data type of output does not match weight, or its shape does not conform to the definition.</td>
      </tr>
      <tr>
      <td>The data type or shape of offset2bag, bagSize, or maxIndices does not match the inferred type or shape.</td>
      </tr>
    </tbody>
    </table>

## aclnnEmbeddingBag

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnEmbeddingBagGetWorkspaceSize.</td>
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
  - **aclnnEmbeddingBag** defaults to a deterministic implementation.

`sparse` and `scaleGradByFreq` support only `False`.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_embedding_bag.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> weightShape = {3, 3};
  std::vector<int64_t> indicesShape = {6};
  std::vector<int64_t> offsetsShape = {4};
  std::vector<int64_t> perSampleWeightsShape = {6};
  std::vector<int64_t> outputShape = {4, 3};
  std::vector<int64_t> offset2bagShape = {6};
  std::vector<int64_t> bagSizeShape = {4};
  std::vector<int64_t> maxIndicesShape = {4};

  void* weightDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* offsetsDeviceAddr = nullptr;
  void* perSampleWeightsDeviceAddr = nullptr;
  void* outputDeviceAddr = nullptr;
  void* offset2bagDeviceAddr = nullptr;
  void* bagSizeDeviceAddr = nullptr;
  void* maxIndicesDeviceAddr = nullptr;

  aclTensor* weight = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* offsets = nullptr;
  aclTensor* perSampleWeights = nullptr;
  aclTensor* output = nullptr;
  aclTensor* offset2bag = nullptr;
  aclTensor* bagSize = nullptr;
  aclTensor* maxIndices = nullptr;

  std::vector<float> weightHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> indicesHostData = {1, 2, 0, 2, 2, 1};
  std::vector<int64_t> offsetsHostData = {0, 2, 4, 5};
  std::vector<float> perSampleWeightsHostData = {1, 1, 1, 1, 1, 1};
  std::vector<float> outputHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int64_t> offset2bagHostData = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> bagSizeHostData = {0, 0, 0, 0};
  std::vector<int64_t> maxIndicesHostData = {0, 0, 0, 0};

  // Create a weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an indices aclTensor.
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an offsets aclTensor.
  ret = CreateAclTensor(offsetsHostData, offsetsShape, &offsetsDeviceAddr, aclDataType::ACL_INT64, &offsets);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a perSampleWeights aclTensor.
  ret = CreateAclTensor(perSampleWeightsHostData, perSampleWeightsShape, &perSampleWeightsDeviceAddr, aclDataType::ACL_FLOAT, &perSampleWeights);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an output aclTensor.
  ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an offset2bag aclTensor.
  ret = CreateAclTensor(offset2bagHostData, offset2bagShape, &offset2bagDeviceAddr, aclDataType::ACL_INT64, &offset2bag);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a bagSize aclTensor.
  ret = CreateAclTensor(bagSizeHostData, bagSizeShape, &bagSizeDeviceAddr, aclDataType::ACL_INT64, &bagSize);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a maxIndices aclTensor.
  ret = CreateAclTensor(maxIndicesHostData, maxIndicesShape, &maxIndicesDeviceAddr, aclDataType::ACL_INT64, &maxIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // Non-tensor parameters
  bool scaleGradByFreq = false;
  int64_t mode = 0;
  bool sparse = false;
  bool includeLastOffset = false;
  int64_t paddingIdx = 1;

  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnEmbeddingBag.
  ret = aclnnEmbeddingBagGetWorkspaceSize(weight, indices, offsets, scaleGradByFreq, mode, sparse, perSampleWeights,
            includeLastOffset, paddingIdx, output, offset2bag, bagSize, maxIndices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingBagGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnEmbeddingBag.
  ret = aclnnEmbeddingBag(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingBag failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto outputSize = GetShapeSize(outputShape);
  std::vector<float> outputResultData(outputSize, 0);
  ret = aclrtMemcpy(outputResultData.data(), outputResultData.size() * sizeof(outputResultData[0]), outputDeviceAddr,
                    outputSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < outputSize; i++) {
    LOG_PRINT("outputResult[%ld] is: %f\n", i, outputResultData[i]);
  }

  auto offset2bagSize = GetShapeSize(offset2bagShape);
  std::vector<int64_t> offset2bagResultData(offset2bagSize, 0);
  ret = aclrtMemcpy(offset2bagResultData.data(), offset2bagResultData.size() * sizeof(offset2bagResultData[0]), offset2bagDeviceAddr,
                    offset2bagSize * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < offset2bagSize; i++) {
    LOG_PRINT("offset2bagResult[%ld] is: %ld\n", i, offset2bagResultData[i]);
  }

  auto bagSizeSize = GetShapeSize(bagSizeShape);
  std::vector<int64_t> bagSizeResultData(bagSizeSize, 0);
  ret = aclrtMemcpy(bagSizeResultData.data(), bagSizeResultData.size() * sizeof(bagSizeResultData[0]), bagSizeDeviceAddr,
                    bagSizeSize * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < bagSizeSize; i++) {
    LOG_PRINT("bagSizeResult[%ld] is: %ld\n", i, bagSizeResultData[i]);
  }

  auto maxIndicesSize = GetShapeSize(maxIndicesShape);
  std::vector<int64_t> maxIndicesResultData(maxIndicesSize, 0);
  ret = aclrtMemcpy(maxIndicesResultData.data(), maxIndicesResultData.size() * sizeof(maxIndicesResultData[0]), maxIndicesDeviceAddr,
                    maxIndicesSize * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < maxIndicesSize; i++) {
    LOG_PRINT("maxIndicesResult[%ld] is: %ld\n", i, maxIndicesResultData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(weight);
  aclDestroyTensor(indices);
  aclDestroyTensor(offsets);
  aclDestroyTensor(perSampleWeights);
  aclDestroyTensor(output);
  aclDestroyTensor(offset2bag);
  aclDestroyTensor(bagSize);
  aclDestroyTensor(maxIndices);
  
  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(weightDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(offsetsDeviceAddr);
  aclrtFree(perSampleWeightsDeviceAddr);
  aclrtFree(outputDeviceAddr);
  aclrtFree(offset2bagDeviceAddr);
  aclrtFree(bagSizeDeviceAddr);
  aclrtFree(maxIndicesDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
