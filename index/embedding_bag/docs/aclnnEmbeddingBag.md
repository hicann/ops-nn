# aclnnEmbeddingBag

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：根据indices从weight中获得一组被聚合的数，然后根据offsets的偏移和mode指定的聚合模式对获取的数进行max、sum、mean聚合。其余参数则更细化了计算过程的控制。
- shape推导方式如下：
  假设:
  ```
  weight的shape为(numWeight, embeddingDim)
  indices的shape为(bagIndices)
  offsets的shape为(bagOffsets)
  ```
  - 当mode为sum模式：
    ```
    output的shape 为 includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
   offset2bag的shape 为 (bagIndices,)
    bagSize的shape 为 includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
   maxIndices的shape 为 includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    ```
  - 当mode为mean模式：
    ```
    output的shape 为 includeLastOffset? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    offset2bag的shape 为 (bagIndices,)
    bagSize的shape 为 includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    maxIndices的shape 为 includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    ```
  - 当mode为max模式：
    ```
    output的shape 为 includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    offset2bag的shape 为 (bagIndices,)
    bagSize的shape 为 includeLastOffset ? (bagOffsets - 1) : (bagOffsets,)
    maxIndices的shape 为 includeLastOffset ? (bagOffsets - 1, embeddingDim) : (bagOffsets, embeddingDim)
    ```

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnEmbeddingBagGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEmbeddingBag”接口执行计算。

  - `aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(const aclTensor* weight, const aclTensor* indices,const aclTensor* offsets, bool scaleGradByFreq,int64_t mode, bool sparse, const aclTensor* perSampleWeights, bool includeLastOffset, int64_t paddingIdx, aclTensor* output, aclTensor* offset2bag, aclTensor* bagSize, aclTensor* maxIndices, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnEmbeddingBag(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnEmbeddingBagGetWorkspaceSize

- **参数说明**：

  - weight(aclTensor*, 计算输入)：词嵌入矩阵，包含所有词的嵌入向量，Device侧的aclTensor，shape支持2维，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - indices(aclTensor*, 计算输入)：包含索引的张量，指定要从`weight`中提取哪些词的嵌入向量，Device侧的aclTensor。数据类型支持UINT8、INT8、INT16、INT32、INT64，shape支持0-1维。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - offsets(aclTensor*, 计算输入): 用于将 indices 分割成多个`bag`的偏移量张量，Device侧的aclTensor。数据类型支持UINT8、INT8、INT16、INT32、INT64。shape支持0-1维。
  - scaleGradByFreq(bool，计算输入): 用于控制是否根据词频缩放梯度，当scaleGradByFreq为true时，会根据词频对梯度进行缩放，当scaleGradByFreq为false时，则不会。
  - mode(int64_t, 计算输入)：用于控制聚合模式，Host侧的整型。0表示sum聚合模式，1表示mean聚合模式，其他表示max聚合模式。
  - sparse(bool, 计算输入)：用于控制稀疏模式，Host侧的bool类型。当为false时，表示weight非稀疏矩阵；当为true时，表示weight是稀疏矩阵。
  - perSampleWeights(aclTensor*, 计算输入): 指定样本权重，Device侧的aclTensor。shape支持1维，数据类型与weight一致，仅在sum模式下，可以不是nullptr，其他模式必须为nullptr。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - includeLastOffset(bool, 计算输入)：控制是否包含最后的偏移，Host侧的bool类型。当为false时，表示不包含最后的偏移；当为true时，表示包含最后的偏移。
  - paddingIdx(int64_t, 计算输入): 控制不参与计算的indices，Host侧的整型。
  - output(aclTensor*, 计算输出)：词嵌入矩阵聚合后的结果，Device侧的aclTensor。数据类型与weight一致，shape支持2维。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - offset2bag(aclTensor*, 计算输出)：`bag`的起始偏移，Device侧的aclTensor。数据类型支持INT32、INT64，且offset2bag的数据类型和indices与offsets中精度高的一致，shape支持0-1维。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - bagSize(aclTensor*, 计算输出): 每个`bag`的大小，Device侧的aclTensor。数据类型支持INT32、INT64，且offset2bag的数据类型和indices与offsets中精度高的一致，shape支持0-1维。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - maxIndices(aclTensor*, 计算输出): 当`mode`为max时，词嵌入向量最大值所在的行，Device侧的aclTensor。数据类型支持INT32、INT64，且offset2bag的数据类型和indices与offsets中精度高的一致。当`mode`为max时，shape支持2维；当`mode`非max时，shape支持1维。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR): 1. 传入的 weight、indices、offsets、output、offset2bag、bagSize、maxIndices是空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID): 1. weight数据类型不在支持范围内,weight维度不是2维。
                                      2. indices数据类型不在支持范围内,indices维度不是1维。
                                      3. offsets数据类型不在支持范围内, offsets维度不是1维。
                                      4. indices和offsets的数据类型都不是INT32或INT64。
                                      5. perSampleWeights在传入非nullptr的情况下，数据类型与weight不一致, perSampleWeights不是1维，perSampleWeights元素数量与indices不相等, 在非sum模式下，perSampleWeights不是nullptr。
                                      6. output数据类型与weight不一致,shape与定义不符。
                                      7. offset2bag、bagSize、maxIndices数据类型和shape与推导得到的数据类型和shape不符。
  ```

## aclnnEmbeddingBag

- **参数说明**：

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnEmbeddingBagGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

sparse与scaleGradByFreq仅支持输入False。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
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
  // 固定写法，资源初始化
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
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
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

  //创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建offsets aclTensor
  ret = CreateAclTensor(offsetsHostData, offsetsShape, &offsetsDeviceAddr, aclDataType::ACL_INT64, &offsets);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建perSampleWeights aclTensor
  ret = CreateAclTensor(perSampleWeightsHostData, perSampleWeightsShape, &perSampleWeightsDeviceAddr, aclDataType::ACL_FLOAT, &perSampleWeights);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建output aclTensor
  ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建offset2bag aclTensor
  ret = CreateAclTensor(offset2bagHostData, offset2bagShape, &offset2bagDeviceAddr, aclDataType::ACL_INT64, &offset2bag);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建bagSize aclTensor
  ret = CreateAclTensor(bagSizeHostData, bagSizeShape, &bagSizeDeviceAddr, aclDataType::ACL_INT64, &bagSize);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建maxIndices aclTensor
  ret = CreateAclTensor(maxIndicesHostData, maxIndicesShape, &maxIndicesDeviceAddr, aclDataType::ACL_INT64, &maxIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //非tensor参数
  bool scaleGradByFreq = false;
  int64_t mode = 0;
  bool sparse = false;
  bool includeLastOffset = false;
  int64_t paddingIdx = 1;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnEmbeddingBag第一段接口
  ret = aclnnEmbeddingBagGetWorkspaceSize(weight, indices, offsets, scaleGradByFreq, mode, sparse, perSampleWeights,
            includeLastOffset, paddingIdx, output, offset2bag, bagSize, maxIndices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingBagGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnEmbeddingBag第二段接口
  ret = aclnnEmbeddingBag(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingBag failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
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

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(weight);
  aclDestroyTensor(indices);
  aclDestroyTensor(offsets);
  aclDestroyTensor(perSampleWeights);
  aclDestroyTensor(output);
  aclDestroyTensor(offset2bag);
  aclDestroyTensor(bagSize);
  aclDestroyTensor(maxIndices);

  // 7. 释放device资源, 需要根据具体API的接口定义修改
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

