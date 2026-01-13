# aclnnFusedCrossEntropyLossWithMaxSum

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：本算子是词汇表并行场景下交叉熵计算模块的一部分，解决超大规模词汇表下的显存和计算效率问题，当前部分为计算loss与softMax的结果。
- 计算公式：

          $$
          lossOut = log(sum_exp_logits) - predicted_logits
          $$

          $$
          softMaxOutOptional = exp(vocab_parallel_logits -logits_max.unsqueeze(dim = -1)) \ sum_exp_logits.unsqueeze(dim = -1)
          $$


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFusedCrossEntropyLossWithMaxSum”接口执行计算。

- `aclnnStatus aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize(const aclTensor* logitsMax, const aclTensor* sumExpLogits, const aclTensor* predictedLogits, float labelSmoothing, const aclTensor* inputOptional, const aclTensor* weightOptional, const aclTensor* vocabParallelLogitsOptional, aclTensor* lossOut, aclTensor* softMaxOutOptional, uint64_t* workspaceSize, aclOpExecutor** executor);`
- `aclnnStatus aclnnFusedCrossEntropyLossWithMaxSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize

- **参数说明：**

  - logitsMax(aclTensor*，计算输入)：matmul计算后各行的最大值，公式中的logitsMax。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据维度支持1维，数据类型支持FLOAT。

  - sumExpLogits(aclTensor*，计算输入)：matmul计算结果与其各行的最大值作差后exp的结果。公式中的sumExpLogits。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据维度支持1维，shape与logitsMax一致，数据类型支持FLOAT。

  - predictedLogits(aclTensor*，计算输入)：表示matmul计算结果与其各行的最大值作差后maskedTargetOut筛选后的结果。公式中的predictedLogits。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据维度支持1维，shape与logitsMax一致，数据类型支持FLOAT。

  - labelSmoothing(float，计算输入)：标签平滑系数，用于缓解过拟合，当前只支持0。

  - inputOptional(aclTensor*，计算输入)：matmul输入左矩阵。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，当前只支持输入空指针。

  - weightOptional(aclTensor*，计算输入)：matmul输入右矩阵。权重矩阵。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，当前只支持输入空指针。

  - vocabParallelLogitsOptional(aclTensor*，计算输入)：matmul计算结果。公式中的vocabParallelLogits。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据维度支持2维，shape第1维需要与logitsMax第1维一致。数据类型支持FLOAT16，BFLOAT16。

  - lossOut(aclTensor*，计算输出)：中间变量。公式中的loss，Device侧的aclTensor，shape与logitsMax一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据类型支持FLOAT。

  - softMaxOutOptional(aclTensor*，计算输出)：中间变量。公式中的vocabParallelLogits，Device侧的aclTensor，shape与vocabParallelLogitsOptional一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据类型支持FLOAT。

  - workspaceSize(uint64_t*，出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. 参数self、index、out是空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1. 输入，输出参数的数据类型不在支持的范围内。
                                   2. 输入，输出参数的维度不在支持的范围内。
                                   3. 输入，输出参数的shape不满足约束。
                                   4. labelSmoothing不等于0。
```

## aclnnFusedCrossEntropyLossWithMaxSum

- **参数说明：**

  - workspace(void*，入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize获取。

  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnFusedCrossEntropyLossWithMaxSum默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_cross_entropy_loss_with_max_sum.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> logitsMaxShape = {2};
  std::vector<int64_t> sumExpLogitsShape = {2};
  std::vector<int64_t> predictedLogitsShape = {2};
  std::vector<int64_t> inputShape = {2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> vocabParallelLogitsOptionalShape = {2, 2};
  std::vector<int64_t> lossOutShape = {2};
  std::vector<int64_t> softMaxOutOptionalShape = {2, 2};

  float labelSmoothing = 0;

  void* logitsMaxDeviceAddr = nullptr;
  void* sumExpLogitsDeviceAddr = nullptr;
  void* predictedLogitsDeviceAddr = nullptr;
  void* inputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* vocabParallelLogitsOptionalDeviceAddr = nullptr;
  void* lossOutDeviceAddr = nullptr;
  void* softMaxOutOptionalDeviceAddr = nullptr;

  aclTensor* logitsMax = nullptr;
  aclTensor* sumExpLogits = nullptr;
  aclTensor* predictedLogits = nullptr;
  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* vocabParallelLogitsOptional = nullptr;
  aclTensor* lossOut = nullptr;
  aclTensor* softMaxOutOptional = nullptr;

  std::vector<float> logitsMaxHostData = {0.5, 1};
  std::vector<float> sumExpLogitsHostData = {0.5, 1};
  std::vector<float> predictedLogitsHostData = {0.5, 1};
  std::vector<float> inputHostData = {0, 1};
  std::vector<float> weightHostData = {0, 1};
  std::vector<float> vocabParallelLogitsOptionalHostData = {1, 0.5, 0.5, 1};
  std::vector<float> lossOutHostData = {0, 0};
  std::vector<float> softMaxOutOptionalHostData = {0, 0, 0, 0};
  // 创建 aclTensor
  ret = CreateAclTensor(logitsMaxHostData, logitsMaxShape, &logitsMaxDeviceAddr, aclDataType::ACL_FLOAT, &logitsMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(sumExpLogitsHostData, sumExpLogitsShape, &sumExpLogitsDeviceAddr, aclDataType::ACL_FLOAT, &sumExpLogits);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(predictedLogitsHostData, predictedLogitsShape, &predictedLogitsDeviceAddr, aclDataType::ACL_FLOAT, &predictedLogits);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(vocabParallelLogitsOptionalHostData, vocabParallelLogitsOptionalShape, &vocabParallelLogitsOptionalDeviceAddr, aclDataType::ACL_FLOAT, &vocabParallelLogitsOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(lossOutHostData, lossOutShape, &lossOutDeviceAddr, aclDataType::ACL_FLOAT, &lossOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(softMaxOutOptionalHostData, softMaxOutOptionalShape, &softMaxOutOptionalDeviceAddr, aclDataType::ACL_FLOAT, &softMaxOutOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnFusedCrossEntropyLossWithMaxSum第一段接口
  ret = aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize(logitsMax, sumExpLogits, predictedLogits, labelSmoothing, input, weight,
                                                          vocabParallelLogitsOptional, lossOut, softMaxOutOptional, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnFusedCrossEntropyLossWithMaxSum第二段接口
  ret = aclnnFusedCrossEntropyLossWithMaxSum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedCrossEntropyLossWithMaxSum failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(lossOutShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), lossOutDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  size = GetShapeSize(softMaxOutOptionalShape);
  std::vector<float> secondResultData(size, 0);
  ret = aclrtMemcpy(secondResultData.data(), secondResultData.size() * sizeof(secondResultData[0]), softMaxOutOptionalDeviceAddr,
                    size * sizeof(secondResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, secondResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(logitsMax);
  aclDestroyTensor(sumExpLogits);
  aclDestroyTensor(predictedLogits);
  aclDestroyTensor(input);
  aclDestroyTensor(weight);
  aclDestroyTensor(vocabParallelLogitsOptional);
  aclDestroyTensor(lossOut);
  aclDestroyTensor(softMaxOutOptional);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(logitsMaxDeviceAddr);
  aclrtFree(sumExpLogitsDeviceAddr);
  aclrtFree(predictedLogitsDeviceAddr);
  aclrtFree(inputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(vocabParallelLogitsOptionalDeviceAddr);
  aclrtFree(lossOutDeviceAddr);
  aclrtFree(softMaxOutOptionalDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

