# aclnnFusedCrossEntropyLossWithMaxSum

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/fused_cross_entropy_loss_with_max_sum)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：本算子是词汇表并行场景下交叉熵计算模块的一部分，解决超大规模词汇表下的显存和计算效率问题，当前部分为计算loss与softMax的结果。
- 计算公式：

    $$
    lossOut = log(sum\_exp\_logits) - predicted\_logits
    $$

    $$
    softMaxOutOptional = exp(vocab\_parallel\_logits -logits\_max.unsqueeze(dim = -1)) \ sum\_exp\_logits.unsqueeze(dim = -1)
    $$


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFusedCrossEntropyLossWithMaxSum”接口执行计算。

```Cpp
aclnnStatus aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize(
    const aclTensor* logitsMax,
    const aclTensor* sumExpLogits,
    const aclTensor* predictedLogits,
    float            labelSmoothing, 
    const aclTensor* inputOptional,
    const aclTensor* weightOptional,
    const aclTensor* vocabParallelLogitsOptional,
    aclTensor*       lossOut,
    aclTensor*       softMaxOutOptional,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnFusedCrossEntropyLossWithMaxSum(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize

- **参数说明：**

  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1268px"><colgroup>
  <col style="width: 267px">
  <col style="width: 87px">
  <col style="width: 274px">
  <col style="width: 193px">
  <col style="width: 118px">
  <col style="width: 113px">
  <col style="width: 108px">
  <col style="width: 108px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">logitsMax（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">matmul计算后各行的最大值，公式中的logitsMax。</td>
      <td class="tg-0pky">数据维度支持1维。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">sumExpLogits（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">matmul计算结果与其各行的最大值作差后exp的结果，公式中的sumExpLogits。</td>
      <td class="tg-0pky">数据维度支持1维，shape与logitsMax一致。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">predictedLogits（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">表示matmul计算结果与其各行的最大值作差后maskedTargetOut筛选后的结果，公式中的predictedLogits。</td>
      <td class="tg-0pky">数据维度支持1维，shape与logitsMax一致。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">labelSmoothing（float）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">标签平滑系数，用于缓解过拟合。</td>
      <td class="tg-0pky">当前只支持0。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">inputOptional（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">matmul输入左矩阵。</td>
      <td class="tg-0pky">当前只支持输入空指针。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightOptional（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">matmul输入右矩阵，权重矩阵。</td>
      <td class="tg-0pky">当前只支持输入空指针。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">vocabParallelLogitsOptional（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">matmul计算结果，公式中的vocabParallelLogits。</td>
      <td class="tg-0pky">数据维度支持2维，shape第1维需要与logitsMax第1维一致。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">lossOut（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">中间变量，公式中的loss。</td>
      <td class="tg-0pky">shape与logitsMax一致。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">softMaxOutOptional（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">中间变量，公式中的vocabParallelLogits。</td>
      <td class="tg-0pky">shape与vocabParallelLogitsOptional一致。</td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize（uint64_t*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor（aclOpExecutor**）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  
  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 263px">
  <col style="width: 88px">
  <col style="width: 619px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">返回值</th>
      <th class="tg-0pky">错误码</th>
      <th class="tg-0pky">描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">logitsMax、sumExpLogits、predictedLogits、lossOut是空指针。</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="4">161002</td>
      <td class="tg-0pky">输入和输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td class="tg-0pky">输入和输出shape不满足约束。</td>
    </tr>
    <tr>
      <td class="tg-0pky">输入和输出维度不满足约束。</td>
    </tr>
    <tr>
      <td class="tg-0lax">labelSmoothing不等于0。</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedCrossEntropyLossWithMaxSum

- **参数说明：**

   <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
      </colgroup>
      <thead>
        <tr>
          <th>参数名</th>
          <th>输入/输出</th>
          <th>描述</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>workspace</td>
          <td>输入</td>
          <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>输入</td>
          <td>在Device侧申请的workspace大小，由第一段接口aclnnBinaryCrossEntropyBackwardGetWorkspaceSize获取。</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>输入</td>
          <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>输入</td>
          <td>指定执行任务的Stream。</td>
        </tr>
      </tbody>
    </table>

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

