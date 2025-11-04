# aclnnBatchNormGatherStatsWithCounts

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：
  
  收集所有device的均值和方差，更新全局的均值和标准差的倒数。BatchNorm的性能和BatchSize相关，BatchSize越大，BatchNorm的统计量也会越准。然而像检测这样的任务，占用显存较高，一张显卡往往只拿较少的图片，比如两张来训练，这就导致BN的表现变差。一个解决方式就是SyncBatchNorm，所有卡共享同一个BatchNorm，得到全局的统计量。

  aclnnBatchNormGatherStatsWithCounts计算时，依赖[aclnnBatchNormStats](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/API/aolapi/context/aclnnBatchNormStats.md)计算单卡数据的均值和标准差的倒数。
  <!--aclnnBatchNormGatherStatsWithCounts计算时，依赖[aclnnBatchNormStats](aclnnBatchNormStats.md)计算单卡数据的均值和标准差的倒数。-->

- 计算公式：

  $$
  y = \frac{(x-E[x])}{\sqrt{Var(x)+ eps}} * γ + β
  $$
  
  其中，runningMean和runningVar更新公式如下：
  
  $$
      runningMean=runningMean*(1-momentum) + E[x]*runningMean
  $$
  
  $$
      runningVar=runningVar*(1-momentum) + E[x]*runningVar
  $$
  
## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBatchNormGatherStatsWithCounts”接口执行计算。

```Cpp
aclnnStatus aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* invstd,
  aclTensor*       runningMean,
  aclTensor*       runningVar,
  double           momentum,
  double           eps,
  const aclTensor* counts,
  aclTensor*       meanAll,
  aclTensor*       invstdAll,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```
```Cpp
aclnnStatus aclnnBatchNormGatherStatsWithCounts(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize

- **参数说明：**

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
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>表示进行统计的样本值。对应公式中的`x`。</td>
      <td><ul><li>不支持空Tensor。</li><li>第2维固定为channel轴。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示输入数据均值，对应公式中的`E(x)`。</td>
      <td><ul><li>不支持空Tensor。</li><li>第一维的size需要与`invstd`和`counts`一致，第二维对应的size需要与`input`的Channel轴size一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd</td>
      <td>输入</td>
      <td>表示输入数据标准差的倒数，对应公式中的`Var(x)+ eps`开平方的倒数。</td>
      <td><ul><li>不支持空Tensor。</li><li>元素值要求均大于0，小于等于0时，精度不做保证。</li><li>第一维的size需要与`mean`和`counts`一致，第二维对应的size需要与`input`的Channel轴size一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningMean</td>
      <td>输入</td>
      <td>可选参数，表示用于跟踪整个训练过程中的均值，对应公式中的`runningMean`。</td>
      <td><ul><li>不支持空Tensor。</li><li>元素值要求均大于0，小于等于0时，精度不做保证。</li><li>当`runningMean`非空指针时，shape支持1维，size需要与`input`的Channel轴size一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningVar</td>
      <td>输入</td>
      <td>可选参数，表示用于跟踪整个训练过程中的方差，对应公式中的`runningVar`。</td>
      <td><ul><li>不支持空Tensor。</li><li>元素值要求均大于0，小于等于0时，精度不做保证。</li><li>当`runningVar`非空指针时，shape支持1维，size需要与`input`的Channel轴size一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>momentum</td>
      <td>输入</td>
      <td>表示runningMean和runningVar的指数平滑参数，对应公式中的`momentum`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>输入</td>
      <td>表示添加到方差中的值，以避免出现除以零的情况。对应公式中的`eps`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>counts</td>
      <td>输入</td>
      <td>表示输入数据元素的个数。</td>
      <td><ul><li>不支持空Tensor。</li><li>支持元素值均为正整数，其余场景不做保证。</li><li>第一维的size需要与mean和invstd的第一维的size一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanAll</td>
      <td>输出</td>
      <td>表示所有卡上数据的均值，对应公式中的`E(x)`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape与入参`input`中channel轴保持一致。</li><li>数据格式与入参`input`中数据格式保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstdAll</td>
      <td>输出</td>
      <td>表示所有卡上数据的标准差的倒数，对应公式中的`Var(x)+ eps`开平方的倒数。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape与入参`input`中channel轴保持一致。</li><li>数据格式与入参`input`中数据格式保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>      
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的input、mean、invstd、counts、meanAll或invstdAll是空指针。</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>input、meanAll、invstdAll、mean、invstd、runningMean、runningVar或counts的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input、mean、invstd、counts的数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input的维度大于8维。</td>
    </tr>
    <tr>
      <td>input的维度小于2维。</td>
    </tr>
    <tr>
      <td>mean或invstd的维度不为2维。</td>
    </tr>
    <tr>
      <td>counts、mean和invstd在第一维上的size不一致。</td>
    </tr>
    <tr>
      <td>mean或invstd在第二维上的size与input的channel轴不一致。</td>
    </tr>
    <tr>
      <td>当runningMean不为空时，runningMean的size与input的channel轴不一致。</td>
    </tr>
    <tr>
      <td>当runningVar不为空时，runningVar的size与input的channel轴不一致。</td>
    </tr>
    <tr>
      <td>counts的维度不为1维。</td>
    </tr>
    <tr>
      <td>meanAll或invstdAll的shape与input的channel轴不一致。</td>
    </tr>
  </tbody></table>

## aclnnBatchNormGatherStatsWithCounts

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
无。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_gather_stats_with_counts.h"

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
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputShape = {2, 4, 2};
  std::vector<int64_t> meanShape = {4, 4};
  std::vector<int64_t> rMeanShape = {4};
  std::vector<int64_t> rVarShape = {4};
  std::vector<int64_t> countsShape = {4};
  std::vector<int64_t> invstdShape = {4, 4};
  std::vector<int64_t> meanAllShape = {4};
  std::vector<int64_t> invstdAllShape = {4};
  void* inputDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* rMeanDeviceAddr = nullptr;
  void* rVarDeviceAddr = nullptr;
  void* countsDeviceAddr = nullptr;
  void* invstdDeviceAddr = nullptr;
  void* meanAllDeviceAddr = nullptr;
  void* invstdAllDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* rMean = nullptr;
  aclTensor* rVar = nullptr;
  aclTensor* counts = nullptr;
  aclTensor* invstd = nullptr;
  aclTensor* meanAll = nullptr;
  aclTensor* invstdAll = nullptr;
  std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> meanHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> rMeanHostData = {1, 2, 3, 4};
  std::vector<float> rVarHostData = {5, 6, 7, 8};
  std::vector<float> countsHostData = {1, 2, 3, 4};
  std::vector<float> invstdHostData = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  std::vector<float> meanAllHostData = {4, 0};
  std::vector<float> invstdAllHostData = {4, 0};
  double momentum = 1e-2;
  double eps = 1e-4;

  // 创建input aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mean aclTensor
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rMean aclTensor
  ret = CreateAclTensor(rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT, &rMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rVar aclTensor
  ret = CreateAclTensor(rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建counts aclTensor
  ret = CreateAclTensor(countsHostData, countsShape, &countsDeviceAddr, aclDataType::ACL_FLOAT, &counts);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建invstd aclTensor
  ret = CreateAclTensor(invstdHostData, invstdShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建meanAll aclTensor
  ret = CreateAclTensor(meanAllHostData, meanAllShape, &meanAllDeviceAddr, aclDataType::ACL_FLOAT, &meanAll);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建invstdAll aclTensor
  ret = CreateAclTensor(invstdAllHostData, invstdAllShape, &invstdAllDeviceAddr, aclDataType::ACL_FLOAT, &invstdAll);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBatchNormGatherStatsWithCounts接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnBatchNormGatherStatsWithCounts第一段接口
  ret = aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(input, mean, invstd, rMean, rVar, momentum, eps, counts, meanAll, invstdAll, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBatchNormGatherStatsWithCounts第二段接口
  ret = aclnnBatchNormGatherStatsWithCounts(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormGatherStatsWithCounts failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(meanAllShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), meanAllDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(input);
  aclDestroyTensor(mean);
  aclDestroyTensor(rMean);
  aclDestroyTensor(rVar);
  aclDestroyTensor(counts);
  aclDestroyTensor(invstd);
  aclDestroyTensor(meanAll);
  aclDestroyTensor(invstdAll);

  // 7. 释放Device资源，需要根据具体API的接口定义修改
  aclrtFree(inputDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(rMeanDeviceAddr);
  aclrtFree(rVarDeviceAddr);
  aclrtFree(countsDeviceAddr);
  aclrtFree(invstdDeviceAddr);
  aclrtFree(meanAllDeviceAddr);
  aclrtFree(invstdAllDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
