# aclnnBatchNormElemtBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 接口功能：[aclnnBatchNormElemt](../../batch_norm_elemt/docs/aclnnBatchNormElemt.md)的反向计算。用于计算输入张量的元素级梯度，以便在反向传播过程中更新模型参数。
- 计算公式：

  $$
  gradInput = ({gradOut} - \frac{sumDy}{ {counter}}) - ((input - mean) * (invstd^{2} *   (\frac{sumDyXmu}{ {counter}}))) * invstd * weight
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBatchNormElemtBackwardGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBatchNormElemtBackward”接口执行计算。

```Cpp
aclnnStatus aclnnBatchNormElemtBackwardGetWorkspaceSize(
  const aclTensor* gradOut,
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* invstd,
  const aclTensor* weight,
  const aclTensor* sumDy,
  const aclTensor* sumDyXmu,
  aclTensor*       counter,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnBatchNormElemtBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormElemtBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <td>gradOut</td>
      <td>输入</td>
      <td>表示正向输出的微分，对应公式中的`gradOut`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与`input`的shape保持一致。</li><li>第2维固定为channel轴。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>表示进行BatchNorm计算的输入，对应公式中的`input`。</td>
      <td><ul><li>支持空Tensor。</li><li>第2维固定为channel轴，且channel轴的size不能为0。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示输入数据均值，对应公式中的`mean`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd</td>
      <td>输入</td>
      <td>表示输入数据标准差倒数，对应公式中的`invstd`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示权重Tensor，对应公式中的`weight`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选参数。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sumDy</td>
      <td>输入</td>
      <td>表示输出梯度的样本均值和的平均值，对应公式中的`sumDy`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sumDyXmu</td>
      <td>输入</td>
      <td>表示样本均值和与输入梯度乘积的平均值，对应公式中的`sumDyXmu`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>counter</td>
      <td>输入</td>
      <td>表示输入数据的数量大小，对应公式中的`counter`。</td>
      <td><ul><li>支持空Tensor。</li></ul></td>
      <td>INT32、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>表示输入Tensor的梯度，对应公式中的`gradInput`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与`input`的shape保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
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
      <td>传入的gradOut、input、mean、invstd、sumDy、sumDyXmu、counter或gradInput是空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>gradOut、input、mean、invstd、sumDy、sumDyXmu、counter、gradInput的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>当weight非空指针时，weight的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradOut、input或gradInput的数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input的维度小于2维。</td>
    </tr>
    <tr>
      <td>input、gradOut、gradInput或counter的维度大于8维。</td>
    </tr>
    <tr>
      <td>input的channel轴的size为0。</td>
    </tr>
    <tr>
      <td>gradOut或gradInput的shape与input不一致。</td>
    </tr>
    <tr>
      <td>mean、invstd、sumDy或sumDyXmu的shape与input的channel轴不一致。</td>
    </tr>
    <tr>
      <td>当weight非空指针时，weight的shape与input的channel轴不一致。</td>
    </tr>
  </tbody></table>

## aclnnBatchNormElemtBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBatchNormElemtBackwardGetWorkspaceSize获取。</td>
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
  - aclnnBatchNormElemtBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_elemt_backward.h"

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
  std::vector<int64_t> gradOutShape = {1, 2, 4};
  std::vector<int64_t> inputShape = {1, 2, 4};
  std::vector<int64_t> meanShape = {2};
  std::vector<int64_t> invstdShape = {2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> sumDyShape = {2};
  std::vector<int64_t> sumDyXmuShape = {2};
  std::vector<int64_t> counterShape = {2};
  std::vector<int64_t> gradInputShape = {1, 2, 4};
  void* gradOutDeviceAddr = nullptr;
  void* inputDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* invstdDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* sumDyDeviceAddr = nullptr;
  void* sumDyXmuDeviceAddr = nullptr;
  void* counterDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* input = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* invstd = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* sumDy = nullptr;
  aclTensor* sumDyXmu = nullptr;
  aclTensor* counter = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> meanHostData = {0, 0};
  std::vector<float> invstdHostData = {1, 1};
  std::vector<float> weightHostData = {1, 1};
  std::vector<float> sumDyHostData = {0, 0};
  std::vector<float> sumDyXmuHostData = {1, 1};
  std::vector<float> counterHostData = {5, 5};
  std::vector<float> gradInputHostData(8, 0);

  // 创建gradOut aclTensor
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mean aclTensor
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建invstd aclTensor
  ret = CreateAclTensor(invstdHostData, invstdShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sumDy aclTensor
  ret = CreateAclTensor(sumDyHostData, sumDyShape, &sumDyDeviceAddr, aclDataType::ACL_FLOAT, &sumDy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sumDyXmu aclTensor
  ret = CreateAclTensor(sumDyXmuHostData, sumDyXmuShape, &sumDyXmuDeviceAddr, aclDataType::ACL_FLOAT, &sumDyXmu);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建counter aclTensor
  ret = CreateAclTensor(counterHostData, counterShape, &counterDeviceAddr, aclDataType::ACL_FLOAT, &counter);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradInput aclTensor
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBatchNormElemtBackward接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnBatchNormElemtBackward第一段接口
  ret = aclnnBatchNormElemtBackwardGetWorkspaceSize(gradOut, input, mean, invstd, weight, sumDy, sumDyXmu, counter, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormElemtBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBatchNormElemtBackward第二段接口
  ret = aclnnBatchNormElemtBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormElemtBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOut);
  aclDestroyTensor(input);
  aclDestroyTensor(weight);
  aclDestroyTensor(mean);
  aclDestroyTensor(invstd);
  aclDestroyTensor(sumDy);
  aclDestroyTensor(sumDyXmu);
  aclDestroyTensor(counter);
  aclDestroyTensor(gradInput);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(inputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(invstdDeviceAddr);
  aclrtFree(sumDyDeviceAddr);
  aclrtFree(sumDyXmuDeviceAddr);
  aclrtFree(counterDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
