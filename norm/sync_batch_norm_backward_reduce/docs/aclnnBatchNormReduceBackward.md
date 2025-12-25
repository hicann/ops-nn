# aclnnBatchNormReduceBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 接口功能：
  
  主要用于反向传播过程中计算BatchNorm操作的梯度，并进行一些中间结果的规约操作以优化计算效率。计算结果如下：
  - 计算损失函数L对缩放权重γ的梯度($\frac{\partial L}{\partial γ}$)。
  - 计算损失函数L对偏移量β的梯度($\frac{\partial L}{\partial β}$)。
  - 以损失函数L相对于输出(y<sub>i</sub>)的偏差d<sub>yi</sub>推导计算$\frac{\partial L}{\partial x_i}$所需的中间量sumDy和sumDyXmu。其中($\frac{\partial L}{\partial x_i}$)为损失函数L相对于对应层各输入(x<sub>i</sub>)的梯度。
  
- 计算公式：
  
  $$
  gradWeight = \frac{\partial l}{\partial γ} = \sum^m_{i=0} \frac{\partial l}{\partial y_i} \cdot \hat{(x_i)} = \frac{1}{{\sqrt{σ^2_B + eps}}} \cdot \sum^m_{i=0} \frac{\partial l}  {\partial y_i} \cdot (x_i-μ_B)
  $$
  
  $$
  gradBias = \frac{\partial l}{\partial β} = \sum^m_{i=0} \frac{\partial l}{\partial y_i}
  $$
  
  $$
  sumDy = sum(l, y_i) = \displaystyle \sum^m_{i=0} \frac{\partial l}{\partial y_i}
  $$
  
  $$
  sumDyXmu = sum(l, y_i, x_i, μ_B) = \displaystyle \sum^m_{i=0} \frac{\partial l}{\partial y_i} \cdot (x_i-μ_B)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBatchNormReduceBackwardGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBatchNormReduceBackward”接口执行计算。

```Cpp
aclnnStatus aclnnBatchNormReduceBackwardGetWorkspaceSize(
  const aclTensor* gradOut,
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* invstd,
  const aclTensor* weight,
  const bool       inputG,
  const bool       weightG,
  const bool       biasG,
  aclTensor*       sumDy,
  aclTensor*       sumDyXmu,
  aclTensor*       gradWeight,
  aclTensor*       gradBias,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnBatchNormReduceBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnBatchNormReduceBackwardGetWorkspaceSize

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
      <td>表示梯度Tensor，对应公式中的`<math ><mfrac><mrow><mi mathvariant="normal">∂</mi><mi>l</mi></mrow>/<mrow><mi mathvariant="normal">∂</mi><mi>y</mi></mrow></mfrac></math>`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape需要与`input`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>表示输入Tensor，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>默认第二维为channel轴，且channel轴的值不能为0。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示均值，对应公式中的`μ<sub>B</sub>`。</td>
      <td><ul><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invstd</td>
      <td>输入</td>
      <td>表示标准差的倒数，对应公式中的`(σ<sub>B</sub>)<sup>2</sup>+eps`的开平方倒数。</td>
      <td><ul><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示权重Tensor，对应公式中的`γ`。</td>
      <td><ul><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputG</td>
      <td>输入</td>
      <td>表示输出掩码，标记是否需要输出`sumDy`和`sumDyXmu`。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightG</td>
      <td>输入</td>
      <td>表示输出掩码，标记是否需要输出`gradWeight`。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasG</td>
      <td>输入</td>
      <td>表示输出掩码，标记是否需要输出`gradBias`。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sumDy</td>
      <td>输出</td>
      <td>表示正向输出梯度`gradOut`的累加和，对应公式中的`sumDy`。</td>
      <td><ul><li>可选输出，如果`inputG`为True则输出，shape的size需要与`input`的channel轴的长度相等。</li><li>数据格式与`gradOut`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sumDyXmu</td>
      <td>输出</td>
      <td>表示正向输出梯度`gradOut`与输入中心化后数据`(x-μ<sub>B</sub>)`乘积之和，对应公式中的`sumDyXmu`。</td>
      <td><ul><li>可选输出，如果`inputG`为True则输出，shape的size需要与`input`的channel轴的长度相等。</li><li>数据格式与`gradOut`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight</td>
      <td>输出</td>
      <td>表示缩放参数的梯度，对应公式中的`gradWeight`。</td>
      <td><ul><li>可选输出，如果`weightG`为True则输出，shape的size需要与`input`的channel轴的长度相等。</li><li>数据格式与`gradOut`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBias</td>
      <td>输出</td>
      <td>表示偏置参数的梯度，对应公式中的`gradBias`。</td>
      <td><ul><li>可选输出，如果`biasG`为True则输出，shape的size需要与`input`的channel轴的长度相等。</li><li>数据格式与`gradOut`保持一致。</li></ul></td>
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
      <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="4">161001</td>
      <td>input、meanAll、invstdAll、mean、invstd、runningMean、runningVar或counts的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>当inputG为true时，sumDy或sumDyXmu是空指针。</td>
    </tr>
    <tr>
      <td>当weightG为true时，gradWeight是空指针。</td>
    </tr>
    <tr>
      <td>当biasG为true时，gradBias是空指针。</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>gradOut、input、mean、invstd、weight、sumDy、sumDyXmu、gradWeight、gradBias的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradOut的数据类型需要与input一致。</td>
    </tr>
    <tr>
      <td>当inputG为true时，sumDy或sumDyXmu的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>当weightG为true时，gradWeight的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>当biasG为true时，gradBias的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradOut和input数据格式不一致。</td>
    </tr>
    <tr>
      <td>gradOut或input的维度大于8维。</td>
    </tr>
    <tr>
      <td>gradOut或input的维度小于2维。</td>
    </tr>
    <tr>
      <td>input的channel轴size为0。</td>
    </tr>
    <tr>
      <td>当inputG为true时，sumDy或sumDyXmu的size与input的channel轴不一致。</td>
    </tr>
    <tr>
      <td>当weightG为true时，gradWeight的size与input的channel轴不一致。</td>
    </tr>
    <tr>
      <td>当biasG为true时，gradBias的size与input的channel轴不一致。</td>
    </tr>
  </tbody></table>

## aclnnBatchNormReduceBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBatchNormReduceBackwardGetWorkspaceSize获取。</td>
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

- 当任一输入的为空Tensor时，输出为空Tensor。
- 确定性计算：
  - aclnnBatchNormReduceBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm_backward_reduce.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
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
  std::vector<int64_t> gradOutShape = {4, 2};
  std::vector<int64_t> inputShape = {4, 2};
  std::vector<int64_t> meanShape = {2};
  std::vector<int64_t> invstdShape = {2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> sumDyShape = {2};
  std::vector<int64_t> sumDyXmuShape = {2};
  std::vector<int64_t> gradWeightShape = {2};
  std::vector<int64_t> gradBiasShape = {2};

  void* gradOutDeviceAddr = nullptr;
  void* inputDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* invstdDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* sumDyDeviceAddr = nullptr;
  void* sumDyXmuDeviceAddr = nullptr;
  void* gradWeightDeviceAddr = nullptr;
  void* gradBiasDeviceAddr = nullptr;

  aclTensor* gradOut = nullptr;
  aclTensor* input = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* invstd = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* sumDy = nullptr;
  aclTensor* sumDyXmu = nullptr;
  aclTensor* gradWeight = nullptr;
  aclTensor* gradBias = nullptr;

  std::vector<float> gradOutHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> meanHostData = {1, 1};
  std::vector<float> invstdHostData = {1, 1};
  std::vector<float> weightHostData = {1, 1};
  std::vector<float> sumDyHostData = {1, 1};
  std::vector<float> sumDyXmuHostData = {1, 1};
  std::vector<float> gradWeightHostData = {1, 1};
  std::vector<float> gradBiasHostData = {1, 1};

  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(invstdHostData, invstdShape, &invstdDeviceAddr, aclDataType::ACL_FLOAT, &invstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  bool inputG = true;
  bool weightG = true;
  bool biasG = true;

  ret = CreateAclTensor(sumDyHostData, sumDyShape, &sumDyDeviceAddr, aclDataType::ACL_FLOAT, &sumDy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sumDyXmuHostData, sumDyXmuShape, &sumDyXmuDeviceAddr, aclDataType::ACL_FLOAT, &sumDyXmu);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBatchNormReduceBackward接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnBatchNormReduceBackward第一段接口
  ret = aclnnBatchNormReduceBackwardGetWorkspaceSize(gradOut, input, mean, invstd, weight,
                                                     inputG, weightG, biasG,
                                                     sumDy, sumDyXmu, gradWeight, gradBias,
                                                     &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormReduceBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnBatchNormReduceBackward第二段接口
  ret = aclnnBatchNormReduceBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormReduceBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  PrintOutResult(sumDyShape, &sumDyDeviceAddr);
  PrintOutResult(sumDyXmuShape, &sumDyXmuDeviceAddr);
  PrintOutResult(gradWeightShape, &gradWeightDeviceAddr);
  PrintOutResult(gradBiasShape, &gradBiasDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOut);
  aclDestroyTensor(input);
  aclDestroyTensor(mean);
  aclDestroyTensor(invstd);
  aclDestroyTensor(weight);
  aclDestroyTensor(sumDy);
  aclDestroyTensor(sumDyXmu);
  aclDestroyTensor(gradWeight);
  aclDestroyTensor(gradBias);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(inputDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(invstdDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(sumDyDeviceAddr);
  aclrtFree(sumDyXmuDeviceAddr);
  aclrtFree(gradWeightDeviceAddr);
  aclrtFree(gradBiasDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
