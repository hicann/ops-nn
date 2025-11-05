# aclnnPreluBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明
算子功能：完成[aclnnPreluBackward](../../prelu_grad_update/docs/aclnnPreluBackward.md)的反向函数。
gradInput的计算公式如下：

$$
gradInput_{i,j,...}=
\begin{cases}
gradOutput_{i,j,...}, & if\ self_{i,j,...} > 0 \\
gradOutput_{i,j,...} * weight_{i}, & if\ self_{i,j,...} <= 0
\end{cases}
$$

gradWeight的计算公式如下：

$$
gradWeight_{j}=\sum_{i,...}
\begin{cases}
0, & if\ self_{i,j,...} > 0 \\
gradOutput_{i,j,...} * self_{i,j,...}, & if\ self_{i,j,...} <= 0
\end{cases}
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnPreluBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnPreluBackward”接口执行计算。

```Cpp
aclnnStatus aclnnPreluBackwardGetWorkspaceSize(
  const aclTensor*  gradOutput,
  const aclTensor*  self,
  const aclTensor*  weight,
  aclTensor*        gradInput,
  aclTensor*        gradWeight,
  uint64_t*         workspaceSize,
  aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnPreluBackward(
  void             *workspace,
  uint64_t          workspace_size,
   aclOpExecutor   *executor,
   aclrtStream      stream)
```


## aclnnPreluBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1458px"><colgroup>
  <col style="width: 154px">
  <col style="width: 120px">
  <col style="width: 276px">
  <col style="width: 308px">
  <col style="width: 212px">
  <col style="width: 107px">
  <col style="width: 136px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>反向传播的梯度值。公式中的gradOutput。</td>
      <td><ul><li>支持空Tensor。</li><li>dtype需要与self相同。</li><li>shape需要与self满足<a href="../../../docs/context/broadcast关系.md" target="_blank">broadcast关系</a>，且Broadcast后shape与self的shape相等。</li></ul></td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>prelu的正向输入值。公式中的self。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>prelu的权重，公式中的weight。</td>
      <td><ul><li>支持空Tensor。</li><li>dtype需要与self相同。</li><li>当self的shape维度大于1维时，weight的shape维度可以与self的shape维度相同且第2维度的值保持一致，同时weight的shape其他维度的值为1；或者weight是1维Tensor，元素个数为self的shape的第2维度。</li><li></li>否则，weight元素个数为1。</ul></td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>为self的梯度值。</td>
      <td><ul><li>dtype需要与self相同。</li><li>shape需要与gradOutput满足<a href="../../../docs/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li><li>gradInput的shape和数据类型与self的相同。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight</td>
      <td>输出</td>
      <td>为weight的梯度值。</td>
      <td><ul><li>dtype需要与self相同。</li><li>需要与weight的数据类型相同。</li><li>gradWeight的shape与weight的shape保持一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
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
  第一段接口会完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>传入的gradOutput、self、weight、gradInput、gradWeight是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>gradOutput、self、weight、gradInput、gradWeight的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradOutput、self、weight、gradInput、gradWeight的数据类型不同。</td>
    </tr>
    <tr>
      <td>gradOutput、self、weight、gradInput、gradWeight大于8维。</td>
    </tr>
    <tr>
      <td>weight的元素个数不等于self的通道数或者1。</td>
    </tr>
    <tr>
      <td>weight的元素个数为1时，gradWeight的shape与weight不相同。</td>
    </tr>
    <tr>
      <td>gradOutput和self的shape不满足条件(支持broadcast同时broadcastshape等于self, 即单向broadcast)。</td>
    </tr>
  </tbody></table>


## aclnnPreluBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnPreluBackwardGetWorkspaceSize获取。</td>
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
#include "aclnnop/aclnn_prelu_backward.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> weightShape = {2};
  std::vector<int64_t> gradOutputShape = {4, 2};
  std::vector<int64_t> gradInputShape = {4, 2};
  std::vector<int64_t> gradWeightShape = {2};

  void* selfDeviceAddr = nullptr;
  void* gradOutputDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  void* gradWeightDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* gradInput = nullptr;
  aclTensor* gradWeight = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> weightHostData = {0.5, 0.5};
  std::vector<float> gradOutputHostData = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> gradInputHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> gradWeightHostData = {0, 0};

  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT,
  &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradInput aclTensor
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradWeight aclTensor
  ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnPreluBackward第一段接口
  ret = aclnnPreluBackwardGetWorkspaceSize(gradOutput, self, weight, gradInput, gradWeight, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPreluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnPreluBackward第二段接口
  ret = aclnnPreluBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPreluBackward failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto gradInputSize = GetShapeSize(gradInputShape);
  std::vector<float> gradInputResultData(gradInputSize, 0);
  ret = aclrtMemcpy(gradInputResultData.data(), gradInputResultData.size() * sizeof(gradInputResultData[0]), gradInputDeviceAddr, gradInputSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < gradInputSize; i++) {
    LOG_PRINT("gradInput[%ld] is: %f\n", i, gradInputResultData[i]);
  }

  auto gradWeightSize = GetShapeSize(gradWeightShape);
  std::vector<float> gradWeightResultData(gradWeightSize, 0);
  ret = aclrtMemcpy(gradWeightResultData.data(), gradWeightResultData.size() * sizeof(gradWeightResultData[0]), gradWeightDeviceAddr, gradWeightSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < gradWeightSize; i++) {
    LOG_PRINT("gradWeight[%ld] is: %f\n", i, gradWeightResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(weight);
  aclDestroyTensor(gradInput);
  aclDestroyTensor(gradWeight);

  // 7. 释放device资源， 需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  aclrtFree(gradWeightDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```

