# aclnnEluBackward

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |


## 功能说明

- 算子功能：[aclnnElu](../../elu/docs/aclnnElu&aclnnInplaceElu.md)激活函数的反向计算，输出ELU激活函数正向输入的梯度。

- 计算公式：$x$是selfOrResult中的某个元素。

  - 当isResult是True时：

    $$
    gradInput = gradOutput *
    \begin{cases}
    scale, \quad x > 0\\
    inputScale \ast (x + \alpha \ast scale),  \quad x \leq 0
    \end{cases}
    $$

  - 当isResult是False时：

    $$
    gradInput = gradOutput *
    \begin{cases}
    scale, \quad x > 0\\
    inputScale \ast \alpha \ast scale \ast exp(x \ast inputScale), \quad x \leq 0
    \end{cases}
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnEluBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnEluBackward”接口执行计算。

```Cpp
aclnnStatus aclnnEluBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclScalar* alpha,
  const aclScalar* scale,
  const aclScalar* inputScale,
  bool             isResult,
  const aclTensor* selfOrResult,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnEluBackward(
  void*            workspace,
  uint64_t         workspaceSize,
  aclOpExecutor*   executor,
  aclrtStream      stream)
```

## aclnnEluBackwardGetWorkspaceSize

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1310px"><colgroup>
  <col style="width: 111px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>表示ELU激活函数正向输出的梯度，公式中的gradInput。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>表示ELU激活函数的激活系数，公式中的\alpha。</td>
      <td><ul><li>如果isResult为true，\alpha必须大于等于0。</li><li>数据类型需要是可转换为FLOAT的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>scale</td>
      <td>输入</td>
      <td>表示ELU激活函数的缩放系数，公式中的scale。</td>
      <td>数据类型需要是可转换为FLOAT的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>inputScale</td>
      <td>输入</td>
      <td>表示ELU激活函数的输入的缩放系数，公式中的inputScale。</td>
      <td>数据类型需要是可转换为FLOAT的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>isResult</td>
      <td>输入</td>
      <td>表示传给ELU反向计算的输入是否是ELU正向的输出。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>selfOrResult</td>
      <td>输入</td>
      <td><ul><li>当isResult为True时，表示ELU激活函数正向的输出。</li><li>当isResult为False时，表示ELU激活函数正向的输入。</li></ul></td>
      <td><ul><li>数据类型需要与gradOutput一致。</li><li>shape需要与gradOutput的shape一致。</li><li>支持空Tensor。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
       <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>表示ELU激活函数正向输入的梯度，即对输入进行求导后的结果，公式中的gradOutput。</td>
      <td><ul><li>数据类型需要是gradOutput推导之后可转换的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</li><li>shape需要和gradOutput的shape一致。</li></ul></td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
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
      <td>参数gradOutput、alpha、scale、inputScale、selfOrResult、gradInput是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>参数gradOutput、selfOrResult的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>参数gradOutput、selfOrResult的数据类型不一致。</td>
    </tr>
    <tr>
      <td>参数alpha、scale、inputScale的数据类型不可转换为FLOAT。</td>
    </tr>
    <tr>
      <td>参数gradInput的数据类型不是gradOutput可转换的。</td>
    </tr>
    <tr>
      <td>参数gradOutput、selfOrResult、gradInput的shape不一致。</td>
    </tr>
     <tr>
      <td>参数gradOutput、selfOrResult、gradInput的维度大于8。</td>
    </tr>
    <tr>
      <td>参数isResult为True时，alpha小于0。</td>
    </tr>
  </tbody></table>


## aclnnEluBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnEluBackwardGetWorkspaceSize获取。</td>
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

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_elu_backward.h"

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
                    aclDataType dataType, aclTensor** selfOrResult) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续selfOrResult的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *selfOrResult = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
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
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfOrResultShape = {2, 2};
  std::vector<int64_t> gradInputShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfOrResultDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclScalar* alpha = nullptr;
  aclScalar* scale = nullptr;
  aclScalar* inputScale = nullptr;
  aclTensor* selfOrResult = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutputHostData = {-2, -1, 0, 1};
  std::vector<float> selfOrResultHostData = {-2, -1, 0, 1};
  std::vector<float> gradInputHostData = {0, 0, 0, 0};
  float alphaValue = 1.0f;
  float scaleValue = 1.0f;
  float inputScaleValue = 1.0f;
  bool isResult = true;
  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT,
                        &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建alpha aclScalar
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // 创建scale aclScalar
  scale = aclCreateScalar(&scaleValue, aclDataType::ACL_FLOAT);
  CHECK_RET(scale != nullptr, return ret);
  // 创建inputScale aclScalar
  inputScale = aclCreateScalar(&inputScaleValue, aclDataType::ACL_FLOAT);
  CHECK_RET(inputScale != nullptr, return ret);
  // 创建selfOrResult aclTensor
  ret = CreateAclTensor(selfOrResultHostData, selfOrResultShape, &selfOrResultDeviceAddr, aclDataType::ACL_FLOAT,
                        &selfOrResult);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradInput aclTensor
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnEluBackward接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // 调用aclnnEluBackward第一段接口
  ret = aclnnEluBackwardGetWorkspaceSize(gradOutput, alpha, scale, inputScale, isResult, selfOrResult, gradInput,
                                         &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnEluBackward第二段接口
  ret = aclnnEluBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEluBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyScalar(alpha);
  aclDestroyScalar(scale);
  aclDestroyScalar(inputScale);
  aclDestroyTensor(selfOrResult);
  aclDestroyTensor(gradInput);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfOrResultDeviceAddr);
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

