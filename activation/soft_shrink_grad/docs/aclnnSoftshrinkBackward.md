# aclnnSoftshrinkBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/soft_shrink_grad)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

- 接口功能：计算[Softshrink](../../softshrink/docs/aclnnSoftshrink.md)的反向传播梯度。`gradOutput`表示上游梯度，`self`表示Softshrink正向计算的输入。
- 计算公式：
  - Softshrink正向计算公式：

    $$
    Softshrink(x)=
    \begin{cases}
    x - {\lambda}, \ if \ x > {\lambda}\\
    x + {\lambda}, \ if \ x < -{\lambda}\\
    0, \ otherwise
    \end{cases}
    $$

  - Softshrink反向计算公式：

    $$
    gradInput=
    \begin{cases}
    gradOutput, \ if \ \lvert self \rvert > \lambda \ or \ self \ is \ NaN \\
    0, \ otherwise
    \end{cases}
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSoftshrinkBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSoftshrinkBackward”接口执行计算。

```Cpp
aclnnStatus aclnnSoftshrinkBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclTensor* self,
  const aclScalar* lambda,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSoftshrinkBackward(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnSoftshrinkBackwardGetWorkspaceSize

- **参数说明：**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>gradOutput（aclTensor*）</td>
      <td>输入</td>
      <td>反向传播的上游梯度，公式中的gradOutput。</td>
      <td><ul><li>不能传入空指针，支持空Tensor。</li><li>shape与self满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast关系</a>。</li><li>数据类型可以与self不同；不同时内部转换为FLOAT计算。</li></ul></td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>Softshrink正向计算的输入，公式中的self。</td>
      <td><ul><li>不能传入空指针，支持空Tensor。</li><li>shape与gradOutput满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast关系</a>。</li><li>数据类型可以与gradOutput不同；不同时内部转换为FLOAT计算。</li></ul></td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>lambda（aclScalar*）</td>
      <td>输入</td>
      <td>Softshrink计算的阈值，公式中的λ。</td>
      <td>不能传入空指针，取值必须大于或等于0。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>gradInput（aclTensor*）</td>
      <td>输出</td>
      <td>Softshrink反向传播的输出梯度，公式中的gradInput。</td>
      <td><ul><li>不能传入空指针，支持空Tensor。</li><li>shape必须与self和gradOutput的broadcast结果一致。</li></ul></td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor（aclOpExecutor**）</td>
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

  <!-- npu="950,A3,910b" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：gradOutput、self和gradInput的数据类型支持FLOAT16、FLOAT、BFLOAT16。
  <!-- end id7 -->
  <!-- npu="310b,310p,910" id8 -->
  - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：gradOutput、self和gradInput的数据类型支持FLOAT16、FLOAT。
  <!-- end id8 -->

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>gradOutput、self、lambda、gradInput、workspaceSize或executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>gradOutput、self或gradInput的数据类型不在当前产品支持范围内。</td>
    </tr>
    <tr>
      <td>self和gradOutput的shape不满足broadcast规则。</td>
    </tr>
    <tr>
      <td>lambda为NaN或小于0。</td>
    </tr>
    <tr>
      <td>gradInput的shape与self和gradOutput的broadcast结果不一致。</td>
    </tr>
    <tr>
      <td>gradOutput或self的维度大于8。</td>
    </tr>
  </tbody></table>

## aclnnSoftshrinkBackward

- **参数说明：**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSoftshrinkBackwardGetWorkspaceSize获取。</td>
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

- **返回值**

   aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnSoftshrinkBackward默认确定性实现。

## 调用示例

以下代码演示基本调用流程，编译和执行方法请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_softshrink_backward.h"

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
  // 初始化运行时资源
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
  // 1. 初始化device和stream
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出
  std::vector<int64_t> gradOutputShape = {4, 2};
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> gradInputShape = {4, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gradInput = nullptr;
  aclScalar* lambda = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> selfHostData = {1, 1, 1, 2, 1, 2, 3, 3};
  std::vector<float> gradInputHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float lambdaValue = 1.2f;
  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // lambda aclScalar
  lambda = aclCreateScalar(&lambdaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(lambda != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return -1);
  // 创建gradInput aclTensor
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // 调用aclnnSoftshrinkBackward第一段接口
  ret = aclnnSoftshrinkBackwardGetWorkspaceSize(gradOutput, self, lambda, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftshrinkBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnSoftshrinkBackward第二段接口
  ret = aclnnSoftshrinkBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftshrinkBackward failed. ERROR: %d\n", ret); return ret);
  // 4. 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出，将device侧结果拷贝至host侧
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyScalar(lambda);
  aclDestroyTensor(gradInput);

  // 7. 释放device资源
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
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
