# aclnnSoftsignBackward

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：完成Softsign激活函数的反向梯度计算。

- 计算公式：

  $$
  output_i = \frac{gradients_i}{(1 + |features_i|)^2}
  $$

  其中：
  - $gradients$ 为上游梯度张量。
  - $features$ 为Softsign前向函数的输入特征张量。
  - $output$ 为计算得到的梯度输出张量。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnSoftsignBackwardGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSoftsignBackward"接口执行计算。

```cpp
aclnnStatus aclnnSoftsignBackwardGetWorkspaceSize(
  const aclTensor  *gradients,
  const aclTensor  *features,
  aclTensor        *output,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnSoftsignBackward(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnSoftsignBackwardGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 150px">
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
      <td>gradients（aclTensor*）</td>
      <td>输入</td>
      <td>表示上游梯度张量，对应公式中gradients。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型需要与features的数据类型一致。</li><li>shape需要与features的shape完全相同。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>features（aclTensor*）</td>
      <td>输入</td>
      <td>表示Softsign前向函数的输入特征张量，对应公式中features。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型需要与gradients的数据类型一致。</li><li>shape需要与gradients的shape完全相同。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output（aclTensor*）</td>
      <td>输出</td>
      <td>表示计算得到的梯度输出张量，对应公式中output。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型需要与输入的数据类型一致。</li><li>shape需要与输入的shape相同。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

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
      <td>gradients、features、output存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>gradients或features的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradients与features的数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradients与features的shape不相同。</td>
    </tr>
  </tbody></table>

## aclnnSoftsignBackward

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnSoftsignBackwardGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- aclnnSoftsignBackward默认确定性实现。
- gradients和features的数据类型必须一致，且仅支持FLOAT、FLOAT16、BFLOAT16。
- gradients和features的shape必须完全相同，不支持广播。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_softsign_backward.h"

// 辅助宏
#define CHECK_RET(cond, return_expr) \
    do { if (!(cond)) { return_expr; } } while (0)
#define LOG_PRINT(message, ...) \
    do { printf(message, ##__VA_ARGS__); } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t s = 1;
    for (auto i : shape) s *= i;
    return s;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) strides[i] = shape[i+1] * strides[i+1];
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    // 2. 构造输入/输出 tensor
    std::vector<int64_t> shape = {4, 4, 4};
    int64_t numElements = GetShapeSize(shape);

    std::vector<float> gradHost(numElements, 1.0f);
    std::vector<float> featHost(numElements);
    for (int64_t i = 0; i < numElements; i++) featHost[i] = i * 0.1f - 3.0f;
    std::vector<float> outHost(numElements, 0.0f);

    aclTensor *gradients = nullptr, *features = nullptr, *out = nullptr;
    void *gradDev = nullptr, *featDev = nullptr, *outDev = nullptr;
    CreateAclTensor(gradHost, shape, &gradDev, aclDataType::ACL_FLOAT, &gradients);
    CreateAclTensor(featHost, shape, &featDev, aclDataType::ACL_FLOAT, &features);
    CreateAclTensor(outHost,  shape, &outDev,  aclDataType::ACL_FLOAT, &out);

    // 3. 调用第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    aclnnSoftsignBackwardGetWorkspaceSize(gradients, features, out, &workspaceSize, &executor);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 4. 调用第二段接口
    aclnnSoftsignBackward(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    // 5. 释放资源
    aclDestroyTensor(gradients);
    aclDestroyTensor(features);
    aclDestroyTensor(out);
    aclrtFree(gradDev);
    aclrtFree(featDev);
    aclrtFree(outDev);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
