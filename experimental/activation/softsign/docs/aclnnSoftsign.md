# aclnnSoftsign

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | x |
| <term>Atlas 推理系列产品</term> | x |
| <term>Atlas 训练系列产品</term> | x |

## 功能说明

完成Softsign激活函数计算。Softsign是一种类似于tanh的激活函数，但其收敛速度为多项式级别而非指数级别。

计算公式：

  $$
  out_i = \frac{self_i}{1 + |self_i|}
  $$

其中：
- $self_i$ 为输入张量的第 $i$ 个元素。
- $out_i$ 为输出张量的第 $i$ 个元素。
- 输出值域为 $(-1, 1)$。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnSoftsignGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSoftsign"接口执行计算。

```cpp
aclnnStatus aclnnSoftsignGetWorkspaceSize(
  const aclTensor  *self,
  aclTensor        *out,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnSoftsign(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnSoftsignGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>self（const&nbsp;aclTensor*）</td>
      <td>输入</td>
      <td>输入张量，对应公式中self。</td>
      <td><ul><li>支持空Tensor。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>输出张量，对应公式中out。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型需要与self一致。</li><li>shape需要与self一致。</li></ul></td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

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
      <td>self或out存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>self的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>out的数据类型与self不一致。</td>
    </tr>
    <tr>
      <td>out的shape与self不一致。</td>
    </tr>
  </tbody></table>

## aclnnSoftsign

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
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnSoftsignGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- aclnnSoftsign默认确定性实现。
- Softsign为逐元素计算，不涉及广播行为，out的shape必须与self完全一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl_rt.h"
#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "aclnn_softsign.h"

#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto __ret = (expr);                                                                            \
        int32_t __code = static_cast<int32_t>(__ret);                                                   \
        if (__code != 0) {                                                                              \
            fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, __code); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)

int32_t main()
{
    // 1. 初始化ACL环境
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclnnInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    // 2. 准备输入数据
    const std::vector<float> inputData = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    const std::vector<int64_t> shape = {static_cast<int64_t>(inputData.size())};
    const int64_t elementCount = static_cast<int64_t>(inputData.size());
    const size_t bufferSize = elementCount * sizeof(float);

    // 3. 分配Device内存，创建Tensor
    void* inputDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&inputDeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* inputTensor = aclCreateTensor(
        shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
        shape.data(), shape.size(), inputDeviceMem);

    void* outputDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&outputDeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* outputTensor = aclCreateTensor(
        shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
        shape.data(), shape.size(), outputDeviceMem);

    // 4. 拷贝输入数据到Device
    CHECK_ACL(aclrtMemcpy(inputDeviceMem, bufferSize, inputData.data(),
                          bufferSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 5. 调用两段式aclnn接口
    // 第一段：获取workspace大小，创建executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnSoftsignGetWorkspaceSize(inputTensor, outputTensor,
                                            &workspaceSize, &executor));

    // 分配workspace（如需要）
    void* workspaceDeviceMem = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspaceDeviceMem, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // 第二段：执行计算
    CHECK_ACL(aclnnSoftsign(workspaceDeviceMem, workspaceSize, executor, stream));

    // 6. 同步Stream，拷贝结果回Host
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<float> outputData(elementCount, 0.0f);
    CHECK_ACL(aclrtMemcpy(outputData.data(), bufferSize, outputDeviceMem,
                          bufferSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // 7. 打印结果
    // 预期输出: [-0.6667, -0.5000, -0.3333, 0.0000, 0.3333, 0.5000, 0.6667, 0.7500]
    for (int64_t i = 0; i < elementCount; i++) {
        printf("softsign(%.1f) = %.4f\n", inputData[i], outputData[i]);
    }

    // 8. 释放资源
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    CHECK_ACL(aclrtFree(inputDeviceMem));
    CHECK_ACL(aclrtFree(outputDeviceMem));
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtFree(workspaceDeviceMem));
    }
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclnnFinalize());

    return 0;
}
```
