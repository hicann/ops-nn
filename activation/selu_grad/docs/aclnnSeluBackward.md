# aclnnSeluBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/selu_grad)

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
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

- 接口功能：完成[aclnnSelu](../../selu/docs/aclnnSelu&aclnnInplaceSelu.md)的反向。

- 计算公式：

  计算激活函数的导数：

  $$
  \frac{\partial selu(x)}{\partial x}=\begin{cases} \alpha e^x,x<0 \\1,x\geq 0\end{cases}
  $$

  计算误差对输入的导数：

  $$
  \frac{\partial E}{\partial x}=\frac{\partial E}{\partial y}\frac{\partial selu(x)}{\partial x}
  $$

  其中$y$为输出，$E$为损失函数
  $\alpha$=1.6732632423543772848170429916717

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSeluBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSeluBackward”接口执行计算。

```Cpp
aclnnStatus aclnnSeluBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclTensor* result,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSeluBackward(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnSeluBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 330px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>gradOutput（aclTensor*）</td>
      <td>输入</td>
      <td>表示Selu计算输出的梯度，公式中的∂E/∂X。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape需要与result，gradInput一致。</li></ul></td>
      <td>FLOAT、FLOAT16、INT32、INT8、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>result（aclTensor*）</td>
      <td>输入</td>
      <td>表示Selu计算的正向输出，公式中的y。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、shape需要与gradOutput，gradInput一致。</li></ul></td>
      <td>FLOAT、FLOAT16、INT32、INT8、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>gradInput（aclTensor*）</td>
      <td>输出</td>
      <td>表示Selu计算输入的梯度，公式中的∂E/∂X。</td>
      <td>数据类型、shape需要与gradOutput，result一致。</td>
      <td>FLOAT、FLOAT16、INT32、INT8、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
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

   <!-- npu="910,310p" id7 -->
   - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32、INT8。
   <!-- end id7 -->

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>传入的gradOutput、result、gradInput是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>gradOutput、result、gradInput的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradOutput、result、gradInput的shape不一致。</td>
    </tr>
    <tr>
      <td>gradOutput、result、gradInput的数据类型不满足数据类型推导规则。</td>
    </tr>
  </tbody></table>

## aclnnSeluBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSeluBackwardGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnSeluBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_selu_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> shape = {4, 2};
    void* gradDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* gradients = nullptr;
    aclTensor* outputs = nullptr;
    aclTensor* y = nullptr;

    // SELU 常量
    const float SCALE = 1.0507009873554804f;
    const float ALPHA = 1.6732632423543772f;
    const float SCALE_ALPHA_PRODUCT = SCALE * ALPHA;

    // 构造输入: gradients = 全1, outputs = [-2, -1, 0, 1, 2, 3, -0.5, 0.5]
    std::vector<float> gradHostData = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> outHostData = {-2, -1, 0, 1, 2, 3, -0.5, 0.5};
    std::vector<float> yHostData(8, 0);

    ret = CreateAclTensor(gradHostData, shape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &gradients);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &outputs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, shape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnSeluBackwardGetWorkspaceSize(gradients, outputs, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSeluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSeluBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSeluBackward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), yDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\n=== SeluGrad Results ===\n");
    LOG_PRINT("outputs >  0: y = SCALE * gradients = %.6f * grad\n", SCALE);
    LOG_PRINT("outputs <= 0: y = grad * (outputs + SCALE_ALPHA) = grad * (out + %.6f)\n\n", SCALE_ALPHA_PRODUCT);

    int pass = 0, fail = 0;
    for (int64_t i = 0; i < size; i++) {
        float expected;
        if (outHostData[i] > 0) {
            expected = SCALE * gradHostData[i];
        } else {
            expected = gradHostData[i] * (outHostData[i] + SCALE_ALPHA_PRODUCT);
        }
        bool ok = std::fabs(resultData[i] - expected) < 0.01f;
        if (ok)
            pass++;
        else
            fail++;
        LOG_PRINT("  [%ld] out=%.2f grad=%.2f => NPU=%.6f  expected=%.6f  %s\n", i, outHostData[i], gradHostData[i],
                  resultData[i], expected, ok ? "PASS" : "FAIL");
    }
    LOG_PRINT("\nTotal: %d PASS, %d FAIL\n", pass, fail);

    aclDestroyTensor(gradients);
    aclDestroyTensor(outputs);
    aclDestroyTensor(y);
    aclrtFree(gradDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return fail > 0 ? 1 : 0;
}
```
