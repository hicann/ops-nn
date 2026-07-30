# aclnnSwigluGroupGrad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：SwigluGroupGrad算子实现SwiGLU激活函数反向传播计算。用于计算输入梯度`grad_x`和权重梯度`grad_weight`。
- 计算公式：

  步骤一 GroupIndex处理（可选）→ 计算trunc：
  $$\text{trunc} = \sum_{g=0}^{G-1} \text{groupIndex}[g]$$

  其中$G$为MoE专家分组数，`grad_x`仅保留前$\text{trunc}$行的有效梯度；`grad_weight`不应用该行掩码。

  步骤二 输入切分（将x切分为x0和x1）：
  $$\mathbf{x}_0[t, h] = \mathbf{x}[t, h], \quad h \in [0, H)$$

  $$\mathbf{x}_1[t, h] = \mathbf{x}[t, h + H], \quad h \in [0, H)$$

  步骤三 Clamp处理（可选）：
  $$\mathbf{x}_0'[t, h] = \min(\mathbf{x}_0[t, h], c)$$

  $$\mathbf{x}_1'[t, h] = \min(\max(\mathbf{x}_1[t, h], -c), c)$$

  其中$c$为`clamp_limit`。

  步骤四 SwiGLU反向传播计算：

  前向传播回顾：
  $$\text{Swish}(\mathbf{x}_0) = \mathbf{x}_0 \cdot \sigma(\mathbf{x}_0)$$

  $$\mathbf{y}_{\text{swiglu}} = \text{Swish}(\mathbf{x}_0) \cdot \mathbf{x}_1$$

  Sigmoid计算：
  $$\sigma(\mathbf{x}_0') = \frac{1}{1 + e^{-\mathbf{x}_0'}}$$

  SiLU计算：
  $$\text{SiLU}(\mathbf{x}_0') = \mathbf{x}_0' \cdot \sigma(\mathbf{x}_0')$$

  SiLU梯度：
  $$\frac{d\text{SiLU}}{d\mathbf{x}_0'} = \sigma(\mathbf{x}_0') \cdot \left(1 + \mathbf{x}_0' \cdot (1 - \sigma(\mathbf{x}_0'))\right)$$

  输入梯度计算：
  $$\mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \mathbf{x}_1'[t, h] \cdot \frac{d\text{SiLU}}{d\mathbf{x}_0'}[t, h]$$

  $$\mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \text{SiLU}(\mathbf{x}_0'[t, h])$$

  其中$\mathbf{grad}_{y_0}$为处理后的梯度输入：如果提供了weight，则$\mathbf{grad}_{y_0} = \mathbf{grad}_{\text{output}} \cdot \mathbf{weight}$；如果未提供weight，则$\mathbf{grad}_{y_0} = \mathbf{grad}_{\text{output}}$。

  步骤五 Weight梯度计算（可选）：
  $$\mathbf{grad}_{\text{weight}}[t] = \sum_{h=0}^{H-1} \mathbf{grad}_{\text{output}}[t, h] \cdot \mathbf{y}_{\text{origin}}[t, h]$$

  其中$\mathbf{y}_{\text{origin}}$为SwiGLU前向输出；存在weight时，该输出已乘weight。沿最后一维（H维度）求和。当提供`groupIndex`时，`grad_weight`仍按完整行计算，不应用$\mathbb{I}(t < \text{trunc})$掩码。

  步骤六 Clamp反向传播（可选）：
  $$\mathbf{m}_{x_0}[t, h] = \mathbb{I}(\mathbf{x}_0[t, h] < c)$$

  $$\mathbf{m}_{x_1}[t, h] = \mathbb{I}(-c < \mathbf{x}_1[t, h] < c)$$

  $$\mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{x_0}[t, h] \cdot \mathbf{m}_{x_0}[t, h]$$

  $$\mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{x_1}[t, h] \cdot \mathbf{m}_{x_1}[t, h]$$

  步骤七 梯度拼接与GroupIndex处理：
  $$\mathbf{grad}_x[t, h] = \begin{cases}
  \mathbf{grad}_{x_0}[t, h] & h \in [0, H) \\
  \mathbf{grad}_{x_1}[t, h-H] & h \in [H, 2H)
  \end{cases}$$

  当提供`groupIndex`时，仅前$\text{trunc}$行的梯度有效：
  $$\mathbf{grad}_x[t, :] = \mathbf{grad}_x[t, :] \cdot \mathbb{I}(t < \text{trunc})$$

- 关键特性：在MoE推理反向传播中，不同专家可能处理不同数量的token，groupIndex允许动态调整处理范围。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnSwigluGroupGradGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSwigluGroupGrad"接口执行计算。

```cpp
aclnnStatus aclnnSwigluGroupGradGetWorkspaceSize(
  const aclTensor *gradY,
  const aclTensor *x,
  const aclTensor *weightOptional,
  const aclTensor *yOriginOptional,
  const aclTensor *groupIndexOptional,
  float            clampLimit,
  aclTensor       *gradXOut,
  aclTensor       *gradWeightOutOptional,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnSwigluGroupGrad(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnSwigluGroupGradGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 350px">
  <col style="width: 220px">
  <col style="width: 100px">
  <col style="width: 150px">
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
      <td>gradY（aclTensor*）</td>
      <td>输入</td>
      <td>梯度输出张量，来自下游层的梯度。</td>
      <td><ul><li>shape=[T, H]或[B, S, H]。</li><li>支持空Tensor，空Tensor时直接返回。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2或3 ([T, H]或[B, S, H])</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>前向传播的输入张量。</td>
      <td><ul><li>shape=[T, 2H]或[B, S, 2H]。</li><li>最后一维必须为2H。</li><li>支持空Tensor，空Tensor时直接返回。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2或3 ([T, 2H]或[B, S, 2H])</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>MoE权重张量。</td>
      <td><ul><li>shape=[T, 1]或[B, S, 1]。</li><li>必须与yOriginOptional同时提供。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2或3 ([T, 1]或[B, S, 1])</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOriginOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>SwiGLU前向输出；存在weight时，该输出已乘weight。</td>
      <td><ul><li>shape=[T, H]或[B, S, H]。</li><li>必须与weightOptional同时提供且与gradY的shape一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2或3 ([T, H]或[B, S, H])</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>GroupIndex张量，动态核分配。</td>
      <td><ul><li>shape=[G]，G > 0。</li><li>数据类型为INT64。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1 ([G])</td>
      <td>√</td>
    </tr>
    <tr>
      <td>clampLimit（float）</td>
      <td>输入</td>
      <td>Clamp阈值。</td>
      <td>取值范围≥0.0。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut（aclTensor*）</td>
      <td>输出</td>
      <td>输入梯度张量。</td>
      <td><ul><li>shape=[T, 2H]或[B, S, 2H]，与x一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2或3 ([T, 2H]或[B, S, 2H])</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeightOutOptional（aclTensor*）</td>
      <td>输出（可选）</td>
      <td>权重梯度张量。</td>
      <td><ul><li>当weightOptional和yOriginOptional同时提供时输出。</li><li>shape=[T, 1]或[B, S, 1]。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2或3 ([T, 1]或[B, S, 1])</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>必选参数gradY、x、gradXOut为nullptr。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>clampLimit不满足≥0（包括NaN）。</td>
    </tr>
    <tr>
      <td>x最后一维不等于gradY最后一维的2倍。</td>
    </tr>
    <tr>
      <td>gradY与x的前导维度不匹配。</td>
    </tr>
    <tr>
      <td>参数shape不合法。</td>
    </tr>
    <tr>
      <td>参数数据类型不匹配。</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupGrad

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
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnSwigluGroupGradGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnSwigluGroupGrad默认确定性实现。
- H必须大于0。
- x最后一维必须为偶数（2H），gradY最后一维为H，与x最后一维的一半对应。
- gradY与x的前导维度必须一致，且二者均为二维或三维Tensor。
- weight与yOrigin必须同时提供或同时为空；weight的前导维度需与gradY一致，且最后一维为1；yOrigin的shape需与gradY一致。
- groupIndex非空时必须是一维非空Tensor。
- gradY、x、yOrigin、gradXOut数据类型必须一致（FLOAT、FLOAT16或BFLOAT16）。
- weight、gradWeightOut必须为FLOAT类型。
- groupIndex必须为INT64类型。
- clampLimit必须≥0.0，clampLimit=0表示不启用Clamp反向传播掩码。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<term>Ascend 950PR/Ascend 950DT</term>：

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group_grad.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, aclFormat format)
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

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. 初始化Device和Stream
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 创建输入tensor
    int64_t T = 2;
    int64_t H = 256;
    int64_t twoH = 2 * H;

    std::vector<int64_t> gradYShape = {T, H};
    std::vector<int64_t> xShape = {T, twoH};
    std::vector<int64_t> weightShape = {T, 1};
    std::vector<int64_t> yOriginShape = {T, H};

    void *gradYDeviceAddr = nullptr;
    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *yOriginDeviceAddr = nullptr;
    aclTensor *gradY = nullptr;
    aclTensor *x = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *yOrigin = nullptr;

    std::vector<uint16_t> hostGradY(GetShapeSize(gradYShape), 1);
    std::vector<uint16_t> hostX(GetShapeSize(xShape), 1);
    std::vector<float> hostWeight(GetShapeSize(weightShape), 1.0f);
    std::vector<uint16_t> hostYOrigin(GetShapeSize(yOriginShape), 1);

    ret = CreateAclTensor(hostGradY, gradYShape, &gradYDeviceAddr, aclDataType::ACL_BF16, &gradY,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostX, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostWeight, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostYOrigin, yOriginShape, &yOriginDeviceAddr, aclDataType::ACL_BF16, &yOrigin,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 创建输出tensor
    std::vector<int64_t> gradXOutShape = {T, twoH};
    std::vector<int64_t> gradWeightOutShape = {T, 1};

    void *gradXOutDeviceAddr = nullptr;
    void *gradWeightOutDeviceAddr = nullptr;
    aclTensor *gradXOut = nullptr;
    aclTensor *gradWeightOut = nullptr;

    std::vector<uint16_t> hostGradXOut(GetShapeSize(gradXOutShape), 0);
    std::vector<float> hostGradWeightOut(GetShapeSize(gradWeightOutShape), 0.0f);

    ret = CreateAclTensor(hostGradXOut, gradXOutShape, &gradXOutDeviceAddr, aclDataType::ACL_BF16, &gradXOut,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostGradWeightOut, gradWeightOutShape, &gradWeightOutDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradWeightOut, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 4. 获取workspace大小
    float clampLimit = 0.0f;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    ret = aclnnSwigluGroupGradGetWorkspaceSize(gradY, x, weight, yOrigin, nullptr, clampLimit, gradXOut,
                                               gradWeightOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 5. 申请workspace并执行计算
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSwigluGroupGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 6. 释放资源
    aclDestroyTensor(gradY);
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(yOrigin);
    aclDestroyTensor(gradXOut);
    aclDestroyTensor(gradWeightOut);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(yOriginDeviceAddr);
    aclrtFree(gradXOutDeviceAddr);
    aclrtFree(gradWeightOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
