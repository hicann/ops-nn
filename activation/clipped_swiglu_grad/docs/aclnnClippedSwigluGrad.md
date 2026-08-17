# aclnnClippedSwigluGrad

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/clipped_swiglu_grad)

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
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：aclnnClippedSwiglu的反向算子，计算输入x的梯度`gradXOut`。

- 计算流程：
  - 步骤一：合轴与切分（将x合轴并切分为x0和x1）
  - 步骤二：GroupIndex处理（可选）→ 计算有效行数
  - 步骤三：重算正向中间量
  - 步骤四：Clamp反向传播掩码计算
  - 步骤五：SwiGLU反向梯度计算
  - 步骤六：梯度拼接输出
  - 步骤七：GroupIndex置零处理（可选）

- 合轴与切分公式：

  将x基于输入参数`dim`进行合轴，合轴后维度为[pre, cut, after]。其中cut轴为需要切分的轴，pre和after可以等于1。由于after轴的元素为连续存放且计算操作为逐元素的，将cut轴与after轴合并，得到x的维度为[pre, cut]。`gradY`合轴后维度为[pre, cut/2]。

  根据`interleaved`参数对x进行切分：

  当`interleaved`为true时，表示奇偶切分：

  $$
    \mathbf{x}_0[p, h] = \mathbf{x}[p, 2h], \quad h \in [0, \text{cut}/2)
  $$

  $$
    \mathbf{x}_1[p, h] = \mathbf{x}[p, 2h+1], \quad h \in [0, \text{cut}/2)
  $$

  当`interleaved`为false时，表示前后切分：

  $$
    H = \text{cut} / 2
  $$

  $$
    \mathbf{x}_0[p, h] = \mathbf{x}[p, h], \quad h \in [0, H)
  $$

  $$
    \mathbf{x}_1[p, h] = \mathbf{x}[p, h + H], \quad h \in [0, H)
  $$

  其中$p$为pre维索引，$h$为切分后半宽索引。

- GroupIndex处理公式（当提供`groupIndex`时）：

  $$
    \text{trunc} = \sum_{g=0}^{G-1} \text{groupIndex}[g]
  $$

  $$
    \text{validRows} = \min(\text{trunc}, \text{pre})
  $$

  其中：$G$为groupIndex的元素个数，后续所有步骤仅处理前$\text{validRows}$行数据。当不提供`groupIndex`时，$\text{validRows} = \text{pre}$。

- 重算正向中间量公式：

  $$
    \mathbf{x}_0' = \min(\mathbf{x}_0, l)
  $$

  $$
    \mathbf{x}_1' = \min(\max(\mathbf{x}_1, -l), l)
  $$

  $$
    s = \sigma(\alpha \times \mathbf{x}_0') = \frac{1}{1 + e^{-\alpha \times \mathbf{x}_0'}}
  $$

  其中：$\mathbf{x}_0'$为对$\mathbf{x}_0$仅做上界截断，$\mathbf{x}_1'$为对$\mathbf{x}_1$做双向截断，$s$为sigmoid激活值，$\alpha$为`alpha`参数，$l$为`limit`参数。

- Clamp反向传播掩码公式：

  $$
    \text{mask}_{x_0} = \mathbb{I}(\mathbf{x}_0 \leq l)
  $$

  $$
    \text{mask}_{x_1} = \mathbb{I}(-l \leq \mathbf{x}_1 \leq l)
  $$

  其中$\mathbb{I}$为指示函数，满足条件时为1，不满足时为0。

- SwiGLU反向梯度计算公式：

  $$
    \mathbf{grad}_{x_0} = \mathbf{grad}_{y} \times (\mathbf{x}_1' + \text{bias}) \times s \times (1 + \alpha \times \mathbf{x}_0' \times (1 - s)) \times \text{mask}_{x_0}
  $$

  $$
    \mathbf{grad}_{x_1} = \mathbf{grad}_{y} \times \mathbf{x}_0' \times s \times \text{mask}_{x_1}
  $$

  其中：$\mathbf{grad}_{x_0}$为对$\mathbf{x}_0$的梯度，$\mathbf{grad}_{x_1}$为对$\mathbf{x}_1$的梯度，$\mathbf{grad}_{y}$为`gradY`。

- 梯度拼接输出公式：

  当`interleaved`为true时：

  $$
    \mathbf{grad}_{x}[p, 2h] = \mathbf{grad}_{x_0}[p, h]
  $$

  $$
    \mathbf{grad}_{x}[p, 2h+1] = \mathbf{grad}_{x_1}[p, h]
  $$

  当`interleaved`为false时：

  $$
    \mathbf{grad}_{x}[p, h] = \mathbf{grad}_{x_0}[p, h], \quad h \in [0, H)
  $$

  $$
    \mathbf{grad}_{x}[p, h+H] = \mathbf{grad}_{x_1}[p, h], \quad h \in [0, H)
  $$

  其中$\mathbf{grad}_{x}$为`gradXOut`。

- GroupIndex置零处理公式（当提供`groupIndex`时）：

  $$
    \mathbf{grad}_{x}[p, :] = \mathbf{grad}_{x}[p, :] \times \mathbb{I}(p < \text{validRows})
  $$

  最后，重塑输出张量`gradXOut`的维度数量与合轴前的`x`的维度数量一致，各维度大小与`x`相同。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnClippedSwigluGradGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnClippedSwigluGrad"接口执行计算。

```Cpp
aclnnStatus aclnnClippedSwigluGradGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *groupIndexOptional,
    int64_t          dim,
    double           alpha,
    double           limit,
    double           bias,
    bool             interleaved,
    const aclTensor *gradXOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```Cpp
aclnnStatus aclnnClippedSwigluGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnClippedSwigluGradGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 270px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 90px">
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
      <td>gradY（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入gradY，表示正向输出y的梯度。</td>
      <td>维度必须大于0且在入参dim对应维度上为x的一半，其他维度与x一致。数据类型必须与x、gradXOut一致。不支持空Tensor，输入不支持包含±inf或nan。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入x，为正向算子的输入。用于重算正向中间量。</td>
      <td>维度必须大于0且必须在入参dim对应维度上是偶数。不支持空Tensor，输入不支持包含±inf或nan。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入groupIndex，表示分组的情况。</td>
      <td>支持空指针。不为空指针时，维度要求为1维，且元素需大于等于0。第i个元素代表第i组需要处理x的batch数量，元素之和应小于等于合轴前dim维之前各维度的乘积。不支持空Tensor，输入不支持包含±inf或nan。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dim（int64_t）</td>
      <td>输入</td>
      <td>公式中的输入dim，表示对x进行合轴以及切分的维度序号。</td>
      <td>取值范围为[-x.dim(), x.dim()-1]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha（double）</td>
      <td>输入</td>
      <td>公式中的输入alpha，表示变体SwiGlu使用的参数。</td>
      <td>建议为1.702。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>limit（double）</td>
      <td>输入</td>
      <td>公式中的输入limit，表示变体SwiGlu使用的门限值。</td>
      <td>必须大于0，建议为7.0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias（double）</td>
      <td>输入</td>
      <td>公式中的输入bias，表示变体SwiGlu使用的偏差参数。</td>
      <td>建议为1.0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>interleaved（bool）</td>
      <td>输入</td>
      <td>公式中的输入interleaved，表示切分x时是否按奇偶方式切分。</td>
      <td>设置为true表示对x进行奇偶切分，设置为false表示对x进行前后切分。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的输出gradXOut。</td>
      <td>不支持空指针。shape与x一致。不支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
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

  第一段接口会完成入参校验，出现以下场景时报错：

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
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>必选参数gradY/x/gradXOut为nullptr。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入或输出的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>多个输入tensor之间的shape信息不匹配、输入属性不在取值范围（详见参数说明）。</td>
    </tr>
  </tbody></table>

## aclnnClippedSwigluGrad

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnClippedSwigluGradGetWorkspaceSize获取。</td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnClippedSwigluGrad默认为确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_clipped_swiglu_grad.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // x shape [2, 32], grad_y shape [2, 16], grad_x shape [2, 32]
    std::vector<int64_t> xShape = {2, 32};
    std::vector<int64_t> gradYShape = {2, 16};
    std::vector<int64_t> gradXOutShape = {2, 32};
    void* xDeviceAddr = nullptr;
    void* gradYDeviceAddr = nullptr;
    void* gradXOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* gradY = nullptr;
    aclTensor* gradXOut = nullptr;

    std::vector<float> xHostData(64, 1.0f);
    std::vector<float> gradYHostData(32, 1.0f);
    std::vector<float> gradXOutHostData(64, 0.0f);

    int dim = -1;
    float alpha = 1.702;
    float limit = 7.0;
    float bias = 1.0;
    bool interleaved = true;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXOutHostData, gradXOutShape, &gradXOutDeviceAddr, aclDataType::ACL_FLOAT, &gradXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnClippedSwigluGradGetWorkspaceSize(gradY, x, nullptr, dim, alpha, limit, bias, interleaved, gradXOut,
                                                 &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClippedSwigluGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnClippedSwigluGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClippedSwigluGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(gradXOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXOutDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    aclDestroyTensor(x);
    aclDestroyTensor(gradY);
    aclDestroyTensor(gradXOut);
    aclrtFree(xDeviceAddr);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(gradXOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
