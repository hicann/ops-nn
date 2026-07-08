# aclnnHardsigmoidBackward

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Ascend 950PR/Ascend 950DT</term>|×|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|√|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|√|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|√|
|<term>Atlas 训练系列产品</term>|√|

## 功能说明

- `aclnnHardsigmoidBackward`：对输入 `gradOutput` 和 `self` 执行 HardSigmoid 反向梯度计算，并将结果写入独立输出 Tensor。
- `experimental/activation/hard_sigmoid_grad_v3` 导出普通版 `aclnnHardsigmoidBackward`，与原始实现保持一致。
- 当 `gradOutput` 和 `self` 的提升 dtype 为 `FLOAT16` 时，内部会进一步提升到 `FLOAT32` 计算。

计算公式：

$$
\operatorname{hardsigmoid\_backward}(gradOutput, self)=
\begin{cases}
\frac{gradOutput}{6}, & -3 < self < 3 \\
0, & \text{otherwise}
\end{cases}
$$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnHardsigmoidBackwardGetWorkspaceSize` 获取执行器和 workspace 大小，再调用第二段接口执行计算。

```cpp
aclnnStatus aclnnHardsigmoidBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnHardsigmoidBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

## aclnnHardsigmoidBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1497px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 247px">
  <col style="width: 300px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gradOutput（aclTensor*）</td>
      <td>输入</td>
      <td>上游梯度张量。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 self、out 完全一致。</li><li>数据类型需要在支持范围内。</li></ul></td>
      <td>FLOAT16、FLOAT32；BFLOAT16 仅支持 Ascend910B 及后续同代 SoC</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>前向输入张量，用于生成 HardSigmoid 反向掩码。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 gradOutput、out 完全一致。</li><li>数据类型需要在支持范围内。</li></ul></td>
      <td>FLOAT16、FLOAT32；BFLOAT16 仅支持 Ascend910B 及后续同代 SoC</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>计算的出参。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 gradOutput、self 完全一致。</li><li>数据类型需要支持从内部计算 dtype cast 回写。</li></ul></td>
      <td>FLOAT16、FLOAT32；BFLOAT16 仅支持 Ascend910B 及后续同代 SoC</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

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
      <td>传入的 gradOutput、self 或 out 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>gradOutput 或 self 的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>gradOutput、self 和 out 的 shape 不一致。</td>
    </tr>
    <tr>
      <td>gradOutput、self 或 out 的维度大于 8。</td>
    </tr>
    <tr>
      <td>提升后的计算 dtype 不能转换为 out 的数据类型。</td>
    </tr>
  </tbody></table>

## aclnnHardsigmoidBackward

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnHardsigmoidBackwardGetWorkspaceSize 获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 实现说明

- 当前实现会先对 `gradOutput` 和 `self` 执行 `Contiguous`，必要时提升 dtype 后调用 `l0op::HardSigmoidGradV3` AscendC kernel 完成主体计算。
- kernel 输出会先 cast 到 `out` 的 dtype，再通过 `ViewCopy` 回写，因此支持非连续输出 Tensor。
- `FLOAT16` 路径会升精度到 `FLOAT32` 计算；`BFLOAT16` 路径按提升规则直接使用 `BFLOAT16`/`FLOAT32` 路径。
- host tiling 采用 512B cache line 对齐的核间切分，并额外限制 `ubFactor <= blockFactor`，避免中等 shape 整块退化到 tail 标量路径。
- full-tile `FLOAT16/FLOAT32` kernel 使用 `Abs(x) < 3` 替代双 compare 判断；tail 继续使用标量安全路径。

## 调用示例

```cpp
#include "aclnnop/aclnn_hardsigmoid_backward.h"

aclnnStatus RunHardsigmoidBackward(const aclTensor *gradOutput,
                                   const aclTensor *self,
                                   aclTensor *out,
                                   aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnHardsigmoidBackwardGetWorkspaceSize(
        gradOutput, self, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }

    ret = aclnnHardsigmoidBackward(workspace, workspaceSize, executor, stream);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```

## 约束说明

- 确定性计算：
  - `aclnnHardsigmoidBackward` 默认确定性实现。

## 调用示例说明

完整样例可参考：
[examples/test_aclnn_hard_sigmoid_grad_v3.cpp](../examples/test_aclnn_hard_sigmoid_grad_v3.cpp)
