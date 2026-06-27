# aclnnThresholdBackward

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|√|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|√|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|√|
|<term>Atlas 训练系列产品</term>|√|

## 功能说明

- `aclnnThresholdBackward`：对输入 `gradOutput`、`self` 和 `threshold` 执行 ReLU 反向梯度计算，并将结果写入独立输出 Tensor。
- `experimental/activation/relu_grad_v2` 目录对外导出的 ACLNN 接口名与当前实现保持一致。
- 当前实现仅接受 `threshold == 0`，用于对齐 ReLU backward 语义。

计算公式：

$$
\operatorname{threshold\_backward}(gradOutput, self, threshold) =
\begin{cases}
gradOutput, & self > 0 \\
gradOutput, & self = \mathrm{NaN}\ \text{且 dtype 为浮点类型} \\
0, & self \le 0
\end{cases}
$$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnThresholdBackwardGetWorkspaceSize` 获取执行器和 workspace 大小，再调用第二段接口执行计算。

```cpp
aclnnStatus aclnnThresholdBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclScalar *threshold,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnThresholdBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

## aclnnThresholdBackwardGetWorkspaceSize

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
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 self、out 完全一致。</li><li>数据类型必须与 self、out 完全一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>前向输入张量，用于生成 ReLU 掩码。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 gradOutput、out 完全一致。</li><li>数据类型必须与 gradOutput、out 完全一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32、INT8、UINT8、INT32、INT64</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>threshold（aclScalar*）</td>
      <td>输入</td>
      <td>阈值标量。</td>
      <td><ul><li>当前实现仅支持数值为 0。</li><li>建议传入 `INT32` 标量。</li></ul></td>
      <td>INT32</td>
      <td>Scalar</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>计算的出参。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 gradOutput、self 完全一致。</li><li>数据类型必须与 gradOutput、self 完全一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32、INT8、UINT8、INT32、INT64</td>
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
      <td>传入的 gradOutput、self、threshold 或 out 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>gradOutput、self 或 out 的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>gradOutput、self 和 out 的数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradOutput、self 和 out 的 shape 不一致。</td>
    </tr>
    <tr>
      <td>gradOutput、self 或 out 的维度大于 8。</td>
    </tr>
    <tr>
      <td>threshold 不等于 0。</td>
    </tr>
  </tbody></table>

## aclnnThresholdBackward

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
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnThresholdBackwardGetWorkspaceSize 获取。</td>
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

- 当前实现调用 `l0op::ReluGradV2` AscendC kernel 完成主体计算。
- 接口内部会先对输入执行 `Contiguous`，再通过 `ViewCopy` 将结果写回 `out`，因此支持非连续 Tensor。
- 浮点类型遵循 PyTorch `threshold_backward` 在 `threshold == 0` 时的 `NaN` 处理语义；整型路径遵循 `self > 0 ? gradOutput : 0`。

## 调用示例

```cpp
#include "aclnnop/aclnn_threshold_backward.h"

aclnnStatus RunThresholdBackward(const aclTensor *gradOutput,
                                 const aclTensor *self,
                                 const aclScalar *threshold,
                                 aclTensor *out,
                                 aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnThresholdBackwardGetWorkspaceSize(
        gradOutput, self, threshold, out, &workspaceSize, &executor);
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

    ret = aclnnThresholdBackward(workspace, workspaceSize, executor, stream);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```
