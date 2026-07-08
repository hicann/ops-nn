# aclnnSwish

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

- `aclnnSwish`：对输入 Tensor 执行 Swish 激活计算，并将结果写入独立输出 Tensor。
- `experimental/activation/swish_v2` 目录对外导出的 ACLNN 接口名与当前实现保持一致。
- 当前仅导出普通接口，未导出任何 inplace 接口。
- `betaOptional` 为空时使用默认缩放系数 `1.0`。

计算公式：

$$
y = \frac{x}{1 + e^{-\text{scale} \times x}}
$$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnSwishGetWorkspaceSize` 获取执行器和 workspace 大小，再调用第二段接口执行计算。

```cpp
aclnnStatus aclnnSwishGetWorkspaceSize(
    const aclTensor *self,
    const aclScalar *betaOptional,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnSwish(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

## aclnnSwishGetWorkspaceSize

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
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>待进行 Swish 计算的输入张量，公式中的 `x`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 out 完全一致。</li><li>数据类型必须与 out 完全一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>betaOptional（aclScalar*）</td>
      <td>输入</td>
      <td>可选缩放系数。</td>
      <td><ul><li>允许传空，空值时默认使用 1.0。</li><li>非空时需要能转换到 `FLOAT`。</li></ul></td>
      <td>可转换到 FLOAT 的标量类型</td>
      <td>Scalar</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>计算的出参。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 self 完全一致。</li><li>数据类型必须与 self 完全一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32</td>
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
      <td>传入的 self 或 out 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>self 或 out 的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>self 和 out 的数据类型不一致。</td>
    </tr>
    <tr>
      <td>self 和 out 的 shape 不一致。</td>
    </tr>
    <tr>
      <td>self 或 out 的维度大于 8，或 betaOptional 无法转换为 FLOAT。</td>
    </tr>
  </tbody></table>

## aclnnSwish

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
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnSwishGetWorkspaceSize 获取。</td>
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

- 当前实现先对 `self` 做 `Contiguous`，再调用 `l0op::SwishV2` AscendC kernel 完成计算，最后通过 `ViewCopy` 将结果回写到 `out`。
- `betaOptional` 非空时会先转换为 host 侧 `float` 标量，再作为 `scale` 属性传入 kernel。
- `FLOAT16` 和 `BFLOAT16` 路径在 kernel 中升精度到 `float32` 计算后回写。

## 调用示例

```cpp
#include "aclnnop/aclnn_swish.h"

aclnnStatus RunSwish(const aclTensor *self,
                     const aclScalar *betaOptional,
                     aclTensor *out,
                     aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnSwishGetWorkspaceSize(
        self, betaOptional, out, &workspaceSize, &executor);
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

    ret = aclnnSwish(workspace, workspaceSize, executor, stream);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ret;
}
```
