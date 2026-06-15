# aclnnHardsigmoidV2 与 aclnnInplaceHardsigmoidV2

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

- `aclnnHardsigmoidV2`：对输入 Tensor 执行 HardSigmoidV2 计算，并将结果写入独立输出 Tensor。
- `aclnnInplaceHardsigmoidV2`：对输入 Tensor 原地执行 HardSigmoidV2 计算，结果直接覆盖输入内存。
- `experimental/activation/hard_sigmoid_v2` 目录对外导出的 ACLNN 接口名与当前实现保持一致。

计算公式：

$$
\operatorname{hard\_sigmoid}(x)=
\begin{cases}
0, & x \le -3 \\
\frac{x}{6} + \frac{1}{2}, & -3 < x < 3 \\
1, & x \ge 3
\end{cases}
$$

其中 `INT32` 路径严格对齐当前 AscendC kernel 的整数语义：

$$
\frac{\min(\max(x + 3, 0), 6)}{6} = \min(\max(x - 2, 0), 1)
$$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnHardsigmoidV2GetWorkspaceSize` 或 `aclnnInplaceHardsigmoidV2GetWorkspaceSize` 获取执行器和 workspace 大小，再调用第二段接口执行计算。

```cpp
aclnnStatus aclnnHardsigmoidV2GetWorkspaceSize(
    const aclTensor *self,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnHardsigmoidV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

```cpp
aclnnStatus aclnnInplaceHardsigmoidV2GetWorkspaceSize(
    const aclTensor *self,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnInplaceHardsigmoidV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

## aclnnHardsigmoidV2GetWorkspaceSize

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
      <td>待进行 HardSigmoid 计算的输入张量。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 out 完全一致。</li><li>数据类型必须与 out 完全一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32、INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>计算的出参。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 self 完全一致。</li><li>数据类型必须与 self 完全一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32、INT32</td>
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
      <td>传入的 self、out、workspaceSize 或 executor 是空指针。</td>
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
      <td>self 或 out 的维度大于 8。</td>
    </tr>
  </tbody></table>

## aclnnHardsigmoidV2

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
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnHardsigmoidV2GetWorkspaceSize 获取。</td>
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

## aclnnInplaceHardsigmoidV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1355px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 297px">
  <col style="width: 108px">
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
      <td>输入/输出</td>
      <td>原地计算的输入输出张量。</td>
      <td><ul><li>支持空Tensor。</li><li>原地计算后 shape 和 dtype 不变。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32、INT32</td>
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

## aclnnInplaceHardsigmoidV2

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
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnInplaceHardsigmoidV2GetWorkspaceSize 获取。</td>
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

- 当前实现先对输入做参数校验。
- 对非连续输入会先走 `Contiguous`。
- kernel 计算完成后，对非连续输出通过 `ViewCopy` 回写。
- `FLOAT16` 路径直接在半精度上计算。
- `BFLOAT16` 路径在 kernel 中升精度到 `float32` 计算后回写。
- `INT32` 路径走自定义 AscendC kernel 的整数语义，不复用原生 `HardSigmoidV2(alpha, beta)` 实现。

## 调用示例

完整示例见 [../examples/test_aclnn_hard_sigmoid_v2.cpp](../examples/test_aclnn_hard_sigmoid_v2.cpp)。
