# aclnnGaussianNllLossGrad

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |

## 功能说明

计算 GaussianNLLLoss 对 `input` 和 `var` 的梯度。设 `d=input-target`、
`v=max(var,eps)`，则 `gradInput=gradOutput*d/v`，
`gradVar=gradOutput*0.5*(1/v-d²/v²)`。`mean` 模式额外乘 `1/N`，
广播 `var` 的梯度归约回原始 shape。`full` 不影响梯度。FLOAT16 和
BFLOAT16 在 FLOAT 中计算后转换回原 dtype。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用
`aclnnGaussianNllLossGradGetWorkspaceSize`，再调用 `aclnnGaussianNllLossGrad`。

```Cpp
aclnnStatus aclnnGaussianNllLossGradGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* input,
    const aclTensor* target,
    const aclTensor* var,
    bool             full,
    float            eps,
    const char*      reduction,
    aclTensor*       gradInput,
    aclTensor*       gradVar,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnGaussianNllLossGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
```

## aclnnGaussianNllLossGradGetWorkspaceSize

- **参数说明**：

  <table class="tg" style="undefined;table-layout: fixed; width: 1450px"><thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th></tr></thead><tbody>
  <tr><td>gradOutput（aclTensor*）</td><td>输入</td><td>上游梯度。</td><td>非空；dtype 与其他张量一致。</td><td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>none 时与 input 相同；sum/mean 时单元素。</td></tr>
  <tr><td>input（aclTensor*）</td><td>输入</td><td>均值预测。</td><td>非空指针。</td><td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>任意维静态 shape。</td></tr>
  <tr><td>target（aclTensor*）</td><td>输入</td><td>目标值。</td><td>非空；可按一个 size-1 维广播。</td><td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>同 input，或同 rank 且一个维度为 1。</td></tr>
  <tr><td>var（aclTensor*）</td><td>输入</td><td>非负方差。</td><td>非空；支持限定广播。</td><td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>同 input、最后一维为 1、缺少最后一维或单元素。</td></tr>
  <tr><td>full（bool）</td><td>输入</td><td>保留的前向一致性属性。</td><td>不影响梯度；默认 false。</td><td>BOOL</td><td>-</td><td>-</td></tr>
  <tr><td>eps（float）</td><td>输入</td><td>方差下限。</td><td>必须大于 0；默认 1e-6。</td><td>FLOAT</td><td>-</td><td>-</td></tr>
  <tr><td>reduction（char*）</td><td>输入</td><td>规约模式。</td><td>none、sum 或 mean；默认 mean。</td><td>STRING</td><td>-</td><td>-</td></tr>
  <tr><td>gradInput（aclTensor*）</td><td>输出</td><td>input 梯度。</td><td>非空；dtype 与 gradOutput 一致。</td><td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>与 input 相同。</td></tr>
  <tr><td>gradVar（aclTensor*）</td><td>输出</td><td>var 梯度。</td><td>非空；dtype 与 gradOutput 一致。</td><td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>与 var 相同。</td></tr>
  <tr><td>workspaceSize（uint64_t*）</td><td>输出</td><td>返回 Device workspace 大小。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>executor（aclOpExecutor**）</td><td>输出</td><td>返回 op 执行器。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody></table>

- **返回值**：

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。
  输入为空指针、dtype 不一致、shape/广播不合法、`eps<=0` 或 reduction 非法时返回错误。

## aclnnGaussianNllLossGrad

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"><thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr></thead><tbody>
  <tr><td>workspace</td><td>输入</td><td>Device workspace 地址。</td></tr>
  <tr><td>workspaceSize</td><td>输入</td><td>由第一段接口获取。</td></tr>
  <tr><td>executor</td><td>输入</td><td>第一段接口返回的执行器。</td></tr>
  <tr><td>stream</td><td>输入</td><td>执行任务的 Stream。</td></tr>
  </tbody></table>

- **返回值**：

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 仅支持 Atlas A2、ND、FLOAT/FLOAT16/BFLOAT16，所有张量 dtype 必须一致。
- `target` 支持同 shape 或恰好一个 size-1 维广播。
- `var` 支持同 shape、最后一维为 1、缺少最后一维或单元素。
- `eps>0`，`var` 值非负；clamp 对梯度透明。
- `none` 的 gradOutput 与 input 同 shape；`sum`/`mean` 的 gradOutput 为单元素。
- 输出分别严格匹配 input 和 var shape；动态未知维在 tiling 前必须具体化。
- 当前实现不申请用户 workspace，框架仍按两段式接口返回实际 workspace 大小。

## 调用示例

具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)，
完整样例见 [test_aclnn_gaussian_nll_loss_grad.cpp](../examples/test_aclnn_gaussian_nll_loss_grad.cpp)。

```Cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus ret = aclnnGaussianNllLossGradGetWorkspaceSize(
    gradOutput, input, target, var, false, 1e-6f, "mean",
    gradInput, gradVar, &workspaceSize, &executor);
if (ret == ACL_SUCCESS) {
    ret = aclnnGaussianNllLossGrad(workspace, workspaceSize, executor, stream);
}
```
