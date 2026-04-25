# ApplyCenteredRMSProp

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

- 算子功能：ApplyCenteredRMSProp 是带"中心化"修正的 RMSProp 优化器算子，功能对标 `tf.raw_ops.ApplyCenteredRMSProp`。在 RMSProp 基础上，额外维护一阶梯度指数移动平均 `mg`，并以 `ms - mg^2` 作为方差估计参与归一化，从而获得更稳定的步长。`var`/`mg`/`ms`/`mom` 均为 Ref Tensor，算子执行后**原地更新**。
- 计算公式：

  $$
  \begin{aligned}
  mg_t   &= \rho \cdot mg_{t-1}  + (1 - \rho) \cdot \text{grad}_t \\
  ms_t   &= \rho \cdot ms_{t-1}  + (1 - \rho) \cdot \text{grad}_t^2 \\
  denom_t &= \sqrt{ms_t - mg_t^2 + \epsilon} \\
  mom_t  &= \text{momentum} \cdot mom_{t-1} + \text{lr} \cdot \frac{\text{grad}_t}{denom_t} \\
  var_t  &= var_{t-1} - mom_t
  \end{aligned}
  $$

- 说明：
  - `var`/`mg`/`ms`/`mom` 为 Ref Tensor，与对应的 `*_out` 输出共享存储以实现 inplace 更新。
  - `lr`、`rho`、`momentum`、`epsilon` 为 0-D 或 1 元素 1-D 标量 Tensor。
  - 逐元素独立计算，无跨元素/跨核依赖。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 150px">
<col style="width: 150px">
<col style="width: 420px">
<col style="width: 200px">
<col style="width: 140px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>var</td>
    <td>输入</td>
    <td>公式中的 var，待更新的模型参数（Ref Tensor，原地更新）。shape 与 mg/ms/mom/grad 一致。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mg</td>
    <td>输入</td>
    <td>公式中的 mg，一阶梯度指数移动平均（Ref Tensor，原地更新）。shape 与 var/ms/mom/grad 一致。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ms</td>
    <td>输入</td>
    <td>公式中的 ms，二阶梯度指数移动平均（Ref Tensor，原地更新）。shape 与 var/mg/mom/grad 一致。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mom</td>
    <td>输入</td>
    <td>公式中的 mom，动量项（Ref Tensor，原地更新）。shape 与 var/mg/ms/grad 一致。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>公式中的 lr，学习率。0-D 或 1 元素 1-D Tensor。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rho</td>
    <td>输入</td>
    <td>公式中的 rho，指数衰减系数。0-D 或 1 元素 1-D Tensor。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>输入</td>
    <td>公式中的 momentum，动量系数。0-D 或 1 元素 1-D Tensor。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>公式中的 epsilon，数值稳定项（&gt; 0）。0-D 或 1 元素 1-D Tensor。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>公式中的 grad，当前步的梯度张量。shape 与 var/mg/ms/mom 一致。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var_out</td>
    <td>输出</td>
    <td>更新后的参数，与输入 var 共享存储（inplace 更新）。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mg_out</td>
    <td>输出</td>
    <td>更新后的一阶梯度均值，与输入 mg 共享存储（inplace 更新）。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ms_out</td>
    <td>输出</td>
    <td>更新后的二阶梯度均值，与输入 ms 共享存储（inplace 更新）。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mom_out</td>
    <td>输出</td>
    <td>更新后的动量项，与输入 mom 共享存储（inplace 更新）。</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- 仅支持 <term>Ascend 950PR/Ascend 950DT</term>（arch35 / DAV_3510），不适配其他芯片代际。
- 支持 `float16` 与 `float32` 数据类型，所有输入 Tensor 的 dtype 必须一致。
- `var`、`mg`、`ms`、`mom`、`grad` 五者 shape 必须完全一致，且均为连续排布的 ND Tensor。
- `lr`、`rho`、`momentum`、`epsilon` 必须为 0-D 或 1 元素 1-D 的标量 Tensor。
- 调用方需保证 `epsilon > 0`、`ms - mg^2 + epsilon > 0`；当 `denom == 0` 时 `rsqrt` 输出 Inf/NaN，行为与 PyTorch / TensorFlow 原生实现一致，需由上游调用方规避。
- `var`/`mg`/`ms`/`mom` 为 Ref Tensor，Host aclnn 侧必须显式构造四个占位输出 Tensor（`var_out`/`mg_out`/`ms_out`/`mom_out`），并与各自输入共享 Device 地址以保证 inplace 语义。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn 调用</td>
    <td><a href="./examples/arch35/test_aclnn_apply_centered_rms_prop.cpp">test_aclnn_apply_centered_rms_prop</a></td>
    <td>Ascend 950 上通过 aclnn 两段式接口 <code>aclnnApplyCenteredRMSPropGetWorkspaceSize</code> → <code>aclnnApplyCenteredRMSProp</code> 调用。<code>var_out</code>/<code>mg_out</code>/<code>ms_out</code>/<code>mom_out</code> 需与对应 Ref 输入共享 Device 地址以保证 inplace 更新。</td>
  </tr>
</tbody></table>
