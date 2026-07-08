# ApplyCenteredRMSProp

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

- 算子功能：实现带"中心化"修正的RMSProp优化器更新步骤，用于深度学习训练阶段的模型参数更新。在标准RMSProp基础上，额外维护一阶梯度的指数移动平均（mg），以 `ms - mg²` 作为方差估计替代原始二阶矩 `ms`，提供更稳定的自适应学习率。
- 计算公式：

  **Step 1 — 一阶矩更新**：

  $$
  mg_t = \rho \cdot mg_{t-1} + (1 - \rho) \cdot g_t
  $$

  **Step 2 — 二阶矩更新**：

  $$
  ms_t = \rho \cdot ms_{t-1} + (1 - \rho) \cdot g_t^2
  $$

  **Step 3 — 方差估计**：

  $$
  \text{variance}_t = ms_t - mg_t^2
  $$

  **Step 4 — 分母计算（epsilon加在sqrt内部）**：

  $$
  \text{denom}_t = \sqrt{\text{variance}_t + \epsilon}
  $$

  **Step 5 — 动量更新（momentum > 0时）**：

  $$
  mom_t = \mu \cdot mom_{t-1} + lr \cdot \frac{g_t}{\text{denom}_t}
  $$

  $$
  var_t = var_{t-1} - mom_t
  $$

  **Step 6 — 直接更新（momentum == 0时）**：

  $$
  var_t = var_{t-1} - lr \cdot \frac{g_t}{\text{denom}_t}
  $$

  其中：
  - `var`：待更新的模型参数
  - `mg`：梯度一阶矩的指数移动平均
  - `ms`：梯度二阶矩的指数移动平均
  - `mom`：动量缓冲
  - `g_t`：当前步梯度
  - `lr`：学习率
  - `ρ`（rho）：衰减系数
  - `μ`（momentum）：动量系数
  - `ε`（epsilon）：数值稳定性常数

> **注意**：epsilon加在sqrt内部（`√(x + ε)`），与TensorFlow `tf.raw_ops.ApplyCenteredRMSProp`语义一致。与PyTorch的 `√x + ε` 存在细微数值差异。

## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 300px">
<col style="width: 250px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>var</td>
    <td>输入</td>
    <td>待更新的模型参数。支持空Tensor。shape与mg/ms/mom/grad一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mg</td>
    <td>输入</td>
    <td>梯度一阶矩的指数移动平均。支持空Tensor。shape与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ms</td>
    <td>输入</td>
    <td>梯度二阶矩的指数移动平均。支持空Tensor。shape与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mom</td>
    <td>输入</td>
    <td>动量缓冲。支持空Tensor。shape与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率。必须为0-D标量Tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rho</td>
    <td>输入</td>
    <td>衰减系数。必须为0-D标量Tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>输入</td>
    <td>动量系数。必须为0-D标量Tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>数值稳定性常数。必须为0-D标量Tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前步梯度。支持空Tensor。shape与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>varOut</td>
    <td>输出</td>
    <td>更新后的模型参数，与var共享存储（inplace）。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mgOut</td>
    <td>输出</td>
    <td>更新后的梯度一阶矩，与mg共享存储（inplace）。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>msOut</td>
    <td>输出</td>
    <td>更新后的梯度二阶矩，与ms共享存储（inplace）。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momOut</td>
    <td>输出</td>
    <td>更新后的动量缓冲，与mom共享存储（inplace）。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- 默认确定性实现。
- var、mg、ms、mom、grad五个ND Tensor的shape必须完全一致。
- lr、rho、momentum、epsilon必须为0-D标量Tensor。
- 所有输入和输出的数据类型必须相同（FLOAT、FLOAT16或BFLOAT16）。
- varOut/mgOut/msOut/momOut分别与var/mg/ms/mom共享存储，实现原地更新（inplace语义）。
- 不支持稀疏梯度。
- FLOAT16/BFLOAT16输入时，内部走FP32混合精度计算路径，最终Cast回原始dtype。
- fp16上溢边界：grad值接近fp16最大值65504时，内部FP32计算路径可防止中间溢出，但最终Cast回fp16时若结果超出范围则饱和截断。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE IR 图模式调用</td>
    <td><a href="./examples/test_geir_apply_centered_rms_prop.cpp">test_geir_apply_centered_rms_prop</a></td>
    <td>构图并运行单算子子图，9输入 / 4输出（var/mg/ms/mom inplace更新）。</td>
  </tr>
</tbody></table>
