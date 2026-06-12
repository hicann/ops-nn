# ApplyProximalAdagrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：ApplyProximalAdagrad是结合Adagrad自适应学习率与FOBOS（Forward-Backward Splitting）Proximal近端算法的优化器算子，功能对标`tf.raw_ops.ApplyProximalAdagrad`。基于梯度平方累加器自适应调整学习率，并通过软阈值（L1正则化）与缩放（L2正则化）对模型参数进行原地更新。
- 计算公式：

  $$
  \begin{aligned}
  \text{accum}_t &= \text{accum}_{t-1} + \text{grad}_t^2 \\
  \eta_t &= \frac{\text{lr}}{\sqrt{\text{accum}_t}} \\
  \text{prox}_t &= \text{var}_{t-1} - \eta_t \cdot \text{grad}_t \\
  \text{var}_t &= \frac{\text{sign}(\text{prox}_t)}{1 + \eta_t \cdot \text{l2}} \cdot \max\!\left(|\text{prox}_t| - \eta_t \cdot \text{l1},\ 0\right)
  \end{aligned}
  $$

  当L1 = 0时简化为：

  $$
  \text{var}_t = \frac{\text{prox}_t}{1 + \eta_t \cdot \text{l2}}
  $$

- 说明：
  - `var`（参数）与`accum`（梯度平方累加器）均为Ref Tensor，算子执行后**原地更新**。
  - `lr`、`l1`、`l2`为0-D标量Tensor，分别要求`lr > 0`、`l1 ≥ 0`、`l2 ≥ 0`。
  - 逐元素独立计算，天然确定性，无跨元素/跨核依赖。
  - `float16` / `bfloat16`输入在算子内部Cast为`float32`完成中间计算，再Cast回`float16` / `bfloat16`输出，与PyTorch / TensorFlow实现约定一致。
  - `bfloat16`路径上`lr` / `l1` / `l2`标量读取与`accum` tail padding 1.0构造均通过LocalTensor借道 + Vector Cast实现，规避`bisheng`编译器后端在scalar路径上对bf16类型转换的限制（"not support bf16 type cast"）。

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
    <td>公式中的var，待更新的模型参数（Ref Tensor，原地更新）。shape与accum/grad一致。</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输入</td>
    <td>公式中的accum，梯度平方累加器（Ref Tensor，原地更新）。shape与var/grad一致，要求各元素非负。</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>公式中的lr，学习率。0-D或1元素1-D Tensor，要求lr &gt; 0。</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l1</td>
    <td>输入</td>
    <td>公式中的l1，L1正则化强度。0-D或1元素1-D Tensor，要求l1 ≥ 0。</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2</td>
    <td>输入</td>
    <td>公式中的l2，L2正则化强度。0-D或1元素1-D Tensor，要求l2 ≥ 0。</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>公式中的grad，当前步的梯度张量。shape与var/accum一致。</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>输出</td>
    <td>更新后的参数，与输入var共享存储（inplace更新）。算子仅暴露此1个输出端口。</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- 支持`float32` / `float16` / `bfloat16`三种数据类型；所有tensor（var / accum / lr / l1 / l2 / grad）数据类型必须严格一致。
- `var`、`accum`、`grad`三者shape必须完全一致，且均为连续排布的ND Tensor。
- `lr`、`l1`、`l2`必须为0-D或1元素1-D的标量Tensor。
- 调用方需保证`accum ≥ 0`、`lr > 0`、`l1 ≥ 0`、`l2 ≥ 0`；算子内部不做运行时值域校验。
- `accum + grad^2 == 0`时`rsqrt`输出Inf/NaN，行为与PyTorch / TensorFlow原生实现一致，需由上游调用方规避。
- `var`与`accum`均为Ref Tensor：`var`通过显式输出端口inplace写回，`accum`直接通过input ref端口原地更新，调用方需将其视为可被算子修改的存储。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| 图模式 | [test_geir_apply_proximal_adagrad.cpp](examples/test_geir_apply_proximal_adagrad.cpp) | 通过GE IR图模式构建并运行ApplyProximalAdagrad算子。示例按`var → accum → lr → l1 → l2 → grad`顺序串接6个`Data`/标量输入，输出端口仅声明`var`；`accum`通过input ref端口原地更新。可执行参数`fp16` / `bf16`切换数据类型。 |
