# SparseApplyProximalAdagrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- **算子功能**：对稀疏参数执行ProximalAdagrad优化算法更新。该算法结合了Adagrad（自适应学习率）和FOBOS proximal更新（L1软阈值 + L2权重衰减），专门用于稀疏参数（如大规模Embedding表）的训练更新。仅更新 `indices` 指定的行，其余行保持不变。

- **计算公式**：

对`indices`中每个索引`i`：

$$
\begin{aligned}
\text{accum}_{\text{new}}[i] &= \text{accum}_{\text{old}}[i] + \text{grad}[i]^2 \\
\text{lr}_1 &= \frac{\text{lr}}{\sqrt{\text{accum}_{\text{new}}[i]}} \\
\text{prox\_var} &= \text{var}_{\text{old}}[i] - \text{grad}[i] \times \text{lr}_1 \\
\text{lr}_2 &= \frac{1}{1 + \text{l}_2 \times \text{lr}_1}
\end{aligned}
$$

若 $\text{l}_1 > 0$：
$$
\text{var}_{\text{new}}[i] = \text{sign}(\text{prox\_var}) \times \max(|\text{prox\_var}| - \text{lr}_1 \times \text{l}_1, 0) \times \text{lr}_2
$$

否则（$\text{l}_1 = 0$）：
$$
\text{var}_{\text{new}}[i] = \text{prox\_var} \times \text{lr}_2
$$

> ⚠️ 分母不含epsilon。accum初始值必须非零以避免除零。

## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
<col style="width: 200px">
<col style="width: 170px">
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
      <td>待优化参数张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入</td>
      <td>梯度平方累计张量，与var同shape。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率标量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l1</td>
      <td>输入</td>
      <td>L1正则系数标量（≥0）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2</td>
      <td>输入</td>
      <td>L2正则系数标量（≥0）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>稀疏梯度张量，shape[0]等于len(indices)，尾维度与var一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>稀疏索引向量，指定var/accum中需要更新的行（值必须唯一）。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>更新后的参数张量（inplace语义，与输入var同shape/dtype）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输出</td>
      <td>更新后的梯度平方累计张量（inplace语义，与输入accum同shape/dtype）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>若为True，var/accum更新受锁保护；若为False，行为未定义但可能减少锁竞争。默认False。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- var和accum必须具有相同的shape。
- var至少1维，indices必须为1维向量。
- grad.shape[0] == len(indices)，var.shape[1:] == grad.shape[1:]。
- lr/l1/l2为标量（rank==0或shape==[1]）。
- indices中的值应在[0, var.shape[0])范围内且应唯一。若indices含重复值，结果不可预测（与TensorFlow行为一致）。
- 所有数据输入（var/accum/lr/l1/l2/grad）dtype必须完全一致。
- fp16/bf16内部以fp32计算后cast回原类型。
- accum初始值必须大于0（分母不含epsilon）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式 | [test_geir_sparse_apply_proximal_adagrad](./examples/arch35/test_geir_sparse_apply_proximal_adagrad.cpp) | 通过[算子IR](./op_graph/sparse_apply_proximal_adagrad_proto.h)构图方式调用SparseApplyProximalAdagrad算子。 |
