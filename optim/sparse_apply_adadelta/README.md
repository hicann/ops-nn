# SparseApplyAdadelta

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>   |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：SparseApplyAdadelta是AdaDelta优化算法的稀疏版本，用于在训练过程中对Embedding等稀疏参数进行自适应学习率更新。仅更新`indices`指定的行，适用于Embedding层等大规模稀疏参数的梯度更新场景。

- 计算公式：

  记第 $k$ 个索引（$k \in [0, N)$，$N$ 为 `indices` 的长度），参数行号 $i = \text{indices}[k]$。下式中带 $_{old}$ 的下标表示更新前的输入值，$\text{grad}[k]$ 取梯度的第 $k$ 行：

$$
\text{accum}_{new}[i] = \rho \times \text{accum}_{old}[i] + (1 - \rho) \times \text{grad}[k]^2
$$

$$
\text{update} = \frac{\sqrt{\text{accum\_update}_{old}[i] + \epsilon}}{\sqrt{\text{accum}_{new}[i] + \epsilon}} \times \text{grad}[k]
$$

$$
\text{var}_{new}[i] = \text{var}_{old}[i] - \text{update} \times \text{lr}
$$

$$
\text{accum\_update}_{new}[i] = \rho \times \text{accum\_update}_{old}[i] + (1 - \rho) \times \text{update}^2
$$

  未被 `indices` 选中的行保持原值，不参与计算。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>待更新的参数张量，第一维为稀疏维度。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入</td>
      <td>梯度平方的指数移动平均，与var同shape同dtype。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_update</td>
      <td>输入</td>
      <td>更新量平方的指数移动平均，与var同shape同dtype。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率，标量，支持0维标量（shape为{}）或仅含1个元素的1维张量（shape为{1}）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rho</td>
      <td>输入</td>
      <td>衰减率，标量（形态同lr），取值范围[0, 1]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>数值稳定性参数，标量（形态同lr），通常取很小正数（如1e-6）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>梯度张量，第一维N为indices的长度。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引向量，指定需要更新的行。取值范围[0, var的第一维大小)，值必须唯一，不支持越界索引。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>更新后的参数张量，与输入var同shape同dtype，并与输入var共享Device内存（inplace）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输出</td>
      <td>更新后的梯度累积器，与输入accum共享Device内存（inplace）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_update</td>
      <td>输出</td>
      <td>更新后的更新量累积器，与输入accum_update共享Device内存（inplace）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>是否使用锁保护变量更新（NPU上未实现锁机制），默认false。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- var、accum、accum_update三者shape必须一致（逐维相等），且秩至少为1，第一维为稀疏维度。
- grad的第一维（N）等于indices的长度，其余维度与var的后续维度一致。
- indices必须为一维张量，dtype支持int32和int64。
- indices取值须落在[0, var的第一维大小)内，且值必须唯一。
- lr、rho、epsilon为标量，支持0维标量（shape为{}）或仅含1个元素的1维张量（shape为{1}）；三者与var、grad同为float32。
- 仅支持float32数值类型。
- 支持动态shape（`-1`）与动态Rank（`-2`）构图：某个输入Rank未知时跳过形状校验，输出沿用对应输入的具体shape；构建图时仍须满足上表参数形态要求。
- 支持空张量：当indices为空（N=0）时不更新任何行，算子直接返回；var某一维为0时，其元素数为0，同样不执行元素计算。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_sparse_apply_adadelta](./examples/test_geir_sparse_apply_adadelta.cpp) | 在host侧通过GE IR构图方式调用sparse_apply_adadelta算子。 |
