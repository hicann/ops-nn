# SparseApplyAdadelta

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：SparseApplyAdadelta是AdaDelta优化算法的稀疏版本，用于在训练过程中对Embedding等稀疏参数进行自适应学习率更新。仅更新`indices`指定的行，适用于Embedding层等大规模稀疏参数的梯度更新场景。

- 计算公式：

$$
\text{accum}[i] = \rho \times \text{accum}[i] + (1 - \rho) \times \text{grad}[i]^2
$$

$$
\text{update} = \sqrt{\frac{\text{accum\_update}[i] + \epsilon}{\text{accum}[i] + \epsilon}} \times \text{grad}[i]
$$

$$
\text{var}[i] = \text{var}[i] - \text{update} \times \text{lr}
$$

$$
\text{accum\_update}[i] = \rho \times \text{accum\_update}[i] + (1 - \rho) \times \text{update}^2
$$

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
      <td>学习率，标量。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rho</td>
      <td>输入</td>
      <td>衰减率，标量，取值范围[0, 1]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>数值稳定性参数，标量，通常很小（如1e-6）。</td>
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
      <td>更新后的参数张量，与输入var同shape同dtype。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输出</td>
      <td>更新后的梯度累积器。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_update</td>
      <td>输出</td>
      <td>更新后的更新量累积器。</td>
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

- var、accum、accum_update三者shape必须一致。
- grad的第一维（N）等于indices的长度，其余维度与var的后续维度一致。
- indices值必须唯一，不支持越界索引。
- 仅支持float32数值类型，indices支持int32和int64。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_sparse_apply_adadelta](./examples/test_geir_sparse_apply_adadelta.cpp) | 在host侧通过GE IR构图方式调用sparse_apply_adadelta算子。 |
