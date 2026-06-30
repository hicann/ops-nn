# SparseApplyAdagradV2

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

- 算子功能：SparseApplyAdagradV2是稀疏版本的AdagradV2优化器算子，根据稀疏梯度和索引向量原地更新参数变量和累加器。

- 计算公式：

$$
\text{accum}[\text{index}] \mathrel{+}= \text{grad}[i] \times \text{grad}[i] \quad (\text{when update\_slots=True})
$$

$$
\text{var}[\text{index}] \mathrel{-}= \text{lr} \times \frac{\text{grad}[i]}{\sqrt{\text{accum}[\text{index}]} + \epsilon}
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
      <td>参数变量，shape (N, C)</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入</td>
      <td>梯度平方累加器，shape (N, C)</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率标量</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>常数因子标量，避免除零</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>稀疏梯度，shape (M, C)</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>稀疏索引向量，shape (M)</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>更新后的参数变量，shape (N, C)</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输出</td>
      <td>更新后的累加器，shape (N, C)</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>是否在更新时加锁（NPU场景下通常忽略）</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>update_slots</td>
      <td>属性</td>
      <td>是否更新累加器（True时accum+=grad^2）</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- var 和 accum 的 shape 必须相同
- lr 和 epsilon 必须为标量（shape 为空或 [1]）
- indices 必须为一维向量
- grad 的第一维必须与 indices 的第一维相同，其余维度必须与 var 的对应维度相同
- var、accum、lr、epsilon、grad 的 dtype 必须相同
- indices 的 dtype 为 int32
- 越界索引（< 0 或 >= var 行数）将被跳过，不报错

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_sparse_apply_adagrad_v2.cpp">test_geir_sparse_apply_adagrad_v2</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
