# Gelu

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：高斯误差线性单元激活函数。

- 计算公式：

  $$
  out=GELU(self)=self × Φ(self)=0.5 * self * (1 + tanh( \sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
  $$



## 参数说明

  <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 250px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
    </tr></thead>
  <tbody>
      <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示GELU激活函数的输入，公式中的self。</td>
      <td><ul><li>数据类型必须和out一样。</li><li>shape必须和out一样。</li><li>支持空Tensor。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示GELU激活函数的输出。</td>
      <td><ul><li>数据类型必须和self一致。</li><li>shape必须和self一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>

  </tbody>
  </table>


## 约束说明

- 无