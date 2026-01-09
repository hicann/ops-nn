# Hardtanh_grad

##  产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：激活函数Hardtanh的反向。
- 计算公式：

  $$
  grad\_self_{i} = 
  \begin{cases}
    0,\ \ \ \ \ \ \ if \ \ self_{i}>max \\
    0,\ \ \ \ \ \ \  if\ \ self_{i}<min \\
    1,\ \ \ \ \ \ \ \ \ \ \ \ otherwise \\
  \end{cases}
  $$

  $$
  res_{i} = grad\_output_{i} \times grad\_self_{i}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 280px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>反向传播过程中上一步输出的梯度，公式中的grad_output。</td>
      <td><ul><li>不支持空Tensor。</li><li>grad、self和out的shape一致。</li><li>grad、self和out的数据类型一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>正向的输入数据，公式中的self。</td>
      <td><ul><li>不支持空Tensor。</li><li>grad、self和out的shape一致。</li><li>gradt、self和out的数据类型一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>min_val</td>
      <td>输入</td>
      <td>线性范围的下限，公式中的min。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>max_val</td>
      <td>输入</td>
      <td>线性范围的上限，公式中的max。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>计算得到梯度，公式中的res。</td>
      <td><ul><li>不支持空Tensor。</li><li>grad、self和out的shape一致。</li><li>grad、self和out的数据类型一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

gradOutput、self和out的shape和数据类型需要一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- |---- | ---- |
| aclnn调用 | [test_aclnn_hardtanh_grad](examples/arch35/test_aclnn_hardtanh_grad.cpp) | 通过[aclnnHardtanhBackWard](docs/aclnnHardtanhBackward.md)方式调用hardtanh_grad算子。|
