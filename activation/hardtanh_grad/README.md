# Hardtanh_grad

##  产品支持情况

| 产品 | 是否支持 |
| :---- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

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

<table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 200px">
  <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>result</td>
      <td>输入</td>
      <td>反向传播过程中上一步输出的梯度，公式中的grad_output。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>正向的输入数据，公式中的self。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>min_val</td>
      <td>属性</td>
      <td>线性范围的下限，公式中的min。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>max_val</td>
      <td>属性</td>
      <td>线性范围的上限，公式中的max。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>计算得到梯度，公式中的res。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

result、grad和out的shape和数据类型需要一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- |---- | ---- |
| aclnn调用 | [test_aclnn_hardtanh_grad](examples/test_aclnn_hardtanh_grad.cpp) | 通过[aclnnHardtanhBackWard](docs/aclnnHardtanhBackward.md)方式调用hardtanh_grad算子。|
