# QuantMatmulReduceSum


##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
 

## 功能说明

- 算子功能：完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出
- 计算公式：

  $$
  out = \sum_{i=0}^{batch}(x1_i @ x2_i) * x1Scale * x2Scale
  $$
  其中 $x1$, $x2$, $out$ 分别是维度为 $(batch, M, K)$, $(batch, K, N)$ 和 $(M, N)$ 的矩阵。$x1Scale$ 和 $x2Scale$ 分别是维度为 $(M,)$ 和 $(N,)$ 的向量。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>x1</td>
      <td>输入</td>
      <td>矩阵乘运算中的左矩阵。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>矩阵乘运算中的右矩阵。</td>
      <td>INT8</td>
      <td>NZ</td>
    </tr>
    <tr>
      <td>x1Scale</td>
      <td>输入</td>
      <td>对矩阵乘结果进行行缩放的一维向量。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2Scale</td>
      <td>输入</td>
      <td>对矩阵乘结果进行列缩放的一维向量。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>结果矩阵。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 不支持空tensor。
- 左右矩阵不支持非连续tensor。


