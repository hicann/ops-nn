# GemmV2


##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：计算α乘以A与B的乘积，再与β和input C的乘积求和。
- 计算公式：

  $$
  out=α(A @ B) + βC
  $$

  其中，$op(A)$，$op(B)$ 和 $op(C)$ 分别是维度为 $(M, K)$, $(K, N)$ 和 $(M, N)$的矩阵。$α$，$β$是一维向量。

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
      <td>a</td>
      <td>输入</td>
      <td>矩阵乘运算中的左矩阵。</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>输入</td>
      <td>矩阵乘运算中的右矩阵。</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>与a、b矩阵乘结果相乘的一维向量。</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>与input c相乘的一维向量。</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c</td>
      <td>输出</td>
      <td>输入input和输出，进行原地累加。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 不支持空tensor。
- 支持连续tensor，[非连续tensor](../../docs/zh/context/非连续的Tensor.md)只支持转置场景。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用  | [参考示例算子调用](../../../examples/add_example/examples/test_geir_add_example.cpp) | 通过[算子IR](op_graph/gemm_v2_proto.h)等方式调用GemmV2算子。 |