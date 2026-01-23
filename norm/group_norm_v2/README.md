# GroupNormV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR/Ascend 950DT</term>                   |     √     |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √      |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |     √     |


## 功能说明

- 算子功能：计算输入x的组归一化结果out，均值meanOut，以及标准差的倒数rstdOut。

- 计算公式：
  - **GroupNorm:**
  记 $E[x] = \bar{x}$代表$x$的均值，$Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$代表$x$的方差，则

  $$
  \left\{
  \begin{array} {rcl}
  out& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
  meanOut& &= E[x]\\
  rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
  \end{array}
  \right.
  $$
      
## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x</td>
      <td>输入</td>
      <td>out计算公式中的x。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>out计算公式中的gamma。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>out计算公式中的beta。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group</td>
      <td>属性</td>
      <td>x的第1维度分为group组。</td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
     <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td>输入数据格式，支持"NHWC"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>可选属性</td>
      <td>out和rstdOut计算公式中的eps值，默认值为0.00001f。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>可选属性</td>
      <td>标记是否训练场景，true表示训练场景，false表示推理场景，默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>out输出张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>输出</td>
      <td>mean输出张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>输出</td>
      <td>rstd输出张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_group_norm_v2](./examples/test_geir_group_norm_v2.cpp)   | 通过[算子IR](./op_graph/group_norm_v2_proto.h)构图方式调用GroupNormV2算子。 |