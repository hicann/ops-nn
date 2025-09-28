# CrossV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  Atlas A3 训练系列产品/Atlas A3 推理系列产品   |     √    |
|  Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件     |     √    |

## 功能说明

- 算子功能：用于计算x1和x2张量在维度dim上的向量叉积。
- 计算公式：

  $$
  a \times b = \left|\begin {array}{c}
  i&j&k \\
  x_1&y_1&z_1 \\
  x_1&y_1&z_1 \\
  \end{array}\right|

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
      <td>x1</td>
      <td>输入</td>
      <td>输入tensor。</td>
      <td>FLOAT16、FLOAT32、INT32、INT8、UINT8、INT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>输入tensor。</td>
      <td>FLOAT16、FLOAT32、INT32、INT8、UINT8、INT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>指定运算维度。</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出tensor。</td>
      <td>FLOAT16、FLOAT32、INT32、INT8、UINT8、INT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

x1和x2的维度要大于dim，且x1和x2的dim维大小必须是3。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_cross_v2](./examples/test_geir_cross_v2.cpp)   | 通过[算子IR](./op_graph/cross_v2_proto.h)构图方式调用CrossV2算子。 |