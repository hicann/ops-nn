# Where

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                         |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

算子功能：找出张量`x`中非零或True元素的位置，设张量`x`的维度为D，非零元素的个数为N，则返回`y`的shape为D * N，每一列表示一个非零元素的位置坐标。

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
      <td>x</td>
      <td>输入</td>
      <td>输入张量。</td>
      <td>INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64<br>
          QINT8、QUINT8、QINT32、FLOAT16、FLOAT、DOUBLE、BOOL<br>
          COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输入张量x中非零元素或True的位置张量。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_where](./examples/test_geir_where.cpp)   | 通过[算子IR](./op_graph/where_proto.h)构图方式调用where算子。 |