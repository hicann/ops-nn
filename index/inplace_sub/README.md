# InplaceSub

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    √     |
| <term>Atlas 推理系列产品</term>                          |    √     |
| <term>Atlas 训练系列产品</term>                          |    √     |

## 功能说明

- 算子功能：根据indices指定的第0维位置，将v中的值从x的对应切片中减去，返回更新后的y。

- 计算逻辑：

  y的初始值与x一致。对于indices中的每个位置i，执行如下更新：

  ```bash
  y[indices[i], ...] = y[indices[i], ...] - v[i, ...]
  ```

- 示例：

  输入x为：

  &emsp;&emsp;[[1, 2, 3],

  &emsp;&emsp;&nbsp;[4, 5, 6],

  &emsp;&emsp;&nbsp;[7, 8, 9]]

  indices = [0, 2]，v为：

  &emsp;&emsp;[[1, 1, 1],

  &emsp;&emsp;&nbsp;[2, 2, 2]]

  算子的计算结果为：

  &emsp;&emsp;[[0, 1, 2],

  &emsp;&emsp;&nbsp;[4, 5, 6],

  &emsp;&emsp;&nbsp;[5, 6, 7]]

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
      <td>x</td>
      <td>输入</td>
      <td>被更新的输入张量，输出y与x的shape相同，格式为ND。</td>
      <td>COMPLEX64、FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>一维索引张量，格式为ND，指定x左侧第0维的位置。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>输入</td>
      <td>更新值张量，数据类型与x相同，第0维大小等于indices的长度，其余维度与x一致，格式为ND。</td>
      <td>COMPLEX64、FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>更新后的张量，shape与x相同，数据类型与x相同，格式为ND。</td>
      <td>COMPLEX64、FLOAT16、FLOAT、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、COMPLEX32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x、v、y的数据类型必须一致。
- x的维度数需大于0，indices必须为一维张量。
- v.shape[0]必须等于indices的长度，v.shape[1:]必须等于x.shape[1:]。
- 当x.shape[0]为0时，indices长度必须为0。
- indices按x的第0维大小进行取模归一。
- 当indices中存在重复值时，输出结果未定义。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_inplace_sub](./examples/test_geir_inplace_sub.cpp) | 通过[算子IR](./op_graph/inplace_sub_proto.h)构图方式调用InplaceSub算子。 |
