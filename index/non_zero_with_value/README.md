# NonZeroWithValue

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：找出2D输入`x`的非零元素，静态max-size一次返回`value`、`index`、`count`三个输出，有效长度由`count`给出。
- 计算说明：
  - `value`：非零元素的值。
  - `index`：非零元素坐标，坐标主序展平（`[2, numel]`，前半段为行号、后半段为列号）。
  - `count`：非零元素个数。
  - 非零判定遵循IEEE语义：nan、±inf计为非零，-0.0视为零；按行主序遍历，结果确定。

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
      <td>待计算的目标张量，严格2D。</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输出</td>
      <td>非零元素的值，数据类型与x保持一致，静态max-size，shape为[row*col]。</td>
      <td>DOUBLE, FLOAT, FLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输出</td>
      <td>非零元素坐标，坐标主序展平，shape为[2*row*col]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>count</td>
      <td>输出</td>
      <td>非零元素个数，shape为[1]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>属性</td>
      <td>标识index排布，仅支持true（坐标主序）。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>控制index输出数据类型，仅支持INT32。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入`x`为严格2D，数据类型支持12类：DOUBLE、FLOAT、FLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL；`value`数据类型与`x`一致，`index`/`count`恒为INT32。
- 输入`x`的元素总数（row×col）不超过INT32_MAX（2³¹-1）：坐标输出`index`为INT32，元素总数超限时坐标无法表示，tiling校验拒绝。
- 输出为静态max-size，有效长度由`count`给出。
- 确定性计算：默认确定性实现。

## 调用说明

| 调用方式   | 样例代码                                                     | 说明                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式调用 | [non_zero_with_value_proto.h](op_graph/non_zero_with_value_proto.h) | 通过算子IR（`REG_OP(NonZeroWithValue)`）构图方式调用NonZeroWithValue算子。 |
