# MapIndex

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：根据输入张量的层级索引关系计算映射索引，输出为 0 维标量。
- 计算原理：通过 `x`（输入索引）、`data_seq`（数据序列）和可选的 `level_index`（层级索引）之间的层级映射关系，计算目标索引值。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>输入索引，1 维 Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>data_seq</td>
    <td>输入</td>
    <td>数据序列，1 维 Tensor，长度须为 x 长度的整数倍。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>level_index</td>
    <td>可选输入</td>
    <td>层级索引，1 维 Tensor，长度须等于 data_seq_length / x_length。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>transpose</td>
    <td>可选属性</td>
    <td><ul><li>预留属性，仅支持 false。</li><li>传 true 会被拒绝。</li><li>默认值为 false。</li></ul></td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>映射索引结果，0 维标量 Tensor。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- `x` 长度范围：[1, 24000]。
- `data_seq_length / x_length` 上限：256。
- `x` 和 `data_seq` 不支持动态 shape（未知维 -1 / 未知秩 -2 会被拒绝）。
- `level_index` 为未知维（-1）或未知秩（-2）时，跳过长度校验。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_map_index.cpp">test_geir_map_index</a></td>
    <td>本算子仅支持图模式（GE）调用，不提供 aclnn 两段式接口。</td>
  </tr>
</tbody></table>
