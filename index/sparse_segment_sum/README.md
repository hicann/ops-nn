# SparseSegmentSum

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                  |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                 |    ×     |
| <term>Atlas 推理系列产品</term>                         |    √     |
| <term>Atlas 训练系列产品</term>                         |    √     |

## 功能说明

- 算子功能：沿 `segment_ids` 指定的稀疏分段对 `x` 的切片求和。
- 计算公式：`y[segment_ids[i]] += x[indices[i]]`。

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
      <td>输入张量，rank 至少为 1。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、DOUBLE、FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>一维索引张量，用于选择 `x` 第 0 维上的切片。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>segment_ids</td>
      <td>输入</td>
      <td>一维分段 ID 张量，shape 需要与 `indices` 一致，取值需要非负且单调非递减。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，数据类型与 `x` 相同。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、DOUBLE、FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x` 的 rank 需要大于等于 1。
- `indices` 与 `segment_ids` 必须为一维张量，且元素数量一致。
- `segment_ids` 必须按非递减顺序排列。
- `indices` 中的元素必须小于 `x` 第 0 维大小。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| 图模式 | [test_geir_sparse_segment_sum](examples/test_geir_sparse_segment_sum.cpp) | 通过 GE IR 构图方式调用 SparseSegmentSum 算子。 |
