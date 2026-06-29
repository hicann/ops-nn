# SparseSegmentSumGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：SparseSegmentSumGrad 是 SparseSegmentSum 的反向传播算子，用于计算稀疏分段求和操作的梯度。

- 计算公式：

$$
output[i] = \sum_{j : indices[j] = i} grad[segment\_ids[j]]
$$

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
      <td>grad</td>
      <td>输入</td>
      <td>输出梯度张量，shape为(N, d1, d2, ...)</td>
      <td>FLOAT16、FLOAT、BF16、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>1-D索引张量，指向原始数据行</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>segment_ids</td>
      <td>输入</td>
      <td>1-D分段标识张量，指向grad行</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_dim0</td>
      <td>输入</td>
      <td>标量，原始数据的第一维大小</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>输入数据的梯度，shape为(output_dim0, d1, d2, ...)</td>
      <td>FLOAT16、FLOAT、BF16、DOUBLE</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- indices 和 segment_ids 必须为 1-D 且长度相同。
- indices 中的值必须在 [0, output_dim0) 范围内。
- segment_ids 中的值必须在 [0, grad.shape[0]) 范围内。
- 多个 segment 指向同一 indices 时使用 atomic add 累加（非确定性）。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir]   | 通过[算子IR](./op_graph/sparse_segment_sum_grad_proto.h)构图方式调用SparseSegmentSumGrad算子。 |
