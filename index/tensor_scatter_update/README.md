# TensorScatterUpdate

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：根据 `indices` 指定的位置，将 `updates` 中的数据更新到输入张量 `x` 的对应位置，生成新的输出张量 `y`。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1350px"><colgroup>
  <col style="width: 120px">
  <col style="width: 180px">
  <col style="width: 340px">
  <col style="width: 560px">
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
      <td>待更新的输入张量，提供更新前的基准数据。</td>
      <td>FLOAT16、FLOAT、DOUBLE、INT64、INT32、UINT8、UINT16、UINT32、UINT64、INT8、INT16、BOOL、COMPLEX64、COMPLEX128、QINT8、QUINT8、QINT16、QUINT16、QINT32、BFLOAT16、STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引张量，用于指定 `x` 中需要被更新的元素或切片位置。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>更新数据张量，按 `indices` 指定的位置写入 `x`；数据类型与 `x` 相同。</td>
      <td>与x相同</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出张量，表示将 `updates` 按 `indices` 更新到 `x` 后得到的结果。</td>
      <td>与x相同</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 无。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_tensor_scatter_update](./examples/test_geir_tensor_scatter_update.cpp) | 通过[算子IR](./op_graph/tensor_scatter_update_proto.h)构图方式调用TensorScatterUpdate算子。 |
