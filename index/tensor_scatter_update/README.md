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

- 算子功能：稀疏更新算子，根据索引更新张量的指定元素或切片。它接受一个输入张量、索引张量和更新值张量，并输出更新后的张量。常用于局部修改、稀疏梯度更新或稀疏数据操作。
- 计算公式：
  设输入张量为$X$，索引集合为$I=\{i_k\}$，更新值为$U=\{u_k\}$，输出张量为$Y$。对任意位置$j$：

  $$
  Y_j =
  \begin{cases}
  u_k, & \text{if } j = i_k \\
  X_j, & \text{otherwise}
  \end{cases}
  $$

  当位置$j$在索引集合中时使用更新值替换，否则保持原张量的值不变。
  其中，$k$表示第$k$个更新项，$i_k$表示该更新项对应的更新位置，$u_k$表示该更新项对应的更新值。

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
      <td>公式中的X，待更新的输入张量，提供更新前的基准数据。</td>
      <td>FLOAT16、FLOAT、DOUBLE、INT64、INT32、UINT8、UINT16、UINT32、UINT64、INT8、INT16、BOOL、COMPLEX64、COMPLEX128、QINT8、QUINT8、QINT16、QUINT16、QINT32、BFLOAT16、STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>公式中的索引集合I，用于指定输入张量X中需要被更新的位置。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>公式中的更新值集合U，表示写入索引集合I对应位置的更新值，数据类型与x相同，shape与indices相同。</td>
      <td>FLOAT16、FLOAT、DOUBLE、INT64、INT32、UINT8、UINT16、UINT32、UINT64、INT8、INT16、BOOL、COMPLEX64、COMPLEX128、QINT8、QUINT8、QINT16、QUINT16、QINT32、BFLOAT16、STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的Y，表示更新后的输出张量，数据类型与x相同，shape与x相同。</td>
      <td>FLOAT16、FLOAT、DOUBLE、INT64、INT32、UINT8、UINT16、UINT32、UINT64、INT8、INT16、BOOL、COMPLEX64、COMPLEX128、QINT8、QUINT8、QINT16、QUINT16、QINT32、BFLOAT16、STRING</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_tensor_scatter_update](./examples/test_geir_tensor_scatter_update.cpp) | 通过[算子IR](./op_graph/tensor_scatter_update_proto.h)构图方式调用TensorScatterUpdate算子。 |
