# ScatterElements

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |

## 功能说明

- 算子功能: 将tensor updates中的值按指定的轴和方向以及对应的位置关系逐个替换/累加/累乘至输出tensor中,输出tensor非更新位置的值和输入tensor一致。

- 示例：
  对于一个3D tensor，输出会按照如下的规则进行更新：

  ```bash
  y[indices[i][j][k]][j][k] += updates[i][j][k] # 如果dim == 0 && reduction == 1
  y[i][indices[i][j][k]][k] *= updates[i][j][k] # 如果dim == 1 && reduction == 2
  y[i][j][indices[i][j][k]] = updates[i][j][k]  # 如果dim == 2 && reduction == 0
  ```

  在计算时需要满足以下要求：
  - data、indices和updates的维度数量必须相同。
  - 对于每一个维度d，有indices.size(d) <= updates.size(d)的限制。
  - 对于每一个维度d，如果d != dim,有indices.size(d) <= data.size(d)的限制。
  - dim的值的大小必须在 [-data的维度数量, data的维度数量-1] 之间。
  - data的维度数应该小于等于8。
  - indices的值大小必须在[0, data.size(dim)-1]之间。

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
      <td>data</td>
      <td>输入</td>
      <td>公式中的`data`，Device侧的aclTensor。</td>
      <td>COMPLEX128, COMPLEX64, DOUBLE, FLOAT32, FLOAT16, INT16, INT32, INT64, INT8, QINT32, QINT8, QUINT8, UINT16, UINT32, UINT64, UINT8, BFLOAT16, COMPLEX32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>用来scatter的维度，数据类型为INT64。</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>公式中的`indices`，Device侧的aclTensor。</td>
      <td>INT32、INT64。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>公式中的`updates`，Device侧的aclTensor。</td>
      <td>COMPLEX128, COMPLEX64, DOUBLE, FLOAT32, FLOAT16, INT16, INT32, INT64, INT8, QINT32, QINT8, QUINT8, UINT16, UINT32, UINT64, UINT8, BFLOAT16, COMPLEX32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>输入</td>
      <td>Host侧的字符串，选择应用的reduction操作。</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出。</td>
      <td>COMPLEX128, COMPLEX64, DOUBLE, FLOAT32, FLOAT16, INT16, INT32, INT64, INT8, QINT32, QINT8, QUINT8, UINT16, UINT32, UINT64, UINT8, BFLOAT16, COMPLEX32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_scatter_elements](./examples/test_geir_scatter_elements.cpp) | 通过[算子IR](./op_graph/scatter_elements_proto.h)构图方式调用ScatterElements算子。 |
