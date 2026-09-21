# SparseSegmentMeanGrad

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                   |    √     |
| <term>Atlas A3系列产品</term> |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                  |    ×     |
| <term>Atlas推理系列产品</term>                          |    √     |
| <term>Atlas训练系列产品</term>                          |    √     |

## 功能说明

- 算子功能：计算SparseSegmentMean的反向梯度。正向SparseSegmentMean按`segment_ids`对被`indices`选中的行求均值，反向时需要把每个分段的梯度按该分段被引用的次数取平均后，散射回`indices`指定的输出行。
- 计算公式：

  记`x`的首维长度为$M$（分段数），`indices`与`segment_ids`的长度为$N$，输出`y`的首维长度为`output_dim0`。先统计每个分段被引用的次数：

  $$
  c_s = \sum_{j=0}^{N-1} [\, \mathrm{segment\_ids}_j = s \,], \quad s = 0, 1, \ldots, M-1
  $$

  再按引用次数取平均并散射回输出：

  $$
  y_{i} = \sum_{j=0}^{N-1} [\, \mathrm{indices}_j = i \,] \cdot \frac{x_{\mathrm{segment\_ids}_j}}{\max(c_{\mathrm{segment\_ids}_j},\ 1)}, \quad i = 0, 1, \ldots, \mathrm{output\_dim0}-1
  $$

  其中$x_s$、$y_i$分别表示`x`的第$s$行和`y`的第$i$行，$[\,\cdot\,]$为条件成立时取1、否则取0。未被任何`indices`引用的输出行取值为0。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 400px">
  <col style="width: 250px">
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
      <td>正向SparseSegmentMean传播回来的梯度，对应计算公式中的`x`。维度至少为1维，首维长度为分段数<i>M</i>。</td>
      <td>FLOAT、FLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>正向SparseSegmentMean使用的索引，对应计算公式中的`indices`，指定每个分段梯度散射到`y`的哪一行。1维Tensor，取值范围为[0, `output_dim0`)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>segment_ids</td>
      <td>输入</td>
      <td>正向SparseSegmentMean使用的分段编号，对应计算公式中的`segment_ids`。1维Tensor，元素个数与`indices`相同，取值范围为[0, <i>M</i>)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_dim0</td>
      <td>输入</td>
      <td>输出`y`的首维长度，对应计算公式中的`output_dim0`。标量Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>散射后的梯度，对应计算公式中的`y`。维度至少为1维，shape为`output_dim0`拼接`x`除首维外的各维。</td>
      <td>FLOAT、FLOAT16、DOUBLE</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- `x`与`y`的数据类型必须一致。
- 算子IR中`x`与`y`声明支持BFLOAT16，当前AI CPU实现未覆盖该数据类型。

## 约束说明

- 四个输入均不支持空Tensor。
- `indices`取值超出[0, `output_dim0`)或`segment_ids`取值超出[0, <i>M</i>)时，算子返回参数校验失败。
- 未被任何`indices`引用的输出行会被置0，而不是保持原值。

## 调用说明

| 调用方式   | 调用样例                                                                                | 说明                                                                                         |
| ---------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| 图模式调用 | [test_geir_sparse_segment_mean_grad](./examples/test_geir_sparse_segment_mean_grad.cpp) | 通过[算子IR](./op_graph/sparse_segment_mean_grad_proto.h)构图方式调用SparseSegmentMeanGrad算子。 |
