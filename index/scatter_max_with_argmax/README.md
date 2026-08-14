# ScatterMaxWithArgmax

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：沿第0维执行scatter max操作，同时记录每个位置最大值来自哪个update(argmax 索引)。

- 计算公式：

给定:

```bash
  x:       [N, D1, D2, ...] float32   # 用于确定输出 shape，不参与计算
  indices: [M] int32                   # scatter 索引，0 <= indices[i] < N
  updates: [M, D1, D2, ...] float32   # scatter 源数据

计算分为三个阶段:

  # 阶段1：初始化
  y      = full(x.shape, negative_large_value)  # 初始化为float32负大值，仅作为内部哨兵
  argmax = full(x.shape, M)                     # 初始化为M，表示“无更新”

  # 阶段2：scatter max
  for i in range(M):
      idx = indices[i]
      for each element position p in (D1, D2, ...):
          if updates[i][p] > y[idx][p]:        # 严格大于(GT 比较)
              y[idx][p]      = updates[i][p]
              argmax[idx][p] = i

  # 阶段3：FillZeroForUncovered
  for each output position q:
      if argmax[q] == M:
          y[q] = 0
```

输出:
```bash
  y:      [N, D1, D2, ...] float32   # scatter max 结果，未被任何index覆盖的位置为0
  argmax: [N, D1, D2, ...] int32     # 最大值来源索引(0~M-1)，未被指到位置为 M
```

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
      <td>输出shape参考张量，仅用于确定输出shape，不参与计算。N为第0维大小。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>scatter索引张量，1D shape=[M]，取值范围 [0, N)，M为索引数量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>scatter源数据张量，shape[0] = M，shape[1:]与x.shape[1:]相同。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>scatter max结果，shape与x相同。未被任何index覆盖的位置最终输出0。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>输出</td>
      <td>最大值来源索引(0~M-1)，shape与x相同。未被指到位置为M。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x支持1D ~ 8D。
- `indices` 必须为1D，shape = [M]。
- `updates.shape[0]` 必须等于 `indices.shape[0]`(M一致)。
- `updates.shape[1:]` 必须等于 `x.shape[1:]`(除第0维外其余维度一致)。
- `x.shape[0]` 必须 >= max(indices)(索引不越界)。
- 仅支持float32/int32数据类型组合。
- 确定性保证：相同输入产生相同输出(GT严格大于比较，相等时保留先到索引)。
- 未被任何index覆盖的位置，最终y输出0、argmax输出M；内部初始化负大值会在FillZeroForUncovered阶段被改写为0。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_scatter_max_with_argmax](examples/arch35/test_geir_scatter_max_with_argmax.cpp) | 通过[算子IR](./op_graph/scatter_max_with_argmax_proto.h)构图方式调用ScatterMaxWithArgmax算子。 |
