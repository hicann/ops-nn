# SparseSegmentMean

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：根据`indices`和有序的`segment_ids`从`x`中选取若干行，并对同一个 segment 内的行按第0维求平均。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| `x` | 输入 | 输入张量，rank至少为1。 | FLOAT、DOUBLE、FLOAT16 | ND |
| `indices` | 输入 | 一维张量，表示从`x`第0维选取的行索引。 | INT32、INT64 | ND |
| `segment_ids` | 输入 | 一维张量，元素个数与`indices`一致，值必须非负且按非递减顺序排列。 | INT32、INT64 | ND |
| `y` | 输出 | 输出张量，数据类型与`x`一致，第0维大小为`segment_ids`最后一个元素加1，其余维度与`x`一致。 | FLOAT、DOUBLE、FLOAT16 | ND |

## 约束说明

- `x`、`indices`、`segment_ids`不能为空张量。
- `x`的rank必须大于等于1。
- `indices`与`segment_ids`元素个数必须一致。
- `indices`中的值必须小于`x`第0维大小。
- `segment_ids`必须非负且按非递减顺序排列。
- `x`和`y`的数据类型必须一致。
- 本算子支持AI CPU实现，AI CPU实现支持FLOAT、DOUBLE、FLOAT16。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_sparse_segment_mean](./examples/test_geir_sparse_segment_mean.cpp) | 通过[算子IR](./op_graph/sparse_segment_mean_proto.h)构图方式调用SparseSegmentMean算子。 |
