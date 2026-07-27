# SparseFillEmptyRows

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

- 算子功能：对稀疏张量的第0维逐行检查，当某一行没有任何非零元素时，在输出稀疏张量中为该行补充一个位置为`[row, 0, ...]`、取值为`default_value`的元素。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| `indices` | 输入 | 形状为`[N, rank]`的二维张量，每一行表示一个稀疏元素在 dense tensor 中的坐标。 | INT64 | ND |
| `values` | 输入 | 形状为`[N]`的一维张量，表示`indices`中每个稀疏元素对应的值。 | BOOL、COMPLEX128、COMPLEX64、DOUBLE、FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT16、UINT32、UINT64、UINT8 | ND |
| `dense_shape` | 输入 | 形状为`[rank]`的一维张量，表示稀疏张量对应 dense tensor 的形状，`dense_shape[0]`表示行数。 | INT64 | ND |
| `default_value` | 输入 | 标量张量，表示空行回补时写入`y_values`的默认值。 | BOOL、COMPLEX128、COMPLEX64、DOUBLE、FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT16、UINT32、UINT64、UINT8 | ND |
| `y_indices` | 输出 | 回补空行后的稀疏索引，形状为`[N + empty_row_count, rank]`。 | INT64 | ND |
| `y_values` | 输出 | 回补空行后的稀疏值，形状为`[N + empty_row_count]`，数据类型与`values`一致。 | BOOL、COMPLEX128、COMPLEX64、DOUBLE、FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT16、UINT32、UINT64、UINT8 | ND |
| `empty_row_indicator` | 输出 | 形状为`[dense_shape[0]]`的一维布尔张量，`true`表示对应行在输入稀疏张量中为空。 | BOOL | ND |
| `reverse_index_map` | 输出 | 形状为`[N]`的一维张量，`reverse_index_map[i]`表示输入第`i`个元素在输出中的位置。 | INT64 | ND |

## 约束说明

- `indices`必须是二维张量，第二维大小为`rank`。
- `values`的元素数必须与`indices`第一维`N`一致。
- `dense_shape`必须是一维张量，元素数为`rank`。
- `indices[i, 0]`必须满足`0 <= indices[i, 0] < dense_shape[0]`。
- 当`dense_shape[0] == 0`时，`N`必须为0。
- `values`、`default_value`、`y_values`的数据类型需要保持一致。
- 本算子为AI CPU实现。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_sparse_fill_empty_rows](./examples/test_geir_sparse_fill_empty_rows.cpp) | 通过[算子IR](./op_graph/sparse_fill_empty_rows_proto.h)构图方式调用SparseFillEmptyRows算子。 |
