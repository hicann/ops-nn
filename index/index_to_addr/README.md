# IndexToAddr

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

- 算子功能：根据块索引、原始矩阵shape、块shape和基地址信息，生成块内每一行对应的地址表。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| `base_addr` | 输入 | 基地址张量，shape为`[2]`。 | INT64、UINT64 | ND |
| `x` | 输入 | 块索引张量，shape为`[2]`，第0个元素为块行索引，第1个元素为块列索引。 | INT64、UINT64 | ND |
| `ori_shape` | 属性 | 原始矩阵shape，长度为2。 | ListInt | - |
| `block_size` | 属性 | 块shape，长度为2。 | ListInt | - |
| `ori_storage_mode` | 属性 | 原始矩阵存储模式，默认`Matrix`。当前仅支持`Matrix`。 | STRING | - |
| `block_storage_mode` | 属性 | 块存储模式，默认`Matrix`。当前仅支持`Matrix`。 | STRING | - |
| `rank_id` | 属性 | rank id，默认0。 | INT | - |
| `dtype` | 属性 | 基础数据类型，默认DT_FLOAT。 | TYPE | - |
| `addrs_table` | 输出 | 地址表张量，shape为`[block_size[0], 4]`。 | INT64、UINT64 | ND |

## 约束说明

- `base_addr`和`x`的数据类型必须一致。
- `base_addr`和`x`必须是一维且元素个数为2的张量。
- `ori_shape`和`block_size`必须是长度为2的ListInt。
- `ori_storage_mode`和`block_storage_mode`当前仅支持`Matrix`。
- `x[0] * block_size[0]`必须小于`ori_shape[0]`，`x[1] * block_size[1]`必须小于`ori_shape[1]`。
- 输出张量元素个数必须不小于`block_size[0] * 4`。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_index_to_addr](./examples/test_geir_index_to_addr.cpp) | 通过[算子IR](./op_graph/index_to_addr_proto.h)构图方式调用IndexToAddr算子。 |
