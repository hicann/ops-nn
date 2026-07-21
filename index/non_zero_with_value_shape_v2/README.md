# NonZeroWithValueShapeV2

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

- 算子功能：根据`NonZeroWithValue`的`count`输入更新`value`和`index`输出shape。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| `value` | 输入 | `NonZeroWithValue`的value输出。 | DOUBLE、FLOAT、FLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL | ND |
| `index` | 输入 | `NonZeroWithValue`的index输出。 | INT32 | ND |
| `count` | 输入 | 非零元素数量。 | INT32 | ND |
| `value` | 输出 | 更新shape后的value输出，shape为`[count[0]]`。 | DOUBLE、FLOAT、FLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL | ND |
| `index` | 输出 | 更新shape后的index输出，shape为`[2, count[0]]`。 | INT32 | ND |

## 约束说明

- `index`和`count`的数据类型必须为INT32。
- `count`输入至少包含1个INT32元素。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_non_zero_with_value_shape_v2](./examples/test_geir_non_zero_with_value_shape_v2.cpp) | 通过[算子IR](./op_graph/non_zero_with_value_shape_v2_proto.h)构图方式调用NonZeroWithValueShapeV2算子。 |
