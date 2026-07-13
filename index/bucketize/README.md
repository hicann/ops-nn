# Bucketize

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

- 算子功能：根据属性`boundaries`为输入张量中的每个元素计算所属区间下标。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| `x` | 输入 | 任意shape的待分桶输入张量。 | INT32、INT64、FLOAT、DOUBLE | ND |
| `boundaries` | 属性 | 升序排列的分桶边界。 | ListFloat | - |
| `dtype` | 属性 | 输出张量类型，支持INT32、INT64。默认INT32。 | TYPE | - |
| `right` | 属性 | `true`表示右插入，`false`表示左插入。默认`true`。 | BOOL | - |
| `y` | 输出 | 与`x`shape相同的分桶结果。 | INT32、INT64 | ND |

## 约束说明

- `boundaries`必须是升序。
- 输出shape与输入`x`保持一致。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_bucketize](./examples/test_geir_bucketize.cpp) | 通过[算子IR](./op_graph/bucketize_proto.h)构图方式调用Bucketize算子。 |
