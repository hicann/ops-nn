# AvgPool1DAvgMatrix

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

- 算子功能：为1DAvgPool反向传播场景生成平均矩阵权重。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| `x` | 输入 | 输入张量，仅使用shape和format信息。 | FLOAT、FLOAT16、INT8、INT16、UINT8、INT32、INT64、DOUBLE | NCHW、NHWC、NC1HWC0 |
| `ksize` | 属性 | 池化窗口大小。 | INT | - |
| `strides` | 属性 | 滑动步长。 | INT | - |
| `pads` | 属性 | 左右padding。 | ListInt | - |
| `ceil_mode` | 属性 | 是否使用向上取整。默认`false`。 | BOOL | - |
| `count_include_pad` | 属性 | 计数是否包含padding。默认`false`。 | BOOL | - |
| `y` | 输出 | 平均矩阵权重输出。 | 与`x`相同，支持FLOAT、FLOAT16、INT8、INT16、UINT8、INT32、INT64、DOUBLE | NC1HWC0 |

## 约束说明

- `pads`至少包含两个元素。
- `strides`不能为0。
- 输入format仅支持`NCHW`、`NHWC`、`NC1HWC0`。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| 图模式调用 | [test_geir_avg_pool1d_avg_matrix](./examples/test_geir_avg_pool1d_avg_matrix.cpp) | 通过[算子IR](./op_graph/avg_pool1d_avg_matrix_proto.h)构图方式调用AvgPool1DAvgMatrix算子。 |
