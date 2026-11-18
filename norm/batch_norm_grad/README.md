# BatchNormGrad

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：计算 BatchNorm 的反向传播梯度，输出输入 x 的梯度以及对 scale、offset 的梯度。
- 输入 y_backprop、x 与输出 x_backprop 支持 FLOAT、FLOAT16、BF16，其余输入输出为 FLOAT。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| y_backprop | 输入 | 输出梯度的反向传播值。 | FLOAT、FLOAT16、BF16 | NCDHW、NCHW、NHWC、NDHWC |
| x | 输入 | 前向计算的输入。 | FLOAT、FLOAT16、BF16 | NCDHW、NCHW、NHWC、NDHWC |
| scale | 输入 | 前向计算的缩放参数。 | FLOAT | ND |
| reserve_space_1 | 输入 | BatchNorm 前向保存的中间量：训练模式下为 x 的保存均值，推理模式下为 x 的运行均值。 | FLOAT | ND |
| reserve_space_2 | 输入 | BatchNorm 前向保存的中间量：训练模式下为 x 的逆标准差，推理模式下为 x 的运行方差。 | FLOAT | ND |
| reserve_space_3 | 可选输入 | 前向保留的中间结果，当前不使用、不参与计算。 | FLOAT | ND |
| x_backprop | 输出 | 输入 x 的梯度。 | FLOAT、FLOAT16、BF16 | NCDHW、NCHW、NHWC、NDHWC |
| scale_backprop | 输出 | scale 的梯度。 | FLOAT | ND |
| offset_backprop | 输出 | offset 的梯度。 | FLOAT | ND |
| reserve_space_4 | 输出 | 反向保留的中间结果。 | FLOAT | ND |
| reserve_space_5 | 输出 | 反向保留的中间结果。 | FLOAT | ND |
| epsilon | 属性 | 数值稳定性参数，默认 1e-4。 | FLOAT | - |
| data_format | 属性 | 数据排布，默认 NHWC。 | STRING | - |
| is_training | 属性 | 是否为训练模式，默认 true。 | BOOL | - |
| output_mask | 属性 | 输出掩码，默认 {true, false, false}；仅推理模式有效，决定 x_backprop、scale_backprop、offset_backprop 是否包含实际结果。 | LIST_BOOL | - |

## 约束说明

- 本算子前一层必须是 BatchNorm 算子。
- scale、reserve_space_1、reserve_space_2 为 1D Tensor，shape 必须为 C 通道。
- x 与 y_backprop 的 shape 相同；x_backprop 与 x 的 shape 相同。
- 输入 y_backprop、x 与输出 x_backprop 支持 NCDHW、NCHW、NHWC、NDHWC 四种格式，其余输入与输出为 ND。
- reserve_space_3 为可选输入。
- 产品清单 Data Types 一栏中，仅 Ascend 950 的 AI Core 列出 bfloat16，Atlas A2/A3 的 AI Core 与 AI CPU 均只列 float16、float32。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | - | 通过[算子IR](./op_graph/batch_norm_grad_proto.h)构图方式调用BatchNormGrad算子 |
