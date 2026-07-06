# AveragePoolingGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品|√|

## 功能说明

- 算子功能：计算 NCHW 格式 average_pool2d 的输入梯度。
- 输入 orig_input_shape 指定前向输入 shape [N, C, H, W]，grad_output 为池化输出侧上游梯度，grad_input 为回传到前向输入侧的梯度。
- 支持 kernel_h/kernel_w、stride_h/stride_w、四向 padding、ceil_mode、count_include_pad 和 divisor_override 语义。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| orig_input_shape | 输入 | 前向输入 shape，长度为 4，内容为 [N, C, H, W]。 | int64 | ND |
| grad_output | 输入 | 上游梯度，shape 为 [N, C, H_out, W_out]。 | fp32/fp16 | ND |
| grad_input | 输出 | 输入侧梯度，shape 与 orig_input_shape 一致。 | fp32/fp16 | ND |
| kernel_h/kernel_w | 属性 | 池化窗口高/宽。 | int64_t | - |
| stride_h/stride_w | 属性 | 池化步长高/宽。 | int64_t | - |
| pad_top/pad_bottom/pad_left/pad_right | 属性 | 四向 padding。 | int64_t | - |
| ceil_mode | 属性 | 输出 shape 是否使用 ceil 规则。 | bool | - |
| count_include_pad | 属性 | divisor 是否包含 padding。 | bool | - |
| divisor_override | 属性 | 非 0 时使用该值作为 divisor。 | int64_t | - |

## 约束说明

1. grad_output 与 grad_input 数据类型必须一致，当前支持 fp32、fp16。
2. 当前支持 4D NCHW shape。
3. padding、kernel、stride 需满足 average_pool2d 的常规合法性约束。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_average_pooling_grad.cpp](./examples/test_aclnn_average_pooling_grad.cpp) | 通过[aclnnAveragePoolingGrad](./docs/aclnnAveragePoolingGrad.md)接口调用AveragePoolingGrad算子。 |
