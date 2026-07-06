# aclnnAveragePoolingGrad

## 产品支持情况

|产品|是否支持|
|:--|:--:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

计算 NCHW 格式 average_pool2d 的输入梯度。每个 grad_input[n,c,h,w] 会累加所有覆盖该输入位置的 grad_output[n,c,oh,ow] / divisor(oh,ow)。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用 aclnnAveragePoolingGradGetWorkspaceSize 获取 workspace 大小和执行器，再调用 aclnnAveragePoolingGrad 执行计算。

`cpp
aclnnStatus aclnnAveragePoolingGradGetWorkspaceSize(
    const aclIntArray *origInputShape,
    const aclTensor *gradOutput,
    int64_t kernelH,
    int64_t kernelW,
    int64_t strideH,
    int64_t strideW,
    int64_t padTop,
    int64_t padBottom,
    int64_t padLeft,
    int64_t padRight,
    bool ceilMode,
    bool countIncludePad,
    int64_t divisorOverride,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
`

`cpp
aclnnStatus aclnnAveragePoolingGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
`

## aclnnAveragePoolingGradGetWorkspaceSize

| 参数名 | 输入/输出 | 描述 | 数据类型/格式 |
| ---- | ---- | ---- | ---- |
| origInputShape | 输入 | 前向输入 shape，长度为 4，内容为 [N, C, H, W]。 | aclIntArray |
| gradOutput | 输入 | 上游梯度，shape 为 [N, C, H_out, W_out]。 | FLOAT/FLOAT16，ND |
| kernelH/kernelW | 输入 | 池化窗口高/宽。 | INT64 |
| strideH/strideW | 输入 | 池化步长高/宽。 | INT64 |
| padTop/padBottom/padLeft/padRight | 输入 | 四向 padding。 | INT64 |
| ceilMode | 输入 | 输出 shape 是否使用 ceil 规则。 | BOOL |
| countIncludePad | 输入 | divisor 是否包含 padding。 | BOOL |
| divisorOverride | 输入 | 非 0 时使用该值作为 divisor。 | INT64 |
| out | 输出 | 输入侧梯度，shape 与 origInputShape 一致。 | FLOAT/FLOAT16，ND |
| workspaceSize | 输出 | 返回需要申请的 workspace 大小。 | uint64_t* |
| executor | 输出 | 返回 op 执行器。 | aclOpExecutor** |

## aclnnAveragePoolingGrad

| 参数名 | 输入/输出 | 描述 |
| ---- | ---- | ---- |
| workspace | 输入 | Device 侧 workspace 地址。 |
| workspaceSize | 输入 | workspace 大小。 |
| executor | 输入 | 第一段接口返回的执行器。 |
| stream | 输入 | 执行 Stream。 |

## 约束说明

- 当前支持 FLOAT、FLOAT16。
- 当前支持 4D NCHW shape。
- gradOutput 的 H/W 应与 origInputShape 和池化参数推导出的输出 shape 一致。
