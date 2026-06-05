# aclnnInternVlAddRmsNorm

## 功能说明

aclnnInternVlAddRmsNorm 用于执行 residual add + RMSNorm 融合计算。

计算流程：

residual_out = x + residual

y = residual_out / sqrt(mean(residual_out * residual_out) + epsilon) * gamma

## 接口

第一段接口：

aclnnInternVlAddRmsNormGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *residual,
    const aclTensor *gamma,
    double epsilon,
    const aclTensor *yOut,
    const aclTensor *residualOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)

第二段接口：

aclnnInternVlAddRmsNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)

## 参数说明

x：输入 Tensor。
residual：残差输入 Tensor。
gamma：RMSNorm 权重 Tensor。
epsilon：RMSNorm 防除零参数，默认 1e-6。
yOut：归一化输出 Tensor。
residualOutOut：x + residual 的输出 Tensor。
workspaceSize：workspace 大小。
executor：算子执行器。

## 约束说明

x、residual、yOut、residualOutOut 的 shape 和 dtype 保持一致。
gamma 的元素个数需要与输入最后一维 hidden_size 一致。
支持 FLOAT16、FLOAT、BFLOAT16。
