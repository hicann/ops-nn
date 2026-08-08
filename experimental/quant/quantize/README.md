# Quantize

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/experimental/quant/quantize)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
| <term>Atlas 推理系列产品</term>    |     ×    |
| <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

对输入张量做量化：`y = round(x / scales + zero_points)`，再按 `dtype` 属性饱和转换为定点输出。
本实现为 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>（ascend910b，DAV_2201 标准编程模型）的原生
AscendC kernel，落在 `experimental/`，复用源仓 `quant/quantize` 的 SoC 无关 aclnn/L0 接口层（`aclnnQuantize`）。

## 支持范围（首版）
- 模式：per-tensor、per-channel（axis 语义）。
- x ∈ {FLOAT, FLOAT16, BFLOAT16}；scales ∈ {FLOAT, FLOAT16, BFLOAT16}（FLOAT16 由 aclnn 层 Cast 成 FLOAT 后下发，kernel 侧只有 FLOAT/BFLOAT16）；zero_points ∈ {INT8, UINT8, INT32, BFLOAT16, 缺省}；y ∈ {INT8, UINT8, INT32}。
- scales 或 zero_points 为 BFLOAT16 时，x、scales、zero_points（若传入）必须同时为 BFLOAT16；其余组合自由。
- 属性：`dtype`（必选，torch.qint8/quint8/qint32）、`axis`（可选，默认 1）。
- 固定 RoundMode=round-to-nearest-even（CAST_RINT）、DivMode=div、SqrtMode=none；输出做饱和转换。

## 不支持（后续扩展）
- fp8 输出（HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN，910b 无 fp8）、per-head、per-channel-nddma、FP32 zero_points。

## aclnn 接口
两段式：`aclnnQuantizeGetWorkspaceSize(x, scales, zeroPoints, dtype, axis, out, &workspaceSize, &executor)` → `aclnnQuantize(...)`。
详见 `docs/aclnnQuantize.md`。

## 构建
```bash
bash build.sh --pkg --experimental --soc=ascend910b --ops=quantize -j16
```
