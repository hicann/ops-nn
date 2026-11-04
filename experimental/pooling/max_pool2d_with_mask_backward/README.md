# MaxPool2dWithMaskBackward

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| Atlas 800T A2 | √ |
| Atlas 300V Pro | √ |

## 功能说明

MaxPool2dWithMaskBackward 是 MaxPool2dWithMask 的反向算子。它根据正向输出
`indices` 中记录的最大值位置，将上游梯度 `gradOutput` 累加回输入梯度
`gradInput`，未被最大值位置选中的输入元素梯度为 0。

对于输出位置 $(n,c,oh,ow)$，设 `indices` 解码得到的最大值位置为
$(n,c,ih,iw)$，则：

$$
\operatorname{gradInput}(n,c,ih,iw)
\mathrel{+}= \operatorname{gradOutput}(n,c,oh,ow)
$$

当池化窗口重叠时，同一输入位置可能接收多个输出位置的梯度。

## 参数说明

| 类别 | 名称 | shape | 数据类型 | 格式 |
| ---- | ---- | ----- | -------- | :--: |
| 输入 | `gradOutput` | `[N, C, H_out, W_out]` | float32、float16；910B 支持 bfloat16 | NCHW |
| 输入 | `self` | `[N, C, H_in, W_in]` | float32、float16；910B 支持 bfloat16 | NCHW |
| 输入 | `indices` | `[N, C, kH*kW, (ceil(H_out*W_out/16)+1)*2*16]` | int8 | NCHW |
| 属性 | `kernelSize` | 长度 1 或 2 | int64 | - |
| 属性 | `stride` | 长度 0、1 或 2 | int64 | - |
| 属性 | `padding` | 长度 1 或 2 | int64 | - |
| 属性 | `dilation` | 长度 1 或 2 | int64 | - |
| 属性 | `ceilMode` | 标量 | bool | - |
| 输出 | `gradInput` | 与 `self` 一致 | 与 `gradOutput` 一致 | NCHW |

`indices` 必须是本文对应的 MaxPool2dWithMask 正向输出，不能在 Host 侧重新编码。

## 约束说明

- 输入和输出的 dtype 必须一致；Atlas 300V Pro 支持 float32、float16，Atlas 800T A2 额外支持 bfloat16。
- `gradInput` 的 shape、dtype 和 format 必须与 `self` 一致。
- `gradOutput` 的 H/W 必须与正向输出 shape 一致。
- `kernelSize` 中每个值必须大于 0。
- `stride` 为空时取 `kernelSize`，非空时每个值必须大于 0。
- `padding` 表示对称填充，必须非负且不大于对应方向的 `kernelSize / 2`。
- `dilation` 当前仅支持 1。
- 输入数据暂不支持 NaN 和 -Inf。
- 重叠窗口按累加语义计算；默认采用非确定性实现，可通过 `aclrtCtxSetSysParamOpt` 开启确定性计算。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| ACLNN 调用 | [test_aclnn_max_pool2d_with_mask_backward.cpp](examples/test_aclnn_max_pool2d_with_mask_backward.cpp) | 使用固定输入、索引和预期梯度，通过两段式接口验证非重叠、重叠累加和 padding 场景。 |

接口原型、参数说明和返回值见
[aclnnMaxPool2dWithMaskBackward](docs/aclnnMaxPool2dWithMaskBackward.md)。

该示例仅调用反向接口，固定索引用于独立冒烟测试，不依赖前向算子包。
实际业务应使用匹配的前向算子输出 `indices`。

## 编译与部署

在 ops-nn 仓库根目录执行：

```bash
# Atlas 800T A2
bash build.sh --pkg --experimental --soc=ascend910b \
    --ops=max_pool2d_with_mask_backward --vendor_name=maxpool2d -j8

# Atlas 300V Pro
bash build.sh --pkg --experimental --soc=ascend310p \
    --ops=max_pool2d_with_mask_backward --vendor_name=maxpool2d_310p -j8
```

生成的安装包位于 `build_out/`。安装后执行对应 vendor 包提供的 `set_env.sh`，
再运行调用样例。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ------ | ------ | -------- | -------- | -------- |
| 叶明豪 | 华中科技大学 | MaxPool2dWithMaskBackward | 2026/10/22 | MaxPool2dWithMaskBackward 算子适配与测试 |
