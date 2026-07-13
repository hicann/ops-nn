# MaxPoolV3

## 算子描述

MaxPoolV3 算子实现了二维最大池化（Max Pooling 2D）操作。在输入张量的每个空间窗口内取最大值作为输出，支持自定义 kernel size、stride、padding 和 ceil_mode。

## 支持的产品

- Atlas 训练系列产品（Ascend 910B）

## 约束说明

- 输入张量 x 支持 4 维（NCHW 格式）
- 支持的数据类型：float16, float, bfloat16
- `ksize` 和 `strides` 必须为 4 元素列表，分别对应 N、C、H、W 维度
- pads 为 4 元素列表，对应 [pad_top, pad_bottom, pad_left, pad_right]
- 输出数据类型与输入一致

## 参数说明

| 参数 | 类型 | 输入/输出 | 说明 |
|------|------|-----------|------|
| x | Tensor | 输入 | 4 维输入张量（NCHW 格式） |
| `ksize` | ListInt | 属性 | 池化窗口大小，长度为 4 |
| strides | ListInt | 属性 | 池化步长，长度为 4 |
| pads | ListInt | 属性 | 填充大小，长度为 4，默认 [0,0,0,0] |
| ceil_mode | Bool | 属性 | 是否使用 ceil 模式计算输出尺寸，默认 false |
| y | Tensor | 输出 | 池化后的输出张量 |

## 输出尺寸计算

- H_out = floor((H_in + pad_top + pad_bottom - kH + (ceil_mode ? sH - 1 : 0)) / sH) + 1
- W_out = floor((W_in + pad_left + pad_right - kW + (ceil_mode ? sW - 1 : 0)) / sW) + 1

当 ceil_mode 为 true 时，如果最后一个窗口超出输入边界，会减少一个输出。

## 编译运行

```bash
# 编译算子包
bash build.sh --pkg --soc=ascend910b --experimental --ops=max_pool_v3

# 运行示例
bash build.sh --run_example max_pool_v3 eager cust --experimental
```
