# InplaceApplyRMSProp

## 产品支持情况

| 产品 | 是否支持 |
|:-----------------------------------------|:------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

`InplaceApplyRMSProp`是RMSProp自适应学习率优化器的参数更新算子。算子
维护梯度的滑动均方根和动量缓冲区，在每次迭代中更新`var`、`ms`和
`mom`三个状态张量。三个显式输出分别返回同名输入的更新结果。

本算子当前仅提供AscendC GE图模式通路，不提供独立的`aclnn`接口。

计算公式如下：

$$
\begin{aligned}
\text{ms}_t &= \rho \cdot \text{ms}_{t-1} + (1 - \rho) \cdot \text{grad}^2 \\
\text{mom}_t &= \text{momentum} \cdot \text{mom}_{t-1} +
\frac{\text{lr} \cdot \text{grad}}{\sqrt{\text{ms}_t + \epsilon}} \\
\text{var}_t &= \text{var}_{t-1} - \text{mom}_t
\end{aligned}
$$

其中`lr`为学习率，`rho`为衰减系数，`momentum`为动量系数，`epsilon`
为数值稳定性常数。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:---|:---|:---|:---|:---|
| `var` | 输入/输出（inplace） | 待更新的权重张量；shape、dtype与`ms/mom/grad`一致。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| `ms` | 输入/输出（inplace） | 梯度平方滑动平均状态；调用方应保证非负，实现不做数值校验。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| `mom` | 输入/输出（inplace） | 动量缓冲区。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| `lr` | 输入 | 学习率，shape为`{1}`的一维ND标量Tensor。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| `rho` | 输入 | 衰减系数，shape为`{1}`的一维ND标量Tensor；建议取值 `[0, 1)`，实现不做数值校验。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| `momentum` | 输入 | 动量系数，shape为`{1}`的一维ND标量Tensor。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| `epsilon` | 输入 | 数值稳定性常数，shape为`{1}`的一维ND标量Tensor。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| `grad` | 输入 | 当前梯度张量；shape、dtype与`var`一致。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| `use_locking` | 属性 | 是否加锁更新，默认 `false`；当前实现仅作兼容占位。 | BOOL | - |
| `var` | 输出（inplace） | 更新后的参数，对应输入`var`。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| `ms` | 输出（inplace） | 更新后的均方滑动平均，对应输入`ms`。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| `mom` | 输出（inplace） | 更新后的动量缓冲区，对应输入`mom`。 | FLOAT、FLOAT16、BFLOAT16 | ND |

## 约束说明

- `var`、`ms`、`mom`、`grad`的shape、dtype必须完全一致，支持rank1-8；调用方应保证`ms`非负，实现不做数值校验，负值可能使平方根计算产生NaN。
- 空Tensor通过含0维的ND形态表示。
- `lr`、`rho`、`momentum`、`epsilon`必须为shape{1}的ND标量张量，dtype与`var`一致；`rho`建议取值[0, 1)，实现不做范围校验。
- `epsilon`在计算域按FP32处理；当其值小于等于`FLT_MIN`或为NaN时，kernel使用`FLT_MIN`作为下限。
- 支持 `FLOAT`、`FLOAT16`、`BFLOAT16`；FP16/BF16路径在FP32域完成计算后再转换回目标dtype。
- 支持空Tensor（`numel == 0`），空Tensor路径不执行更新。
- `use_locking`不改变计算结果，当前实现不引入额外互斥锁。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:---|:---|:---|
| 图模式 | [test_geir_inplace_apply_rms_prop](./examples/test_geir_inplace_apply_rms_prop.cpp) | 通过 [算子IR](./op_graph/inplace_apply_rms_prop_proto.h) 构图调用。 |

当前不提供ACLNN、PyTorch、ONNX或融合规则接口；TensorFlow原生RMSProp
接口与本算子的三显式输出原型不一致，因此不作为有效调用通路。
