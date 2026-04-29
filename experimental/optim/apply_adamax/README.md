# ApplyAdamax

## 算子说明

执行 Adamax 优化器的单步参数更新（Adam 的无穷范数变种）。根据当前梯度和一阶矩估计、无穷范数估计，计算参数更新量并**原地**更新权重参数（inplace 语义）。对标 TensorFlow `tf.raw_ops.ApplyAdaMax` 接口。

### 计算公式

$$
\begin{aligned}
m_{t} &= \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot grad \\
v_{t} &= \max(\beta_2 \cdot v_{t-1},\ |grad|) \\
var_{t} &= var_{t-1} - \frac{lr}{1 - \beta_1^t} \cdot \frac{m_{t}}{v_{t} + \epsilon}
\end{aligned}
$$

其中：
- `var`：待更新的权重参数
- `m`：一阶矩估计（梯度指数移动平均）
- `v`：无穷范数估计
- `beta1Power`：$\beta_1^t$（外部传入）
- `lr`：学习率
- `beta1`：一阶矩衰减系数，取值范围 [0, 1)
- `beta2`：二阶矩衰减系数，取值范围 [0, 1)
- `epsilon`：数值稳定常数，必须 >= 0
- `grad`：当前梯度

## 产品支持情况

| 产品 | 是否支持 |
|------|:------:|
| Ascend 950PR / Ascend 950DT（Atlas A5 推理系列，DAV_3510） | √ |
| Atlas A3 训练系列 / Atlas A3 推理系列 | × |
| Atlas A2 训练系列 / Atlas A2 推理系列 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列 | × |
| Atlas 训练系列 | × |

## 调用方式支持

| 调用方式 | 是否支持 |
|---------|:--------:|
| ACLNN 单算子调用 | √ |
| torch_npu 单算子调用 | × |
| torch.compile 入图 | × |
| GE 图模式（静态 shape） | × |
| GE 图模式（动态 shape） | × |

> **说明**：本算子仅支持 ACLNN 单算子调用，暂不支持 GE 图模式、torch_npu 单算子及 torch.compile 入图。因此仅提供 ACLNN 调用示例 `test_aclnn_apply_adamax.cpp`，**不提供 GE IR 示例**。

## 数据类型

| 参数 | 支持 dtype |
|-----|-----------|
| var / m / v / grad | FLOAT16、FLOAT |
| beta1Power / lr / beta1 / beta2 / epsilon (aclScalar) | FLOAT |

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|-------|----------|------|
| var | 输入+输出 (inplace) | 权重参数 Tensor，dtype ∈ {FLOAT16, FLOAT}，1-8 维 ND 格式 |
| m | 输入+输出 (inplace) | 一阶矩 Tensor，shape/dtype 必须与 var 一致 |
| v | 输入+输出 (inplace) | 无穷范数 Tensor，shape/dtype 必须与 var 一致 |
| beta1Power | 输入 | $\beta_1^t$ 标量 |
| lr | 输入 | 学习率标量 |
| beta1 | 输入 | 一阶矩衰减系数标量，取值范围 [0, 1) |
| beta2 | 输入 | 二阶矩衰减系数标量，取值范围 [0, 1) |
| epsilon | 输入 | 数值稳定常数标量，必须 >= 0 |
| grad | 输入 | 梯度 Tensor，shape/dtype 必须与 var 一致 |
| varOut | 输出 | 与 var 共享 Device 内存 |
| mOut | 输出 | 与 m 共享 Device 内存 |
| vOut | 输出 | 与 v 共享 Device 内存 |

## 约束说明

- 本算子仅支持 Ascend 950PR / Ascend 950DT 产品（Atlas A5 推理系列，DAV_3510 架构）。
- `var`、`m`、`v`、`grad` 的 shape 必须**完全相同**，不支持广播。
- `var`、`m`、`v`、`grad` 的 dtype 必须**完全一致**。
- 不支持 0 维标量 Tensor 作为 var / m / v / grad。
- 参数值域：`epsilon >= 0`，`beta1Power ∈ [0, 1)`。
- **Inplace 语义**：`varOut / mOut / vOut` 分别与 `var / m / v` 共享 GM 地址。
- 空 Tensor 处理：元素数为 0 时直接返回成功，不执行计算。
- FP16 精度路径：Kernel 内部提升为 FP32 计算，输出时转回 FP16，以保证精度。

## 目录结构

```
apply_adamax/
├── CMakeLists.txt
├── README.md
├── examples/
│   └── arch35/
│       └── test_aclnn_apply_adamax.cpp  # aclnn 调用示例
├── op_host/
├── op_kernel/
├── op_api/
└── tests/
```

## 调用示例

本算子提供 ACLNN 调用示例，样例代码位于 [`examples/arch35/test_aclnn_apply_adamax.cpp`](examples/arch35/test_aclnn_apply_adamax.cpp)。

具体编译和运行过程可参考 [编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
