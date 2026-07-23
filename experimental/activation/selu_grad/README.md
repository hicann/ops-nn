# SeluGrad

## 产品支持情况

| 产品 | 是否支持 |
|:--|:--:|
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |
| 其他产品 | × |

## 功能说明

SeluGrad 计算 SELU（Scaled Exponential Linear Unit）激活函数的反向梯度。输入分别为上游梯度
`gradients` 和 SELU 前向输出 `outputs`，输出为输入梯度 `y`。

$$
y =
\begin{cases}
\text{gradients} \times \text{scale}, & \text{outputs} \ge 0 \\
\text{gradients} \times (\text{outputs} + \text{scale} \times \alpha), & \text{outputs} < 0
\end{cases}
$$

其中：

- $\alpha = 1.6732632423543772848170429916717$
- $\text{scale} = 1.0507009873554804934193349852946$
- $\text{scale} \times \alpha = 1.7580993408473768599402175208123$

比较条件采用数值逻辑比较 `outputs < 0`，不是二进制位模式比较。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | Shape |
|:--|:--:|:--|:--|:--:|:--|
| gradients | 输入 | 反向传播的上游梯度 | FLOAT、FLOAT16、BFLOAT16、INT32、INT8、UINT8 | ND | 与 outputs、y 一致 |
| outputs | 输入 | SELU 前向输出 | 与 gradients 一致 | ND | 与 gradients、y 一致 |
| y | 输出 | SELU 反向梯度 | 与 gradients 一致 | ND | 与 gradients、outputs 一致 |

通过 `aclnnSeluBackward` 调用时，公开接口支持 FLOAT、FLOAT16、BFLOAT16、INT32、INT8 和 UINT8。

## 实现说明

- Host 侧根据数据类型选择 TilingKey，并依据输入规模、AIV 核数和 UB 容量动态计算分核与切块参数。
- FP32 直接使用 FP32 计算。
- FP16 直接使用 FP16 向量计算。
- BF16 和 INT32 转换为 FP32 计算后回写。
- INT8 和 UINT8 转换为 FP16 计算后回写。
- 大数据量使用双缓冲队列；单 Tile 场景收敛为单缓冲，减少固定开销。
- 算子无跨核归约和核间依赖，workspace 为 0。

## 约束说明

- `gradients`、`outputs` 和 `y` 的数据类型必须一致。
- 三个 Tensor 的 Shape 必须一致，不支持广播。
- 输入输出格式为 ND。
- 支持空 Tensor。
- ACLNN 层支持非连续 Tensor，调用底层算子前会转换为连续 Tensor。
- 计算为确定性实现。

## 编译

在已加载 CANN 环境的仓库根目录执行：

```bash
bash build.sh --pkg --experimental --soc=ascend910b --ops=selu_grad -j8
```

## 单元测试

```bash
bash build.sh -u --experimental --soc=ascend910b --ops=selu_grad
```

也可以分别运行 Host、Kernel 或 API 单元测试：

```bash
bash build.sh -u --ophost --experimental --soc=ascend910b --ops=selu_grad
bash build.sh -u --opkernel --experimental --soc=ascend910b --ops=selu_grad
bash build.sh -u --opapi --experimental --soc=ascend910b --ops=selu_grad
```

## 调用说明

| 调用方式 | 调用样例 | 接口文档 |
|:--|:--|:--|
| ACLNN | [test_aclnn_selu_grad.cpp](./examples/test_aclnn_selu_grad.cpp) | [aclnnSeluBackward](./docs/aclnnSeluBackward.md) |

完成自定义算子包的编译和安装后，可在仓库根目录运行 ACLNN 样例：

```bash
bash build.sh --run_example selu_grad eager cust --vendor_name=custom --soc=ascend910b --experimental
```

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|:--|:--|:--|:--|:--|
| Delicate02 | 个人开发者 | SeluGrad | 2026/07/21 | SeluGrad 算子适配开源仓 |
