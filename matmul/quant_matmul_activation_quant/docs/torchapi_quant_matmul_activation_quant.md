# quant_matmul_activation_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

  融合量化的矩阵乘、激活以及动态量化，封装`aclnnQuantMatmulActivationQuantWeightNz`。当前支持激活为gelu、MX量化模式。输入`x1`、`x2`为P8量化矩阵，必选输入`x2_scale`、可选输入`x1_scale`为MX量化缩放因子，`bias`为偏置项；矩阵乘结果经激活函数后做动态量化，输出量化结果`y`和量化尺度`y_scale`。M/N/K维度及转置标志由`x1`、`x2`最后两维自动匹配推导，无需显式传入。

- 计算公式：

  - QuantMatmul MX量化模式：

    $$
    matmulOut[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
    $$

    其中，gsM、gsN 和 gsK 分别代表 groupSizeM、groupSizeN 和 groupSizeK；x1Slice 代表 x1 第 m 行长度为 groupSizeK 的向量，x2Slice 代表 x2 第 n 列长度为 groupSizeK 的向量；K 轴均从 j*groupSizeK 起始切片，j 的取值范围为 [0, kLoops)，kLoops = ceil(K / groupSizeK)，K 为 K 轴长度，支持最后的切片长度不足 groupSizeK。

  - 激活计算公式：

    - gelu_tanh（高性能近似）：
    $$
    activationOut=GELU(matmulOut)=matmulOut × Φ(matmulOut)=0.5 * matmulOut * (1 + tanh( \sqrt{2 / \pi} * (matmulOut + 0.044715 * matmulOut^{3})))
    $$

    - gelu_erf：
    $$
    activationOut=GELU(matmulOut)=0.5 * matmulOut * (1 + erf(matmulOut / \sqrt{2}))
    $$

  - 动态量化计算公式：

    - **场景1，当scale_alg为0时**：
      - 将输入 activationOut 在尾轴上按 $k = 32$ 个数分组，一组 k 个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}， k = 32$

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
      mxscale = 2^{shared\_exp}\\
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space k\\
      $$

      - 量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按尾轴上的分组输出yScaleOut。

      - emax：对应数据类型的最大正则数的指数位。

          |   DataType    | emax |
          | :-----------: | :--: |
          |  FLOAT4_E2M1  |  2   |
          |  FLOAT4_E1M2  |  0   |
          | FLOAT8_E4M3FN |  8   |
          |  FLOAT8_E5M2  |  15  |

    - **场景2，当scale_alg为1时，只涉及FP8类型**：
      - 将输入activationOut在尾轴上按$k = 32$个数分块，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一块不足$k = 32$个元素，把缺失值视为0，按照完整块处理。
      - 找到该块中数值的最大绝对值：

        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
        $$

      - 将 FP32 映射到目标数据类型 FP8 可表示的范围内，其中 $Amax(DType)$ 是目标精度能表示的最大值：

        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$

      - 将块缩放因子 $S_{fp32}^b$ 转换为 FP8 格式下可表示的缩放值 $S_{ue8m0}^b$
      - 从块的浮点缩放因子 $S_{fp32}^b$ 中提取无偏指数 $E_{int}^b$ 和尾数 $M_{fixp}^b$
      - 为保证量化时不溢出，对指数进行向上取整，且在 FP8 可表示的范围内：

        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
        $$

      - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
      - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
      - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是 $\left(S^b, [d^i]_{i=1}^k\right)$，其中 $S^b$ 代表块的缩放因子，这里指 $S_{ue8m0}^b$，$[d^i]_{i=1}^k$ 代表块内量化后的数据。

## 函数原型

```python
cann_ops_nn.quant_matmul_activation_quant(x1, x2, x2_scale, *, x1_scale=None, bias=None,
    output_dtype=None, x1_dtype=None, x2_dtype=None, x1scale_dtype=None, x2scale_dtype=None,
    group_sizes=None, activation_type="gelu_tanh", quant_mode="mx", round_mode="rint",
    scale_alg=0, dst_type_max=0.0) -> (Tensor y, Tensor y_scale)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `x1` | Tensor | 必选 | 矩阵乘运算中的左矩阵。数据格式为ND。最后两维为`(M, K)`或`(K, M)`，由与`x2`的维度匹配自动推导。 | torch.float8_e4m3fn、torch.float8_e5m2 | 2-6维，`(..., M, K)`或`(..., K, M)` |
| `x2` | Tensor | 必选 | 矩阵乘运算中的右矩阵。数据格式为FRACTAL_NZ。最后两维为`(K, N)`或`(N, K)`，由与`x1`的维度匹配自动推导。 | torch.float8_e4m3fn | 2-6维，`(..., K, N)`或`(..., N, K)` |
| `x2_scale` | Tensor | 必选 | 矩阵乘计算时x2的MX量化缩放因子。数据格式为ND。batch维须与`x2`一致。 | torch.float8_e8m0fnu | `(..., K//64, N, 2)`或`(..., N, K//64, 2)`（随`x2 方向） |
| `x1_scale` | Tensor | 可选 | 矩阵乘计算时x1的MX量化缩放因子。数据格式为ND。batch维须与`x1`一致。 | torch.float8_e8m0fnu | `(..., M, K//64, 2)`或`(..., K//64, M, 2)`（随`x1`方向） |
| `bias` | Tensor | 可选 | 矩阵乘运算后累加的偏置。数据格式为ND。 | float32 | `(N,)` |
| `output_dtype` | int | 可选 | 输出`y`的数据类型枚举值。支持torch.float8_e4m3fn、torch.float8_e5m2等。默认值None（等价于 0，表示与`x1`同类型）。 | int | - |
| `x1_dtype` | int | 可选 | `x1`的数据类型枚举值。不传入时根据`x1`的scalar_type自动推导。 | int | - |
| `x2_dtype` | int | 可选 | `x2`的数据类型枚举值。不传入时根据`x2`的scalar_type自动推导。 | int | - |
| `x1scale_dtype` | int | 可选 | `x1_scale`的数据类型枚举值。不传入时根据`x1_scale`的scalar_type自动推导。 | int | - |
| `x2scale_dtype` | int | 可选 | `x2_scale`的数据类型枚举值。不传入时根据`x2_scale`的scalar_type自动推导。 | int | - |
| `group_sizes` | List[int] | 可选 | 分组量化大小 `[groupSizeM, groupSizeN, groupSizeK]`，每个元素取值范围为[0, 65535]。 | list | `(3,)` |
| `activation_type` | str | 可选 | 激活函数类型，支持`"gelu_tanh"`、`"gelu_erf"`，默认值`"gelu_tanh"`。 | string | - |
| `quant_mode` | str | 可选 | 量化模式，当前支持`"mx"`，默认值`"mx"`。 | string | - |
| `round_mode` | str | 可选 | 舍入模式，支持`"rint"`、`"floor"`、`"round"`，当前仅支持`"rint"`。 | string | - |
| `scale_alg` | int | 可选 | 缩放算法。0 表示MX动态量化（场景 1），1表示FP8块缩放量化（场景 2），默认值0。 | int | - |
| `dst_type_max` | float | 可选 | 目标数据类型最大值，用于量化范围控制，默认值0.0（表示使用目标类型的默认最大值）。 | float32 | - |

## 返回值说明

| 输出名 | 输出类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `y` | Tensor | 必选 | 动态量化后的矩阵乘及激活计算结果。 | torch.float8_e4m3fn、torch.float8_e5m2 | `(..., M, N)`； |
| `y_scale` | Tensor | 必选 | 动态量化后每个分组对应的量化尺度，最后一维固定为2。 | torch.float8_e8m0fnu | `(..., M, CeilDiv(N, 64), 2)` |

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式调用。
- 不支持空Tensor。
- 支持连续Tensor，非连续Tensor仅支持最后两根轴转置场景。
- `x1`支持2-6维，`x2`仅支持NZ 4-8 维。
- 当`K`或`N`为1时，无法使用weightNz特性，本接口不支持此种场景。
- M/N/K维度及`transpose_x1`、`transpose_x2`由`x1`、`x2`最后两维自动匹配推导：取`x1`最后两维和`x2`最后两维共四个值中相等的一对作为K，`x1`中剩余的为M，`x2`中剩余的为N。若四组组合中无相等维度则报错。
- `x1`、`x2`的batch维度（除最后两维外的维度）支持广播（右对齐），如`x1=(1,M,K)`、`x2=(8,K,N)`输出`(8,M,N)`。
- `x1_scale`、`x2_scale`若传入，其batch维度（除最后三维外的维度）的数量和每一维的值必须与对应的`x1`、`x2`完全一致；若`x1`无 batch维度（2D），则`x1_scale`、`x2_scale`须为3D。
- `x1_scale`、`x2_scale`最后一维必须为2。
- `group_sizes`若传入，必须包含三个元素`[groupSizeM, groupSizeN, groupSizeK]`，每个元素取值范围为[0, 65535]，当前MX场景仅支持[1, 1, 32]。
- 输入和输出支持以下数据类型组合：

  | x1            | x2            | x1_scale   |  x2_scale    | bias             | y                         | y_scale      |
  |---------------|---------------|-------------|-------------|------------------|---------------------------|-------------|
  | torch.float8_e4m3fn | torch.float8_e4m3fn | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float8_e4m3fn2 | torch.float8_e8m0fnu |
  | torch.float8_e5m2   | torch.float8_e4m3fn | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float8_e5m2 | torch.float8_e8m0fnu |

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    m, k, n = 5, 64, 128
    group_size = 32
    # x1 物理形状 (M, K)；x2 物理形状 (K, N)
    x1 = torch.randn(m, k, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
    x2 = torch.randn(k, n, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
    x2_nz = torch_npu.npu_format_cast(x2, 29) # 29为NZ格式
    x1_scale = torch.ones(m, k // group_size // 2, 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(k // group_size // 2, n, 2, dtype=torch.float8_e8m0fnu).npu()

    y, y_scale = torch.ops.cann_ops_nn.quant_matmul_activation_quant(
        x1, x2_nz, x2_scale, x1_scale=x1_scale, bias=None,
        activation_type="gelu_tanh", quant_mode="mx", round_mode="rint",
        scale_alg=0, dst_type_max=0.0)
    print("y: ", y.cpu())
    print("y_scale: ", y_scale.cpu())
    ```
