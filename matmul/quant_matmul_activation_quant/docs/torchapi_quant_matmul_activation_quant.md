# quant_matmul_activation_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

  融合量化的矩阵乘、激活以及动态量化，按x2格式封装ND或WeightNZ接口。当前支持激活为gelu和swiglu、MX量化模式。输入`x1`、`x2`为FP8或FP4量化矩阵，必选输入`x2_scale`、`x1_scale`为MX量化缩放因子，`bias`为偏置项；矩阵乘结果经激活函数后做动态量化，输出量化结果`y`和量化尺度`y_scale`。Torch的逻辑视图固定按`(..., M, K)`和`(..., K, N)`传入；FP4在PyTorch侧以uint8双元素打包，连续非转置布局的被打包轴物理长度减半，转置布局则由stride决定被打包轴，由C++桥接层恢复逻辑维度。M/N/K不通过相等维度猜测转置，ACLNN根据最后两维stride处理转置视图。

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

    - swiglu：原始N必须为正数且是64的倍数。先在完整FP32矩阵乘结果上加bias，再沿尾轴按`matmulOut=[gate | linear]`连续等分：

    $$
    activationOut=SiLU(gate) * linear
    $$

    SwiGLU结果以RNE舍入为BF16后执行MX量化，输出宽度为原始N的二分之一。

  - 动态量化计算公式：

    - **场景1，当scale_alg为0时**：
      - 将输入 activationOut 在尾轴上按 $k = 32$ 个数分组，一组 k 个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}， k = 32$

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
      mxscale = 2^{shared\_exp}\\
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space k\\
      $$

      - 量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出`y`，mxscale按尾轴上的分组组成输出`y_scale`。

      - emax：对应数据类型的最大正则数的指数位。

          |   DataType    | emax |
          | :-----------: | :--: |
          |  FLOAT4_E2M1  |  2   |
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
      - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$，最终输出的量化结果是 $\left(S^b, [d^i]_{i=1}^k\right)$，其中 $S^b$ 代表块的缩放因子，这里指 $S_{ue8m0}^b$，$[d^i]_{i=1}^k$ 代表块内量化后的数据。

    - **场景3，当scale_alg为2时，只涉及FP4_E2M1类型**：
      - 当dstTypeMax = 0.0/6.0/7.0时：
        - 将输入x在axis维度上按k = blocksize个数分组，一组k个数  $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize：
        $$
        shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果} 尾数位的高比特前一/两位 \text{为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases} \\
        $$
        $$
        P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
        $$
        - ​量化后的$P_{i}$按对应的$V_{i}$的位置组成输出`y`，mxscale按对应的axis维度上的分组组成输出`y_scale`。
      - 当dstTypeMax != 0.0/6.0/7.0时：
        - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
        - 找到该块中数值的最大绝对值:
        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
        $$
        - 将FP32映射到目标数据类型可表示的范围内，其中当dst_max_value=0时，$Amax(DType)$是目标精度能表示的最大值；当dst_max_value!=0时，$Amax(DType)$是dst_max_value传入值。
        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$
        - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$。
        - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
        - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：
          $$
          E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{否则} \end{cases}
          $$
        - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
        - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
        - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。
        - ​量化后的$P_{i}$按对应的$V_{i}$的位置组成输出`y`，mxscale按对应的axis维度上的分组组成输出`y_scale`。

  - FP8缩放算法选择：
    - `scale_alg=0`使用OCP共享指数算法，与[DynamicMxQuant](../../../quant/dynamic_mx_quant/README.md)、[SwigluMxQuant](../../../quant/swiglu_mx_quant/README.md)的OCP语义一致，不会根据目标FP8最大有限值额外上调scale；归一化结果位于FP8表示边界外时，输出由底层FP8类型转换语义决定。
    - `scale_alg=1`使用BLAS算法，按目标FP8最大有限值计算scale并将E8M0指数向上取整，可降低边界值量化溢出的风险。输入可能触及FP8表示边界且业务要求有限输出时，建议使用该算法。
    - 在相同输出dtype、32元素分组和`scale_alg`下，融合算子只是在量化前增加矩阵乘和激活，scale生成方式以及FP8转换规则与拆分调用小算子保持一致。

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
| `x1` | Tensor | 必选 | 矩阵乘运算中的左矩阵。数据格式为ND。逻辑形状为`(..., M, K)`；FP4以uint8双元素打包，非转置连续布局通常为末轴`K/2`，转置布局的打包轴由stride决定，显式dtype=296时恢复逻辑维度。可传入最后两轴转置后的Tensor视图。 | torch.float8_e4m3fn、torch.float8_e5m2、torch.uint8（FP4双元素打包，显式dtype=296） | 2-6维，逻辑形状`(..., M, K)` |
| `x2` | Tensor | 必选 | 矩阵乘运算中的右矩阵。数据格式为ND或FRACTAL_NZ。逻辑形状为`(..., K, N)`；ND FP4以uint8双元素打包，非转置连续布局通常为末轴`N/2`，转置布局的打包轴由stride决定，显式dtype=296时恢复逻辑维度。WeightNZ的物理存储形状由torch_npu管理。可传入最后两轴转置后的Tensor视图。 | torch.float8_e4m3fn、torch.float8_e5m2、torch.uint8（FP4双元素打包，显式dtype=296） | 2-6维，逻辑形状`(..., K, N)` |
| `x2_scale` | Tensor | 必选 | 矩阵乘计算时x2的MX量化缩放因子。数据格式为ND。batch维须与`x2`一致。 | torch.float8_e8m0fnu | `(..., ceil(K/64), N, 2)` |
| `x1_scale` | Tensor | 必选 | 矩阵乘计算时x1的MX量化缩放因子。数据格式为ND。batch维须与`x1`一致。 | torch.float8_e8m0fnu | `(..., M, ceil(K/64), 2)` |
| `bias` | Tensor | 可选 | 矩阵乘运算后累加的偏置。数据格式为ND。 | float32 | `(N,)`，三维输出时也支持`(B,1,N)` |
| `output_dtype` | int | 可选 | 输出`y`的数据类型枚举值，必须与`x1`的逻辑数据类型一致。支持torch.float8_e4m3fn、torch.float8_e5m2、torch.uint8（FP4双元素打包，显式dtype=296）等。默认值None（表示与`x1`同类型）。 | int | - |
| `x1_dtype` | int | 可选 | `x1`的数据类型枚举值。不传入时根据`x1`的scalar_type自动推导。 | int | - |
| `x2_dtype` | int | 可选 | `x2`的数据类型枚举值。不传入时根据`x2`的scalar_type自动推导。 | int | - |
| `x1scale_dtype` | int | 可选 | `x1_scale`的数据类型枚举值。不传入时根据`x1_scale`的scalar_type自动推导。 | int | - |
| `x2scale_dtype` | int | 可选 | `x2_scale`的数据类型枚举值。不传入时根据`x2_scale`的scalar_type自动推导。 | int | - |
| `group_sizes` | List[int] | 可选 | 分组量化大小 `[groupSizeM, groupSizeN, groupSizeK]`，每个元素取值范围为[0, 65535]。 | list | `(3,)` |
| `activation_type` | str | 可选 | 激活函数类型，支持`"gelu_tanh"`、`"gelu_erf"`、`"swiglu"`，默认值`"gelu_tanh"`。 | string | - |
| `quant_mode` | str | 可选 | 量化模式，当前支持`"mx"`，默认值`"mx"`。 | string | - |
| `round_mode` | str | 可选 | 舍入模式。当`output_dtype`为FLOAT4_E2M1时，支持`"rint"`、`"floor"`、`"round"`；当`output_dtype`为FLOAT8_E4M3FN/FLOAT8_E5M2时，仅支持`"rint"`。默认值`"rint"`。 | string | - |
| `scale_alg` | int | 可选 | 缩放算法。当`output_dtype`为FLOAT4_E2M1时，支持取值0和2，0表示场景1，2表示场景3（开启`dst_type_max`）；当`output_dtype`为FLOAT8_E4M3FN/FLOAT8_E5M2时，支持取值0和1，0表示场景1，1表示场景2。默认值0。 | int | - |
| `dst_type_max` | float | 可选 | 目标数据类型最大值，用于量化范围控制。当`scale_alg`为0或1时不生效，传入0.0即可；当`scale_alg`为2时，支持取值0.0和6.0-12.0，0.0表示使用目标类型的默认最大值。默认值0.0。 | float32 | - |

## 返回值说明

| 输出名 | 输出类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `y` | Tensor | 必选 | 动态量化后的矩阵乘及激活计算结果。 | torch.float8_e4m3fn、torch.float8_e5m2、torch.uint8（FP4双元素打包，显式dtype=296） | GELU：`(..., M, N)`；SwiGLU：`(..., M, N/2)` |
| `y_scale` | Tensor | 必选 | 动态量化后每个分组对应的量化尺度，最后一维固定为2。 | torch.float8_e8m0fnu | `(..., M, CeilDiv(H, 64), 2)`，GELU时H=N，SwiGLU时H=N/2 |

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式调用。
- 不支持空Tensor。
- 支持连续Tensor，非连续Tensor仅支持最后两根轴转置场景。
- 本接口不暴露转置参数，内部调用aclnn时transposeX1、transposeX2固定为false，转置只能通过Tensor的stride表达：
  - `x1`的逻辑shape始终为`(..., M, K)`，`x2`的逻辑shape始终为`(..., K, N)`。
  - 需要转置时，以最后两维的转置视图传入：`x1`的stride为`(..., 1, M)`，`x2`的stride为`(..., 1, K)`；不转置时stride为正常的连续步长，`x1`为`(..., K, 1)`，`x2`为`(..., N, 1)`。
  - 接口依据stride自动识别转置并推导M/N/K；仅支持沿最后两维的转置，其他轴的视图/非连续Tensor不支持。
- 令`B = CeilDiv(K, 64)`。`x1_scale`的输入view shape始终为`(..., M, B, 2)`，`x2_scale`始终为`(..., B, N, 2)`；最后一维的2个scale值不参与转置。转置时view shape不变，`x1_scale`尾三维stride为`(2, 2M, 1)`，`x2_scale`为`(2, 2B, 1)`；非转置时分别为`(2B, 2, 1)`和`(2N, 2, 1)`。`x1`与`x1_scale`、`x2`与`x2_scale`的转置布局必须分别一致。
- `x1`、`x2`的逻辑Tensor均支持2-6维；NZ的内部存储形状由torch_npu管理。
- `x2`为NZ时支持E4M3FN或显式声明为E2M1的FP4打包数据；不支持E5M2。
- WeightNZ路径不支持`K`或`N`为1；ND路径另按具体激活的约束校验。
- `x1`、`x2`的batch维度（除最后两维外的维度）支持广播（右对齐），如`x1=(1,M,K)`、`x2=(8,K,N)`输出`(8,M,N)`。
- `x1_scale`、`x2_scale`必须传入，其batch维度（除最后三维外的维度）的数量和每一维的值必须与对应的`x1`、`x2`完全一致；对应输入为2D时，该输入的scale须为3D。
- `x1_scale`、`x2_scale`最后一维必须为2。
- `group_sizes`若传入，必须包含三个元素`[groupSizeM, groupSizeN, groupSizeK]`，每个元素取值范围为[0, 65535]，当前MX场景仅支持[0, 0, 0]、[1, 1, 32]。
- GELU和SwiGLU的输出类型都必须与`x1_dtype`或`x1`的逻辑数据类型一致。
- 下表列出支持的数据类型组合；SwiGLU仅支持其中的FP8组合：

  | x1            | x2            | x1_scale   |  x2_scale    | bias             | y                         | y_scale      |
  |---------------|---------------|-------------|-------------|------------------|---------------------------|-------------|
  | torch.float8_e4m3fn | torch.float8_e4m3fn | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float8_e4m3fn | torch.float8_e8m0fnu |
  | torch.float8_e5m2   | torch.float8_e4m3fn | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float8_e5m2 | torch.float8_e8m0fnu |
  | torch.float8_e5m2   | torch.float8_e5m2 | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float8_e5m2 | torch.float8_e8m0fnu |
  | torch.float8_e4m3fn | torch.float8_e5m2 | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float8_e4m3fn | torch.float8_e8m0fnu |
  | torch.uint8（FP4双元素打包，显式dtype=296） | torch.uint8（FP4双元素打包，显式dtype=296） | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.uint8（FP4双元素打包，显式dtype=296） | torch.float8_e8m0fnu |

- MXFP4场景约束（`x1`、`x2`、`y`实际数据类型均为`torch.uint8`，逻辑类型由dtype参数296指定）：
  - 当前`x1`、`x2`、`y`都已打包为uint8，打包前的shape尾轴需要为偶数。
  - 当`x2`为NZ格式时，`x1`不支持转置。
  - `scale_alg`仅支持取值0和2。
  - 当`scale_alg`为2时，`dst_type_max`支持取值0.0和6.0-12.0。
  - `round_mode`支持`"rint"`、`"floor"`、`"round"`。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用

  - GELU FP8场景示例：

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_nn

    m, k, n = 5, 64, 128
    group_size = 32
    # x1 物理形状 (M, K)；x2 物理形状 (K, N)
    x1 = torch.randn(m, k, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
    x2 = torch.randn(k, n, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
    x2_nz = torch_npu.npu_format_cast(x2, 29) # 29为NZ格式
    x1_scale = torch.ones(m, math.ceil(k / group_size / 2), 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(math.ceil(k / group_size / 2), n, 2, dtype=torch.float8_e8m0fnu).npu()

    y, y_scale = torch.ops.cann_ops_nn.quant_matmul_activation_quant(
        x1, x2_nz, x2_scale, x1_scale=x1_scale, bias=None,
        activation_type="gelu_tanh", quant_mode="mx", round_mode="rint",
        scale_alg=0, dst_type_max=0.0)
    print("y: ", y.cpu())
    print("y_scale: ", y_scale.cpu())
    ```

  - SwiGLU FP8场景示例：

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_nn

    m, k, n = 5, 64, 128  # SwiGLU中的n是矩阵乘原始输出宽度，必须是64的倍数
    group_size = 32
    x1 = torch.randn(m, k, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
    x2 = torch.randn(k, n, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
    x2_nz = torch_npu.npu_format_cast(x2, 29)  # 29为NZ格式
    x1_scale = torch.ones(m, math.ceil(k / group_size / 2), 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(math.ceil(k / group_size / 2), n, 2, dtype=torch.float8_e8m0fnu).npu()
    bias = torch.zeros(n, dtype=torch.float32).npu()  # bias按原始n传入，在SwiGLU split前相加

    y, y_scale = torch.ops.cann_ops_nn.quant_matmul_activation_quant(
        x1, x2_nz, x2_scale, x1_scale=x1_scale, bias=bias,
        activation_type="swiglu", quant_mode="mx", round_mode="rint",
        scale_alg=1, dst_type_max=0.0)
    # y shape: (M, N/2)；y_scale shape: (M, CeilDiv(N/2, 64), 2)
    print("y: ", y.cpu())
    print("y_scale: ", y_scale.cpu())
    ```

  - FP4场景示例：

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_nn

    m, k, n = 5, 64, 128
    group_size = 32
    # x1 物理形状 (M, K//2)；x2 物理形状 (K, N//2)，FP4双nibble打包为uint8末维减半
    x1 = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8).npu()
    x2 = torch.randint(0, 256, (k, n // 2), dtype=torch.uint8).npu()
    x2_nz = torch_npu.npu_format_cast(x2, 29) # 29为NZ格式
    x1_scale = torch.ones(m, math.ceil(k / group_size / 2), 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(math.ceil(k / group_size / 2), n, 2, dtype=torch.float8_e8m0fnu).npu()

    y, y_scale = torch.ops.cann_ops_nn.quant_matmul_activation_quant(
        x1, x2_nz, x2_scale, x1_scale=x1_scale, bias=None,
        output_dtype=296,
        x1_dtype=296,
        x2_dtype=296,
        activation_type="gelu_tanh", quant_mode="mx", round_mode="rint",
        scale_alg=0, dst_type_max=0.0)
    # y 物理形状 (M, N//2)，FP4双nibble打包为uint8末维减半，y_scale 物理形状(M, CeilDiv(N, 64), 2)
    print("y: ", y.cpu())
    print("y_scale: ", y_scale.cpu())
    ```

## SwiGLU支持范围

`activation_type="swiglu"`在完整FP32矩阵乘结果加bias后，沿最后一维按`C=[gate | linear]`连续等分，计算`SiLU(gate) * linear`，将SwiGLU结果以RNE舍入为BF16后执行MX量化。

- 原始`N`必须为正数且是64的倍数；输出宽度`N/2`可有32列尾块。
- `x1`、`x2`为E4M3FN或E5M2，`y`的数据类型必须与`x1`一致；输入和输出scale为E8M0。
- `x2`支持ND或WeightNZ；WeightNZ场景`x2`仅支持E4M3FN。`transpose_x1=false`，`transpose_x2`支持false或true；不支持MXFP4。
- `bias`可为空；非空时为FP32`[N]`，三维输出场景还支持`[B,1,N]`。bias在split之前作用于完整原始N宽度，不接受`[N/2]`。
- 输入rank为2～6，batch轴右对齐广播；scale的batch维必须与对应输入一致。
- `quant_mode="mx"`、`round_mode="rint"`、`scale_alg=0/1`；group size为默认值或`[1,1,32]`。

SwiGLU输出形状为`y=[...,M,N/2]`、`y_scale=[...,M,ceil((N/2)/64),2]`。GELU输出宽度仍为N。两个32元素量化组构成一个64元素scale存储组，末尾不足的元素不写入y。

ND与WeightNZ接口约束分别见[aclnnQuantMatmulActivationQuant](aclnnQuantMatmulActivationQuant.md)和[aclnnQuantMatmulActivationQuantWeightNz](aclnnQuantMatmulActivationQuantWeightNz.md)。

### PyTorch扩展约定

`x1_scale`在当前MX路径必须提供实际输入scale，不会自动假定单位scale；缺失或形状/类型不匹配由ACLNN和tiling层校验。输出在x1所在NPU设备分配，输入、scale和bias必须处于同一设备。

`output_dtype=None`使用x1的逻辑数据类型（提供`x1_dtype`时使用该覆盖值）。运行时直接把dtype编码交给C++桥接层，由根目录`torch_extension/cann_ops_nn/common/aclnn_common.h`统一转换；Python不再做dtype编码归一化。合法FP8组合的Meta输出直接继承`x1.dtype`，FP4仅保留uint8双元素打包所需的特殊形状处理。
