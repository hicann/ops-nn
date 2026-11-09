# QuantMatmulActivationQuant

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：融合MX（Microscaling，微缩放）量化矩阵乘、激活和动态MX量化，输出量化结果及对应的缩放因子（scale）。支持GELU（Gaussian Error Linear Unit，高斯误差线性单元）的tanh/erf形式、SiLU（Sigmoid Linear Unit，Sigmoid线性单元）及SwiGLU（Swish Gated Linear Unit，Swish门控线性单元）。数据类型及转置组合见[约束说明](#约束说明)，MX模式介绍见[量化模式](../../docs/zh/context/quant_mode_introduction.md)。

- 计算流程：先完成MX量化矩阵乘及偏置相加，再计算激活，最后执行动态MX量化。各步骤的计算公式如下。

以下公式省略batch下标，$X_1$、$X_2$表示按转置状态解释后的矩阵，形状分别为$(M,K)$、$(K,N)$。M为左矩阵的行数，K为公共归约维度，N为右矩阵的列数。

### MX量化矩阵乘

沿K轴每32个元素划为一组，每组矩阵乘结果乘以对应的输入scale，再与偏置累加，得到$C$：

$$
C_{m,n}=\sum_{j=0}^{L-1}\left[
\left(\sum_{t=0}^{g_j-1}X_{1,m,jg_K+t}\,X_{2,jg_K+t,n}\right)
S_{1,m,j}\,S_{2,j,n}\right]+b_n
$$

其中，$g_M$、$g_N$、$g_K$分别为M、N、K方向的分组大小，当前固定为$(1,1,32)$；$L=\lceil K/g_K\rceil$，$g_j=\min(g_K,K-jg_K)$。$S_1$、$S_2$分别表示`x1_scale`、`x2_scale`中第$j$组的scale，实际存储形状见[形状](#形状)。矩阵乘采用FP32累加，偏置在激活前相加；未传入偏置时，$b_n=0$。

### 激活

对矩阵乘结果$C$应用指定的激活函数，得到$A$：

- `activation_type="gelu_tanh"`：GELU的tanh近似。

  $$
  A=\frac{C}{2}\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(C+0.044715C^3\right)\right)\right]
  $$

- `activation_type="gelu_erf"`：GELU的erf形式。

  $$
  A=\frac{C}{2}\left[1+\operatorname{erf}\left(\frac{C}{\sqrt{2}}\right)\right]
  $$

- `activation_type="silu"`：逐元素计算SiLU。

  $$
  A=\operatorname{SiLU}(C)=\frac{C}{1+e^{-C}}
  $$

- `activation_type="swiglu"`：将$C$沿最后一维等分为左半部分$G$和右半部分$U$，对$G$计算SiLU后与$U$逐元素相乘。

  $$
  C=[G\mid U],\qquad A=\operatorname{SiLU}(G)\odot U=\frac{G}{1+e^{-G}}\odot U
  $$

上述激活均在FP32中计算。记激活后的输出宽度为$H$：GELU和SiLU的$H=N$，SwiGLU的$H=N/2$。

### 动态MX量化

将激活结果按最近偶数舍入模式（RNE，Round to Nearest, ties to Even）转换为BFLOAT16，即舍入到最近的可表示值，中间值取尾数最低位为0的值，再沿最后一维每32个元素划为一组。最后一组不足32个元素时补0参与scale计算，补齐元素不写入输出。

$$
V_i=\operatorname{BF16}_{RNE}(A_i),\qquad a=\max_i|V_i|,\qquad s=2^e
$$

每组共用一个scale$s$，按`round_mode`转换为输出数据类型$D$：

$$
Q_i=\begin{cases}
\operatorname{FP32}(V_i/s), & D\text{为FP8}\\
\operatorname{BF16}_{RNE}(V_i/s), & D\text{为FP4}
\end{cases},\qquad y_i=\operatorname{Cast}_{D,\,\mathrm{round\_mode}}(Q_i)
$$

`y_scale`使用FLOAT8_E8M0存储scale。第$j$组写入`y_scale[..., m, j // 2, j % 2]`，输出形状为`(..., M, ceil(H/64), 2)`。

#### scale的计算方法

以下公式适用于$a>0$且有限的情况，令$p=\lfloor\log_2a\rfloor$。

- `scale_alg=0`：OCP（Open Compute Project，开放计算项目）共享指数算法，支持FP8和FP4。

  $$
  e=\max(-127,p-e_{\max})
  $$

  FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1的$e_{\max}$分别为8、15、2。

- `scale_alg=1`：按目标类型的最大有限值计算scale，仅支持FP8。

  $$
  S=\operatorname{FP32}\left(a\cdot\operatorname{FP32}(1/F_{\max})\right),\qquad
  e=\max(-127,\lceil\log_2S\rceil)
  $$

  FLOAT8_E4M3FN、FLOAT8_E5M2的$F_{\max}$分别为448、57344。

- `scale_alg=2`：按`dst_type_max`指定的范围计算scale，仅支持FP4。令$d=\operatorname{FP32}(\mathrm{dst\_type\_max})$，传入0时取$d=6$。

  当$d=6$或$d=7$时：

  $$
  e=\max\left(-127,\begin{cases}
  p-1, & a>d\cdot2^{p-2}\\
  p-2, & a\le d\cdot2^{p-2}
  \end{cases}\right)
  $$

  当$d$为$[6,12]$内的其他值时：

  $$
  S=\operatorname{FP32}\left(a\cdot\operatorname{FP32}(1/d)\right),\qquad
  e=\max(-127,\lceil\log_2S\rceil)
  $$

E8M0以字节值$e+127$编码$2^e$。特殊情况如下：

- 全零组的量化结果为0，scale的编码为0；输出scale中无对应数据分组的补齐位置也填入编码0。该编码表示$2^{-127}$。
- $S=0$时取$e=-127$。算法0，以及算法2中$d=6$或$d=7$的分支，在scale编码为0时将该组量化结果置0。
- 含NaN或Inf的组，scale的编码为255（NaN）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x1 | 输入 | 矩阵乘的左矩阵，对应公式中的X<sub>1</sub>。 | FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1 | ND |
| x2 | 输入 | 矩阵乘的右矩阵，对应公式中的X<sub>2</sub>。NZ不支持FLOAT8_E5M2。 | FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1 | ND、FRACTAL_NZ |
| bias | 可选输入 | <ul><li>在激活前相加的偏置，对应公式中的b。</li><li>未提供时不加偏置。</li></ul> | FLOAT32 | ND |
| x1_scale | 可选输入 | <ul><li>x1的分组scale，对应公式中的S<sub>1</sub>。</li><li>本算子仅支持MX模式，执行时须提供对应的分组scale。</li></ul> | FLOAT8_E8M0 | ND |
| x2_scale | 可选输入 | <ul><li>x2的分组scale，对应公式中的S<sub>2</sub>。</li><li>本算子仅支持MX模式，执行时须提供对应的分组scale。</li></ul> | FLOAT8_E8M0 | ND |
| y | 输出 | 动态量化结果，对应公式中的y，数据类型须与x1一致。 | FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1 | ND |
| y_scale | 输出 | 每32个输出元素对应的scale，对应公式中的s。 | FLOAT8_E8M0 | ND |
| transpose_x1 | 可选属性 | <ul><li>是否转置x1的最后两维：false表示不转置，true表示转置。</li><li>默认false。</li></ul> | BOOL | - |
| transpose_x2 | 可选属性 | <ul><li>是否转置x2的最后两维：false表示不转置，true表示转置。</li><li>默认false。</li></ul> | BOOL | - |
| group_size | 可选属性 | <ul><li>输入量化在M、N、K方向的分组大小编码，仅支持0或4295032864，均表示[1,1,32]。</li><li>默认0。</li></ul> | INT64 | - |
| activation_type | 可选属性 | <ul><li>激活类型，支持"gelu_tanh"、"gelu_erf"、"silu"、"swiglu"。</li><li>默认"gelu_tanh"。</li></ul> | STRING | - |
| y_dtype | 可选属性 | <ul><li>y的数据类型枚举，支持FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1对应的枚举值，所指定的数据类型须与x1一致。</li><li>默认为FLOAT8_E4M3FN对应的枚举值。</li></ul> | INT64 | - |
| quant_mode | 可选属性 | <ul><li>量化模式，仅支持"mx"。</li><li>默认"mx"。</li></ul> | STRING | - |
| round_mode | 可选属性 | <ul><li>转换为输出数据类型时的舍入模式。FP8仅支持"rint"，表示舍入到最近的可表示值，中间值取尾数最低位为0的值。</li><li>FP4还支持"floor"（向负无穷方向舍入）和"round"（舍入到最近的可表示值，中间值向远离0的方向舍入）。</li><li>默认"rint"。</li></ul> | STRING | - |
| scale_alg | 可选属性 | <ul><li>输出scale的计算方法，FP8支持0、1，FP4支持0、2。</li><li>算法定义见<a href="#scale的计算方法">scale的计算方法</a>，默认0。</li></ul> | INT64 | - |
| dst_type_max | 可选属性 | <ul><li>scale_alg=2时计算输出scale所用的参考上限，对应公式中的d。</li><li>支持0.0（等价于6.0）或[6.0,12.0]内的有限值。</li><li>默认0.0；其他算法忽略该属性。</li></ul> | FLOAT32 | - |

## 约束说明

以下为算子执行时的输入要求。接口布局限制见[aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant.md)和[aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz.md)。

- x1、x2的逻辑维数为2～6；不支持空Tensor，各维度须为正整数。
- x1、x2的各逻辑维度及广播后的batch维乘积不超过$2^{32}-1$，激活前输出元素数不超过$2^{63}-1$。

矩阵和输出支持以下组合：

| x1 | x2 | y | x2格式 |
| --- | --- | --- | --- |
| FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E4M3FN | ND、FRACTAL_NZ |
| FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT8_E5M2 | ND、FRACTAL_NZ |
| FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT8_E4M3FN | ND |
| FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT8_E5M2 | ND |
| FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT4_E2M1 | ND、FRACTAL_NZ |

### 激活与转置

<a id="激活能力对照表"></a>

下表中，FP8指FLOAT8_E4M3FN或FLOAT8_E5M2，FP4指FLOAT4_E2M1；同时须满足上方的数据类型组合要求。

| activation_type | 数据类型 | 输出宽度H | N的要求 | x2为ND时的transpose_x1 | x2为NZ时的transpose_x1 |
| --- | --- | --- | --- | --- | --- |
| "gelu_tanh"、"gelu_erf" | FP8 | N | $N>0$ | false、true | false、true |
| "gelu_tanh"、"gelu_erf" | FP4 | N | 正偶数 | false、true | false |
| "silu" | FP8 | N | $N>0$ | false、true | false |
| "swiglu" | FP8 | N/2 | 64的正整数倍 | false、true | false |

`transpose_x2`均支持false、true。x2为NZ时，K、N均须大于1。

FP4仅支持GELU，且$K>2$、N为偶数。FP4输入x1和x2在连续存储布局下的最内轴须为偶数：x1不转置时检查K，转置时检查M；x2不转置时检查N，转置时检查K。

### 形状

`batch1`、`batch2`表示各输入前0～4个batch维，`batch`表示右对齐广播后的batch维；没有batch维时省略该部分。令`K64=ceil(K/64)`，其中ceil表示向上取整。以下均为逻辑形状，FP4的维度按4位元素计数：

| 张量 | 不转置 | 转置 |
| --- | --- | --- |
| x1 | `(batch1, M, K)` | `(batch1, K, M)` |
| x1_scale | `(batch1, M, K64, 2)` | `(batch1, K64, M, 2)` |
| x2 | `(batch2, K, N)` | `(batch2, N, K)` |
| x2_scale | `(batch2, K64, N, 2)` | `(batch2, N, K64, 2)` |

- 两个输入的K必须相等；输入scale的batch维必须与对应矩阵逐维一致。
- 每32个K元素对应一个输入scale，相邻两组scale按顺序存放在长度为2的末维中。
- `y`的形状为`(batch, M, H)`，`y_scale`为`(batch, M, ceil(H/64), 2)`，H由激活类型决定。
- bias的形状为`(N,)`；仅当输出为三维时，还支持`(B, 1, N)`，B须等于广播后的batch大小。SwiGLU的bias长度仍为激活前的N。

`group_size`的非零值按`(groupSizeM << 32) | (groupSizeN << 16) | groupSizeK`编码，高16位为0；本算子的分组固定为`[1,1,32]`。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | [GELU样例](examples/arch35/test_aclnn_quant_matmul_gelu_mx_quant.cpp)、[SwiGLU样例](examples/arch35/test_aclnn_quant_matmul_swiglu_mx_quant.cpp) | 通过[aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant.md)调用，x2使用ND格式。 |
| aclnn API | [GELU的NZ权重样例](examples/arch35/test_aclnn_quant_matmul_gelu_mx_quant_weight_nz.cpp) | 通过[aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz.md)调用，x2使用FRACTAL_NZ格式。 |
| GE图模式 | - | 通过[QuantMatmulActivationQuant算子IR](op_graph/quant_matmul_activation_quant_proto.h)构图调用。 |
| PyTorch API | [FP8和FP4调用示例](docs/torchapi_quant_matmul_activation_quant.md#调用示例) | 通过[cann_ops_nn.quant_matmul_activation_quant](docs/torchapi_quant_matmul_activation_quant.md)调用；扩展安装方法见[PyTorch扩展说明](../../torch_extension/README.md)。 |

## 参考资源

- 测试运行方式见[测试说明](tests/README.md)。
