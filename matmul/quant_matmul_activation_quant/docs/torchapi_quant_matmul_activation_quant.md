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

- 接口功能：融合MX（Microscaling，微缩放）量化矩阵乘、激活和动态MX量化，返回量化结果`y`及分组缩放因子（scale）`y_scale`。支持GELU（Gaussian Error Linear Unit，高斯误差线性单元）的tanh/erf形式、SiLU（Sigmoid Linear Unit，Sigmoid线性单元）及SwiGLU（Swish Gated Linear Unit，Swish门控线性单元），x2可使用ND或FRACTAL_NZ格式。

- 计算流程：先完成MX量化矩阵乘及偏置相加，再计算激活，最后执行动态MX量化。

以下公式省略batch下标。$X_1$、$X_2$表示参与计算的矩阵，形状分别为$(M,K)$、$(K,N)$；M为左矩阵的行数，K为公共归约维度，N为右矩阵的列数。转置通过Tensor视图表达，具体要求见[布局与存储](#布局与存储)。

### MX量化矩阵乘

沿K轴每32个元素划为一组，每组矩阵乘结果乘以对应的输入scale，再与偏置累加，得到$C$：

$$
C_{m,n}=\sum_{j=0}^{L-1}\left[
\left(\sum_{t=0}^{g_j-1}X_{1,m,jg_K+t}\,X_{2,jg_K+t,n}\right)
S_{1,m,j}\,S_{2,j,n}\right]+b_n
$$

其中，$g_M$、$g_N$、$g_K$分别为M、N、K方向的分组大小，当前固定为$(1,1,32)$；$L=\lceil K/g_K\rceil$，$g_j=\min(g_K,K-jg_K)$。$S_1$、$S_2$分别表示`x1_scale`、`x2_scale`中第$j$组的scale，实际存储形状见[布局与存储](#布局与存储)。矩阵乘采用FP32累加，偏置在激活前相加；未传入偏置时，$b_n=0$。

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

将激活结果按最近偶数舍入模式（Round to Nearest, ties to Even，RNE）转换为BF16，即舍入到最近的可表示值，中间值取尾数最低位为0的值，再沿最后一维每32个元素划为一组。最后一组不足32个元素时补0参与scale计算，补齐元素不写入输出。

$$
V_i=\operatorname{BF16}_{RNE}(A_i),\qquad a=\max_i|V_i|,\qquad s=2^e
$$

每组共用一个scale $s$，按`round_mode`转换为输出数据类型$D$：

$$
Q_i=\begin{cases}
\operatorname{FP32}(V_i/s), & D\text{为FP8}\\
\operatorname{BF16}_{RNE}(V_i/s), & D\text{为FP4}
\end{cases},\qquad y_i=\operatorname{Cast}_{D,\,\mathrm{round\_mode}}(Q_i)
$$

`y_scale`的数据类型为`torch.float8_e8m0fnu`。第$j$组的scale写入`y_scale[..., m, j // 2, j % 2]`，输出形状为`(..., M, ceil(H/64), 2)`。

各`scale_alg`的计算公式及特殊值处理见[scale的计算方法](../README.md#scale的计算方法)。

## 函数原型

通过`cann_ops_nn.quant_matmul_activation_quant`调用：

```python
cann_ops_nn.quant_matmul_activation_quant(
    x1,
    x2,
    x2_scale,
    *,
    x1_scale=None,
    bias=None,
    output_dtype=None,
    x1_dtype=None,
    x2_dtype=None,
    x1scale_dtype=None,
    x2scale_dtype=None,
    group_sizes=None,
    activation_type="gelu_tanh",
    quant_mode="mx",
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0
) -> Tuple[Tensor, Tensor]
```

## 参数说明

所有Tensor须位于同一NPU设备，且不支持空Tensor。下表中的数据类型和shape均按逻辑数据描述，FP4的打包存储及实际Tensor形状见[布局与存储](#布局与存储)。`batch1`、`batch2`表示各输入前0～4个batch维，`batch`表示右对齐广播后的batch维；没有batch维时省略该部分。ceil表示向上取整。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x1 | Tensor | 必选 | 左矩阵，对应公式中的$X_1$，使用ND格式。 | torch.float8_e4m3fn、torch.float8_e5m2、torch_npu.float4_e2m1fn_x2 | 2～6维，`(batch1,M,K)` |
| x2 | Tensor | 必选 | 右矩阵，对应公式中的$X_2$，使用ND或FRACTAL_NZ格式。NZ不支持torch.float8_e5m2。 | torch.float8_e4m3fn、torch.float8_e5m2、torch_npu.float4_e2m1fn_x2 | 2～6维，`(batch2,K,N)` |
| x2_scale | Tensor | 必选 | x2的分组scale，对应公式中的$S_2$，使用ND格式。 | torch.float8_e8m0fnu | 3～7维，`(batch2,ceil(K/64),N,2)` |
| x1_scale | Tensor | 可选 | x1的分组scale，对应公式中的$S_1$，使用ND格式。默认None；本接口仅支持MX模式，调用时必须传入非None的Tensor。 | torch.float8_e8m0fnu | 3～7维，`(batch1,M,ceil(K/64),2)` |
| bias | Tensor | 可选 | 激活前相加的偏置，对应公式中的$b$，使用ND格式。默认None，表示不加偏置。支持`(N,)`；仅当y为三维时，才支持`(B,1,N)`，B须等于广播后的batch大小。SwiGLU的bias长度仍为激活前的N。 | torch.float32 | `(N,)`或`(B,1,N)` |
| output_dtype | int | 可选 | y的逻辑数据类型标识。默认None，使用x1的逻辑类型；取值范围同x1_dtype，且指定的类型须与x1的逻辑类型一致。 | - | - |
| x1_dtype | int | 可选 | x1的逻辑数据类型标识。默认None，按x1.dtype推导。FP8可显式传入torch_npu.float8_e5m2或torch_npu.float8_e4m3fn；FP4必须传入torch_npu.float4_e2m1fn_x2。 | - | - |
| x2_dtype | int | 可选 | x2的逻辑数据类型标识。默认None，按x2.dtype推导。取值范围同x1_dtype，但NZ不支持torch_npu.float8_e5m2；FP4必须传入torch_npu.float4_e2m1fn_x2。 | - | - |
| x1scale_dtype | int | 可选 | x1_scale的逻辑数据类型标识。默认None，按x1_scale.dtype推导；显式指定时使用torch_npu.float8_e8m0fnu。 | - | - |
| x2scale_dtype | int | 可选 | x2_scale的逻辑数据类型标识。默认None，按x2_scale.dtype推导；显式指定时使用torch_npu.float8_e8m0fnu。 | - | - |
| group_sizes | List[int] | 可选 | 输入量化在M、N、K方向的分组大小，按`[groupSizeM,groupSizeN,groupSizeK]`排列。默认None；仅支持None、空列表、`[0,0,0]`或`[1,1,32]`，均表示使用`[1,1,32]`分组。 | - | - |
| activation_type | str | 可选 | 激活类型。支持"gelu_tanh"、"gelu_erf"、"silu"、"swiglu"，默认"gelu_tanh"。 | - | - |
| quant_mode | str | 可选 | 量化模式。仅支持"mx"，默认"mx"。 | - | - |
| round_mode | str | 可选 | 转换为输出数据类型时的舍入模式。默认"rint"，舍入到最近的可表示值，中间值取尾数最低位为0的值；FP8仅支持此模式。FP4还支持"floor"（向负无穷方向舍入）和"round"（舍入到最近的可表示值，中间值向远离0的方向舍入）。 | - | - |
| scale_alg | int | 可选 | 输出scale的计算方法：0为OCP（Open Compute Project，开放计算项目）共享指数算法，1按FP8最大有限值计算，2按dst_type_max指定的范围计算。FP8支持0、1，FP4支持0、2。默认0，None等价于0。 | - | - |
| dst_type_max | float | 可选 | scale_alg=2时计算输出scale所用的参考上限，对应量化公式中的d。默认0.0，按6.0处理；还支持[6.0,12.0]内的有限值。其他算法忽略该参数。 | - | - |

类型参数中的`torch_npu`常量是整数类型标识；使用原生FP8或E8M0 Tensor时，对应参数可保持None。显式指定输入类型只改变数据的解释方式，不执行数值转换，数据须已按对应格式存储。

## 返回值说明

返回`(y, y_scale)`，两者均位于x1所在的NPU设备。下表中的数据类型为逻辑类型，shape为返回Tensor的实际形状。H为激活后的逻辑输出宽度：GELU、SiLU取N，SwiGLU取N/2。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| y | Tensor | 必选 | 动态量化结果，对应公式中的$y$，逻辑类型与x1一致；当前扩展的FP4输出以torch.uint8打包返回。 | torch.float8_e4m3fn、torch.float8_e5m2、torch_npu.float4_e2m1fn_x2 | FP8为`(batch,M,H)`；FP4为`(batch,M,H/2)` |
| y_scale | Tensor | 必选 | 每32个输出元素对应的scale，即公式中的$s$；两个相邻scale存放在长度为2的末维中。 | torch.float8_e8m0fnu | `(batch,M,ceil(H/64),2)` |

## 约束说明

- 该接口用于量化前向计算，当前扩展未提供反向梯度接口。
- 该接口支持单算子模式调用，当前扩展未提供TorchAir图转换适配。

### 数据类型与形状

- x1、x2的逻辑维数为2～6；K必须相等，batch维按右对齐方式广播。两个输入scale的batch维须分别与对应矩阵逐维一致。
- 各逻辑维度及广播后的batch维乘积不超过$2^{32}-1$，激活前输出元素数不超过$2^{63}-1$。
- 两个输入scale和输出scale均为torch.float8_e8m0fnu。

矩阵及输出支持以下逻辑数据类型组合：

| x1 | x2 | y | x2格式 |
| --- | --- | --- | --- |
| torch.float8_e4m3fn | torch.float8_e4m3fn | torch.float8_e4m3fn | ND、FRACTAL_NZ |
| torch.float8_e5m2 | torch.float8_e4m3fn | torch.float8_e5m2 | ND、FRACTAL_NZ |
| torch.float8_e4m3fn | torch.float8_e5m2 | torch.float8_e4m3fn | ND |
| torch.float8_e5m2 | torch.float8_e5m2 | torch.float8_e5m2 | ND |
| torch_npu.float4_e2m1fn_x2 | torch_npu.float4_e2m1fn_x2 | torch_npu.float4_e2m1fn_x2 | ND、FRACTAL_NZ |

下表中，FP8指torch.float8_e4m3fn或torch.float8_e5m2，FP4指torch_npu.float4_e2m1fn_x2；同时须满足上方的数据类型组合要求。

| activation_type | 数据类型 | N的要求 | x2为ND时的x1布局 | x2为NZ时的x1布局 |
| --- | --- | --- | --- | --- |
| "gelu_tanh"、"gelu_erf" | FP8 | N > 0 | 支持转置视图 | 支持转置视图 |
| "gelu_tanh"、"gelu_erf" | FP4 | 正偶数 | 支持转置视图 | 不支持转置视图 |
| "silu" | FP8 | N > 0 | 支持转置视图 | 不支持转置视图 |
| "swiglu" | FP8 | 64的正整数倍 | 支持转置视图 | 不支持转置视图 |

上述组合均支持x2的转置视图。x2为NZ时，K、N均须大于1。FP4仅支持GELU，且K > 2、N为偶数。FP4输入在转置前的连续存储布局中，最内轴必须为偶数：x1不转置时为K、转置时为M；x2不转置时为N、转置时为K。FP4的NZ场景要求K、N均为偶数。

### 布局与存储

本接口通过Tensor视图表达转置。FP8输入的shape始终按`x1=(batch1,M,K)`、`x2=(batch2,K,N)`传入，scale的shape保持参数表所示的顺序。

例如，使用按`(K,M)`存储的x1和对应scale时，分别交换矩阵的最后两轴、scale的倒数第3和第2轴：

```python
# x1_base.shape=(K,M)，x1_scale_base.shape=(ceil(K/64),M,2)
x1 = x1_base.transpose(-1, -2)
x1_scale = x1_scale_base.transpose(-3, -2)
```

以下以`torch.uint8`为例说明FP4输入的打包布局：每字节包含两个FP4元素，低4位对应前一个元素，高4位对应后一个元素。沿打包轴，每个Tensor元素对应两个逻辑元素，因此该轴的Tensor维度为逻辑维度的一半。输入的字节布局、shape、stride和storage_offset须符合这一约定。

调用时将`x1_dtype`和`x2_dtype`均设置为`torch_npu.float4_e2m1fn_x2`；`output_dtype`可省略或设置为相同常量。上述打包布局对应的Tensor形状如下，batch维省略：

| 输入 | 连续布局 | 转置视图 |
| --- | --- | --- |
| x1 | `(M,K/2)` | 由`(K,M/2)`转置得到`(M/2,K)` |
| x2 | `(K,N/2)` | 由`(N,K/2)`转置得到`(K/2,N)` |

scale的shape按逻辑M、N、K计算。每32个K元素对应一个输入scale，相邻两组scale存放在长度为2的末维中；scale本身使用E8M0，不按FP4方式打包。

ND输入支持[非连续Tensor](../../../docs/zh/context/non_contiguous_tensor.md)，stride和storage_offset须非负，且访问范围不得超出底层storage。NZ权重仅支持完整矩阵及其最后两维的转置视图，不支持截取部分NZ存储的切片，`storage_offset()`须为0。NZ格式转换可使用`torch_npu.npu_format_cast`，见下方调用示例。

## 确定性/Batch一致性

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - 确定性说明：默认支持确定性计算。
  - Batch一致性说明：默认支持[Batch一致性](../../../docs/zh/context/batch_consistency.md)：在相同运行环境下，保持N、K、数据类型、布局及激活/量化属性不变，且同一Token的x1、x1_scale及其对应的x2、x2_scale、bias数据不变时，仅改变batch大小、M大小、Token的位置或组合，该Token对应的y和y_scale保持逐比特一致，无需额外开启开关。
<!-- end id7 -->

## 调用示例

需安装配套的`torch`、`torch_npu`及本仓库的PyTorch扩展，安装方法见[PyTorch扩展说明](../../../torch_extension/README.md)。

### 单算子模式调用

#### FP8

以下示例使用SwiGLU和NZ权重。将`activation_type`改为`"gelu_tanh"`、`"gelu_erf"`或`"silu"`可使用其他激活；使用ND权重时，省略`npu_format_cast`这一行。

```python
import torch
import torch_npu
import cann_ops_nn

m, k, n = 5, 64, 128
activation_type = "swiglu"
x1 = torch.randn(m, k).to(torch.float8_e4m3fn).npu()
x2 = torch.randn(k, n).to(torch.float8_e4m3fn).npu()
x2 = torch_npu.npu_format_cast(x2, 29)  # FRACTAL_NZ

scale_k = (k + 63) // 64
# E8M0的字节编码127表示scale=1。
x1_scale = torch.full((m, scale_k, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
x2_scale = torch.full((scale_k, n, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
bias = torch.zeros(n, dtype=torch.float32).npu()

y, y_scale = cann_ops_nn.quant_matmul_activation_quant(
    x1, x2, x2_scale, x1_scale=x1_scale, bias=bias,
    activation_type=activation_type, scale_alg=1,
)
h = n // 2 if activation_type == "swiglu" else n
assert tuple(y.shape) == (m, h)
assert tuple(y_scale.shape) == (m, (h + 63) // 64, 2)
print(y.cpu(), y_scale.cpu())
```

#### FP4

以下示例使用ND权重和GELU，以torch.uint8承载打包后的FP4输入。当前扩展返回的FP4输出y也以torch.uint8打包，scale的shape按逻辑输出宽度N计算。

```python
import torch
import torch_npu
import cann_ops_nn

m, k, n = 5, 64, 128
# 0x22表示两个值为1的FP4（E2M1）元素。
x1 = torch.full((m, k // 2), 0x22, dtype=torch.uint8).npu()
x2 = torch.full((k, n // 2), 0x22, dtype=torch.uint8).npu()
scale_k = (k + 63) // 64
x1_scale = torch.full((m, scale_k, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()
x2_scale = torch.full((scale_k, n, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu).npu()

y, y_scale = cann_ops_nn.quant_matmul_activation_quant(
    x1, x2, x2_scale, x1_scale=x1_scale,
    x1_dtype=torch_npu.float4_e2m1fn_x2,
    x2_dtype=torch_npu.float4_e2m1fn_x2,
    output_dtype=torch_npu.float4_e2m1fn_x2,
    activation_type="gelu_tanh", scale_alg=2, dst_type_max=6.0,
)
assert tuple(y.shape) == (m, n // 2)
assert tuple(y_scale.shape) == (m, (n + 63) // 64, 2)
print(y.cpu(), y_scale.cpu())
```
