# rotate_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

对输入最后一维按长度 K 划分，每段与对应的旋转矩阵相乘，再对乘积进行量化，返回量化结果和缩放因子。

令 B=N/K，输入第 m 行的第 b 段为 $X_{m,b}\in\mathbb{R}^{K}$，则旋转结果为：

$$
Z_{m,b}=X_{m,b}R_b.
$$

rotation 为 (K,K) 时所有段共享同一旋转矩阵；为 (B,K,K) 时第 b 段使用 rotation[b]。分组模式沿 M 轴连续划分 X，第 e 组为 $X_e$，计算规则为：

$$
Z_{e,m,b}=X_{e,m,b}R_{e,b}.
$$

rotation 为 (E,K,K) 时组内所有段共享 rotation[e]；为 (E,B,K,K) 时使用 rotation[e,b]。各组的旋转结果按原 M 轴位置组成输出。

完整数学语义分别见 [aclnnRotateQuant](aclnnRotateQuant.md) 和 [aclnnRotateQuantV2](aclnnRotateQuantV2.md)。

## 函数原型

```python
cann_ops_nn.rotate_quant(
x, rotation, *, alpha=None, dst_dtype=None, axis=-1, round_mode="rint", scale_alg=0, dst_type_max=0.0, transpose_y=False, group_list=None, group_list_type=0)
    -> (Tensor y, Tensor scale)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x | Tensor | 必选 | 待旋转量化的输入张量。 | torch.float16、torch.bfloat16 | (*, N)或(M, N) |
| rotation | Tensor | 必选 | 旋转矩阵，数据类型与x一致。 | torch.float16、torch.bfloat16 | (K, K)或(N/K, K, K)或(E, K, K)或(E, N/K, K, K) |
| alpha | Tensor | 可选 | 裁剪系数，用于控制旋转结果的clamp范围。默认值None，表示不执行clamp。 | torch.bfloat16 | (1,) |
| dst_dtype | int | 可选 | 指定量化目标类型。支持torch.int8、torch.quint4x2、torch_npu.float4_e2m1fn_x2、torch.float8_e5m2和torch.float8_e4m3fn。默认值None，按torch.int8处理。 | - | - |
| axis | int | 可选 | 量化轴，只支持-1或x.dim()-1。默认值-1。 | - | - |
| round_mode | str | 可选 | 量化结果的取整方式，支持rint、round、floor。默认值rint。 | - | - |
| scale_alg | int | 可选 | 缩放因子计算方法。默认值0。 | - | - |
| dst_type_max | float | 可选 | 量化目标类型的最大值。scale_alg为2时取值范围为[6, 12]，其他情况仅支持0.0。默认值0.0。 | - | - |
| transpose_y | bool | 可选 | 是否对输出y进行转置，当前仅支持False。默认值False。 | - | - |
| group_list | Tensor | 可选 | 代表量化分组大小分布，E为分组数，满足1≤E≤1024，默认值None。 | torch.int64 | (E,) |
| group_list_type | int | 可选 | group_list的分组表示方式：0表示累计行数（cumsum），1表示各组行数（count）。默认值0； | - | - |

## 返回值说明

| 输出名 | 输出类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| y | Tensor | 必选 | 输出的量化结果。FP4场景使用torch.uint8承载打包编码，每字节存储两个FP4值，查看实际值需自行解包；其他场景的数据类型根据dst_dtype决定。数据格式支持ND。 | torch.int8、torch.int32、torch.uint8、torch.float8_e5m2、torch.float8_e4m3fn | INT8为(M, N)；INT4为(M, N/8)；FP4为(*, N/2)；FP8为(*, N) |
| scale | Tensor | 必选 | 输出的缩放因子。当dst_dtype为MX类型时，scale的数据类型为torch.float8_e8m0fnu，对应底层ACL的FLOAT8_E8M0类型；当dst_dtype为INT8或INT4类型时，scale为torch.float32类型的pertoken缩放因子。 | torch.float32、torch.float8_e8m0fnu | pertoken量化时为(M,)；MX量化时为(*, ceil(N/64), 2) |

## 约束说明

<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - group_list支持传入None。
  - group_list不为None时：
    - group_list_type必须从[0, 1]中取值。
    - group_list的shape为(E,)，E表示分组数，E需要大于等于1且小于等于1024。
    - group_list的数值需要满足以下条件，否则无法保证输出是否符合预期：
      - 当group_list_type为0时，group_list必须为非负单调非递减数列，表示分组后每组大小的cumsum结果（累计和），最后一个值应小于等于x中tensor的第一维。
      - 当group_list_type为1时，group_list必须为非负数列，表示分组后每组大小，数值的总和应小于等于x中tensor的第一维。
  - group_list传入None时，x的shape为(*, N)，维度范围[1, 7]；rotation的shape为(K, K)或(N/K, K, K)，维度范围[2, 3]。
  - group_list不为None时，x的shape为(M, N)，维度仅限二维；rotation的shape为(E, K, K)或(E, N/K, K, K)，维度范围[3, 4]。
  - K值仅支持取32，64，128，256。
  - x最后一维的长度(N)必须是K的整数倍。
  - 当dst_dtype为torch_npu.float4_e2m1fn_x2时，输入x最后一维的长度(N)必须同时可被2整除。
  - dst_dtype支持torch_npu.float4_e2m1fn_x2、torch.float8_e4m3fn或torch.float8_e5m2。
  - x和rotation的数据类型必须相同，必须同时为torch.bfloat16或torch.float16。
  - rotation最后两维的长度必须一致。
  - scale的shape为(*, ceil(N/64), 2)，维度支持2-8，返回的Tensor数据类型为torch.float8_e8m0fnu，对应底层ACL的FLOAT8_E8M0类型。
  - 当dst_dtype为FP4类型时，为兼容未提供原生FP4类型的Torch版本，y使用torch.uint8返回，每字节打包两个FP4值，shape为(*, N/2)，查看实际值需按FP4编码自行解包。当dst_dtype为FP8类型时，y的数据类型为对应的torch.float8_e4m3fn或torch.float8_e5m2，shape与x相同。
  - MX量化需要运行环境支持原生torch.float8_e8m0fnu，以保证scale返回E8M0类型；不支持时接口报错。
  - alpha为可选输入，不为None时，shape为(1,)，数据类型为torch.bfloat16，有效取值范围(0.0, 1.0)。传入None或者不在有效取值范围内不做clamp处理。
  - axis目前只支持-1或者D-1，D为x的shape的维数。
  - round_mode支持"rint"、"round"、"floor"，传入None时，采用"rint"模式。当dst_dtype为FP8类型时，round_mode仅支持"rint"。
  - scale_alg支持取值0、1、2，当dst_dtype为FP8类型时只支持0和1，当dst_dtype为torch_npu.float4_e2m1fn_x2时只支持0和2。
  - dst_type_max：当scale_alg=2时dst_type_max必须在[6.0, 12.0]范围内，其余场景仅支持0.0。
  - transpose_y目前只支持False。
<!-- end id9 -->
<!-- npu="A3,910b" id10 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - group_list仅支持传入None。
  - x的shape为(M, N)，rotation的shape为(K, K)。
  - rotation的shape必须是方阵(K, K)。
  - x第二维的长度(N)必须是K的整数倍，N必须可以整除8。
  - 当dst_dtype为torch.int8时，y的数据类型为torch.int8，shape与x相同，为(M, N)。
  - 当dst_dtype为torch.quint4x2时，y以torch.int32存储，每个元素打包八个INT4值，shape为(M, N//8)。
  - x和rotation的数据类型必须相同。
  - scale的shape为(M,)，返回的Tensor数据类型为torch.float32，对应底层ACL的FLOAT类型，表示pertoken缩放因子。
  - alpha为可选输入，当前版本仅支持传入None。
  - axis目前只支持-1或者1。
  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>的round_mode当前仅支持rint。
  <!-- end id7 -->
  - scale_alg目前只支持0。
  - dst_type_max目前只支持0.0。
  - transpose_y目前只支持False。
  - N的范围为[128, 16000]。
  - K的范围为[16, 1024]。
  - scale的shape仅支持1维。
<!-- end id10 -->

## 确定性计算

默认支持确定性计算

## 调用示例

单算子模式调用（eager）

```python
import torch
import torch_npu
import cann_ops_nn

M, N, K = 640, 4096, 128
x = torch.randn(M, N, dtype=torch.bfloat16, device="npu:0")
rotation = torch.randn(N // K, K, K, dtype=x.dtype, device=x.device)
y, scale = cann_ops_nn.rotate_quant(x, rotation, dst_dtype=torch.float8_e4m3fn)
```

带可选输入：

```python
import torch
import torch_npu
import cann_ops_nn

M, N, K, E = 640, 4096, 128, 4
x = torch.randn(M, N, dtype=torch.bfloat16, device="npu:0")
group_list = torch.tensor([160, 160, 160, 160], dtype=torch.int64, device=x.device)
group_cumsum = torch.tensor([160, 320, 480, 640], dtype=torch.int64, device=x.device)
alpha = torch.tensor([0.95], dtype=torch.bfloat16, device=x.device)

for rotation_shape in ((E, K, K), (E, N // K, K, K)):
    rotation = torch.randn(rotation_shape, dtype=x.dtype, device=x.device)
    y, scale = cann_ops_nn.rotate_quant(
        x, rotation, alpha=alpha, dst_dtype=torch_npu.float4_e2m1fn_x2,
        scale_alg=2, dst_type_max=6.0, group_list=group_list, group_list_type=1,
    )

    # 仅将组表改为 cumsum 表达，保持输入、rotation 和量化参数一致。
    y_cumsum, scale_cumsum = cann_ops_nn.rotate_quant(
        x, rotation, alpha=alpha, dst_dtype=torch_npu.float4_e2m1fn_x2,
        scale_alg=2, dst_type_max=6.0, group_list=group_cumsum, group_list_type=0,
    )

```
