# TransposeQuantBatchMatMul

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>    |    √    |
| <term>Atlas A2系列产品</term> |    √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：完成张量x1与张量x2的量化矩阵乘计算，支持K-C、MX、T-C[量化模式](../../docs/zh/context/quant_mode_introduction.md)。仅支持三维的Tensor传入。

  Tensor支持转置，转置序列由perm_x1、perm_x2、perm_y指定。perm_x1表示张量x1的转置序列，仅支持[1, 0, 2]；perm_x2表示张量x2的转置序列，K-C模式支持[0, 1, 2]，MX和T-C模式支持[0, 1, 2]和[0, 2, 1]；perm_y表示矩阵乘输出矩阵的转置序列，仅支持[1, 0, 2]。序列值为0的维度是batch维度，其余两个维度进行矩阵乘法。

  x1_scale和x2_scale表示量化缩放因子，其shape和数据类型随产品和量化模式变化；bias为预留参数，当前暂不支持。详细约束条件见[约束说明](#约束说明)或[aclnnTransposeQuantBatchMatMul](docs/aclnnTransposeQuantBatchMatMul.md)调用说明文档。

- 计算公式：

  记转置后的x1为$[B, M, K]$、x2为$[B, K, N]$，矩阵乘中间结果为$mm$，最终输出为$y$。

  1. K-C量化模式：

     $$
     mm[b,m,n] = \sum_{k=0}^{K-1}(x1[b,m,k] \cdot s1[m,b]) \cdot (x2[b,k,n] \cdot s2[b,n])
     $$

     其中，$s1$和$s2$分别为x1_scale和x2_scale按约束说明中的shape广播后的值。

  2. MX量化模式（MXFP8和MXFP4，group size为32）：

     $$
     mm[b,m,n] = \sum_{k=0}^{K-1}(x1[b,m,k] \cdot s1[m,b,\lfloor k/64 \rfloor,(\lfloor k/32 \rfloor) \bmod 2]) \cdot (x2[b,k,n] \cdot s2[b,\lfloor k/64 \rfloor,n,(\lfloor k/32 \rfloor) \bmod 2])
     $$

     其中，$s1$和$s2$为FLOAT8_E8M0类型的缩放因子，$s1$的shape为$(M, B, \lceil K/64 \rceil, 2)$；$s2$的shape在perm_x2为[0, 1, 2]时为$(B, \lceil K/64 \rceil, N, 2)$，此时按下标$[b, \lfloor k/64 \rfloor, n, (\lfloor k/32 \rfloor) \bmod 2]$取值，在perm_x2为[0, 2, 1]时为$(B, N, \lceil K/64 \rceil, 2)$，此时按下标$[b, n, \lfloor k/64 \rfloor, (\lfloor k/32 \rfloor) \bmod 2]$取值。最后一维的2表示每64个K元素包含2个32元素分组的缩放因子。

  3. T-C量化模式：

     $$
     mm[b,m,n] = \sum_{k=0}^{K-1}(x1[b,m,k] \cdot s1) \cdot (x2[b,k,n] \cdot s2[n])
     $$

     其中，$s1$为x1_scale，x1_scale为空时$s1$取1.0，x1_scale为(1,)时$s1$取该元素的值；$s2[n]$为x2_scale按UINT64/INT64位模式解析得到的per-channel缩放因子，每个元素的低32位为FLOAT32位模式。

  最后对矩阵乘中间结果按perm_y做转置，得到输出：

  $$
  y = \text{permute}(mm, perm\_y)
  $$

- 示例：
  假设x1的shape是[M, B, K]，x2的shape是[B, K, N]，x1_scale和x2_scale不为None，batch_split_factor等于1时，计算输出y的shape是[M, B, N]。

## 参数说明

<table class="tg" style="undefined;table-layout: fixed; width: 1034px"><colgroup>
<col style="width: 120px">
<col style="width: 123px">
<col style="width: 352px">
<col style="width: 320px">
<col style="width: 125px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x1</td>
    <td>输入</td>
    <td>矩阵乘运算中的左矩阵，对应公式中的x1。</td>
    <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、HIFLOAT8、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>矩阵乘运算中的右矩阵，对应公式中的x2。x2支持ND和FRACTAL_NZ格式，FRACTAL_NZ格式的支持范围随产品不同，详见约束说明。</td>
    <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、HIFLOAT8、INT8</td>
    <td>ND、FRACTAL_NZ</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>可选输入</td>
    <td>矩阵乘运算后累加的偏置，预留参数，当前暂不支持。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x1_scale</td>
    <td>可选输入</td>
    <td>x1的量化缩放因子，对应公式中的x1_scale，其shape随产品和量化模式变化，详见约束说明。</td>
    <td>FLOAT32、FLOAT8_E8M0、UINT64、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2_scale</td>
    <td>可选输入</td>
    <td>x2的量化缩放因子，对应公式中的x2_scale，其shape随产品和量化模式变化，详见约束说明。</td>
    <td>FLOAT32、FLOAT8_E8M0、UINT64、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dtype</td>
    <td>属性</td>
    <td>用于指定输出矩阵的数据类型，必填。取值1表示FLOAT16，27表示BFLOAT16，34表示HIFLOAT8（仅T-C量化模式支持）。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>group_size</td>
    <td>可选属性</td>
    <td>用于指定量化分组大小，默认值为0。由group_size_m、group_size_n、group_size_k三个值拼接组成，详见约束说明。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>perm_x1</td>
    <td>可选属性</td>
    <td>表示矩阵乘的第一个矩阵的转置序列，仅支持[1, 0, 2]，默认值为[1, 0, 2]。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>perm_x2</td>
    <td>可选属性</td>
    <td>表示矩阵乘的第二个矩阵的转置序列，支持[0, 1, 2]和[0, 2, 1]（K-C量化模式仅支持[0, 1, 2]），默认值为[0, 1, 2]。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>perm_y</td>
    <td>可选属性</td>
    <td>表示矩阵乘输出矩阵的转置序列，仅支持[1, 0, 2]，默认值为[1, 0, 2]。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>batch_split_factor</td>
    <td>可选属性</td>
    <td>用于指定矩阵乘输出矩阵中B维的切分大小，当前仅支持取值为1，默认值为1。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>矩阵乘运算的计算结果，对应公式中的y。</td>
    <td>FLOAT16、BFLOAT16、HIFLOAT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

- <term>Ascend 950PR&950DT系列产品</term>：x1和y仅支持ND格式，仅输入x2支持FRACTAL_NZ格式。x2为NZ格式时需通过[aclnnTransposeQuantBatchMatMulWeightNz](docs/aclnnTransposeQuantBatchMatMulWeightNz.md)接口调用，且仅支持MXFP8、MXFP4和HIFP8量化模式。

## 约束说明

- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持 INT8 W8A8 K-C。x1、x2 为 INT8，x1_scale、x2_scale 为 FLOAT32，y 为 BF16。x1 为 [M, B, K]，x2 为 [B, K, N]，x1_scale 为 [M, B]，x2_scale 为 [B, N]，y 为 [M, B, N]。perm_x1、perm_y 为 [1, 0, 2]，perm_x2 为 [0, 1, 2]。K 为 32 的倍数，N 为 16 的倍数，batch_split_factor 为 1。A2 上 B*K 小于 65536。x2 支持 ND 与 FRACTAL_NZ。
- <term>Ascend 950PR&950DT系列产品</term>：
    - perm_x1和perm_y仅支持[1, 0, 2]。
    - K-C量化场景，perm_x2支持输入[0, 1, 2]；MX和T-C量化场景，perm_x2支持输入[0, 1, 2]或[0, 2, 1]。
    - x1和y仅支持ND格式；x2支持ND和FRACTAL_NZ格式。x2为FRACTAL_NZ格式时，其storage shape为5维（逻辑shape仍为3维），仅支持MXFP8、MXFP4和HIFP8量化模式，且需通过[aclnnTransposeQuantBatchMatMulWeightNz](docs/aclnnTransposeQuantBatchMatMulWeightNz.md)接口调用。
    - K-C[量化模式](../../docs/zh/context/quant_mode_introduction.md)，K仅支持512，N仅支持128。x1_scale和x2_scale为1维，并且x1_scale为(M,)，x2_scale为(N,)，group_size仅支持配置为0，其他取值不生效。x1/x2输入支持FLOAT8_E5M2、FLOAT8_E4M3FN两种类型，x1_scale/x2_scale仅支持FLOAT32类型。
    - MX[量化模式](../../docs/zh/context/quant_mode_introduction.md)，支持MXFP8和MXFP4两种数据类型。K仅支持64的倍数。x1_scale和x2_scale为4维，并且x1_scale为(M, B, K/64, 2)，x2_scale为(B, K/64, N, 2)或(B, N, K/64, 2)，group_size的group_size_m和group_size_n仅支持0或1，group_size_k仅支持32。x1/x2输入支持FLOAT8_E4M3FN、FLOAT4_E2M1数据类型，x1_scale/x2_scale仅支持FLOAT8_E8M0类型。
    - T-C[量化模式](../../docs/zh/context/quant_mode_introduction.md)，仅支持静态量化。x1_scale支持配置为空或(1,)，非空时仅支持UINT64/INT64类型；x2_scale为1维且为(N,)，需经过[trans_quant_param](../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2.md)预处理转换为UINT64/INT64类型；x1/x2仅支持HIFLOAT8类型，group_size配置不生效。
    - group_size相关约束：
        - 仅在MX量化场景中生效。
        - 传入的group_size内部会按如下公式分解得到group_size_m、group_size_n、group_size_k，group_size_m和group_size_n取值为0时由接口根据scale的shape推断，group_size_k仅支持32。原理：假设group_size_m=0，表示M方向量化分组值由接口推断，推断公式为group_size_m = M / scale_m（需保证M能被scale_m整除），其中M与x1 shape中的M一致，scale_m与x1_scale shape中的M一致。

        $$
        group\_size = group\_size\_k\ |\ group\_size\_n << 16\ |\ group\_size\_m << 32
        $$
    - 不支持空Tensor。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_transpose_quant_batch_mat_mul](examples/arch22/test_aclnn_transpose_quant_batch_mat_mul.cpp)<br>[test_aclnn_quant_batch_mat_mul](examples/arch35/test_aclnn_transpose_quant_batch_mat_mul.cpp) | 通过<br>- [aclnnTransposeQuantBatchMatMul](docs/aclnnTransposeQuantBatchMatMul.md)<br>- [aclnnTransposeQuantBatchMatMulWeightNz](docs/aclnnTransposeQuantBatchMatMulWeightNz.md)<br>等方式调用TransposeQuantBatchMatMul算子。arch22 样例为 Atlas A2/A3 的 INT8 W8A8，arch35 样例为 Ascend 950 的 FP8。|
| torch接口  | [torchapi_transpose_quant_batch_mat_mul](docs/torchapi_transpose_quant_batch_mat_mul.md) | 通过`cann_ops_nn.transpose_quant_batch_mat_mul`调用TransposeQuantBatchMatMul算子，支持MX（MXFP8/MXFP4）、FP8（K-C）、HIFP8与INT8 W8A8量化模式。|
