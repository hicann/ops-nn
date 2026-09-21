# TransposeQuantBatchMatMul

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT系列产品</term>                              |    √     |
| <term>Atlas A3系列产品</term>     |    ×    |
| <term>Atlas A2系列产品</term> |    ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>     |     ×    |

## 功能说明

- 算子功能：完成张量x1与张量x2量化的矩阵乘计算，支持K-C、MX、T-C[量化模式](../../docs/zh/context/quant_mode_introduction.md)。仅支持三维的Tensor传入。Tensor支持转置，转置序列根据传入的序列进行变更。permX1代表张量x1的转置序列，支持[1,0,2]，permX2代表张量x2的转置序列，K-C模式支持[0,1,2]，MX和T-C模式支持[0,1,2]和[0,2,1]，permY表示矩阵乘输出矩阵的转置序列，当前仅支持[1,0,2]，序列值为0的是batch维度，其余两个维度做矩阵乘法。x1Scale和x2Scale表示输出矩阵的量化系数；bias为预留参数，当前暂不支持，详细约束条件可见约束说明或者[aclnnTransposeQuantBatchMatMul](docs/aclnnTransposeQuantBatchMatMul.md)调用说明文档。

- 示例：
  假设x1的shape是[M, B, K]，x2的shape是[B, K, N]，x1Scale和x2Scale不为None，batchSplitFactor等于1时，计算输出out的shape是[M, B, N]。

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
    <td>矩阵乘运算中的左矩阵。</td>
    <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>矩阵乘运算中的右矩阵。</td>
    <td>FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, HIFLOAT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>矩阵乘运算后累加的偏置，预留参数。</td>
    <td>FLOAT32, FLOAT16, BF16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x1Scale</td>
    <td>输入</td>
    <td>量化参数的缩放因子。</td>
    <td>FLOAT32、FLOAT8_E8M0、UINT64、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2Scale</td>
    <td>输入</td>
    <td>量化参数的缩放因子。</td>
    <td>FLOAT32、FLOAT8_E8M0、UINT64、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dtype</td>
    <td>输入</td>
    <td>用于指定输出矩阵的数据类型。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>groupSize</td>
    <td>输入</td>
    <td>用于指定量化分组大小。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>permX1</td>
    <td>输入</td>
    <td>表示矩阵乘的第一个矩阵的转置序列。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>permX2</td>
    <td>输入</td>
    <td>表示矩阵乘的第二个矩阵的转置序列。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>permY</td>
    <td>输入</td>
    <td>表示矩阵乘输出矩阵的转置序列。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>batchSplitFactor</td>
    <td>输入</td>
    <td>用于指定矩阵乘输出矩阵中B维的切分大小，当前仅支持取值为1。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>矩阵乘运算的计算结果。</td>
    <td>FLOAT16, BFLOAT16, HIFLOAT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

- <term>Ascend 950PR&950DT系列产品</term> ：只有输入x2支持FRACTAL_NZ格式。

## 约束说明

- <term>Ascend 950PR&950DT系列产品</term> ：
    - permX1和permY支持[1, 0, 2]。
    - K-C量化场景，permX2支持输入[0, 1, 2]；MX和T-C量化场景，permX2支持输入[0, 1, 2]或[0, 2, 1]。
    - K-C[量化模式](../../docs/zh/context/quant_mode_introduction.md)，K仅支持512，N仅支持128。x1Scale和x2Scale为1维，并且x1Scale为(M,)，x2Scale为(N,)，group_size仅支持配置为0，其他取值不生效。x1/x2输入支持FLOAT8_E5M2、FLOAT8_E4M3FN两种类型，x1Scale/x2Scale仅支持FLOAT32类型。
    - MX[量化模式](../../docs/zh/context/quant_mode_introduction.md)，支持MXFP8和MXFP4两种数据类型。K仅支持64的倍数。x1Scale和x2Scale为4维，并且x1Scale为(M, B, K/64, 2)，x2Scale为(B, K/64, N, 2)或(B, N, K/64, 2)，group_size的groupSizeM和groupSizeN仅支持0或1，groupSizeK仅支持32。x1/x2输入支持FLOAT8_E4M3FN、FLOAT4_E2M1数据类型，x1Scale/x2Scale仅支持FLOAT8_E8M0类型。
    - T-C[量化模式](../../docs/zh/context/quant_mode_introduction.md)，仅支持静态量化。x1Scale支持配置为空或(1,)，非空时仅支持UINT64/INT64类型；x2Scale为1维且为(N,)，需经过[trans_quant_param](../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2.md)预处理转换为UINT64/INT64类型；x1/x2仅支持HIFLOAT8类型，group_size配置不生效。
    - groupSize相关约束：
        - 仅在MX量化场景中生效。
        - 传入的groupSize内部会按如下公式分解得到groupSizeM、groupSizeN、groupSizeK，groupSizeM和groupSizeN取值为0时由接口根据scale的shape推断，groupSizeK仅支持32。原理：假设groupSizeM=0，表示M方向量化分组值由接口推断，推断公式为groupSizeM = M / scaleM（需保证M能被scaleM整除），其中M与x1 shape中的M一致，scaleM与x1Scale shape中的M一致。

        $$
        groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
        $$
    - 不支持空Tensor。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_quant_batch_mat_mul](examples/arch35/test_aclnn_transpose_quant_batch_mat_mul.cpp) | 通过<br>- [aclnnTransposeQuantBatchMatMul](docs/aclnnTransposeQuantBatchMatMul.md)<br>- [aclnnTransposeQuantBatchMatMulWeightNz](docs/aclnnTransposeQuantBatchMatMulWeightNz.md)<br>等方式调用TransposeQuantBatchMatMul算子。|
| torch接口  | [torchapi_transpose_quant_batch_mat_mul](docs/torchapi_transpose_quant_batch_mat_mul.md) | 通过`cann_ops_nn.transpose_quant_batch_mat_mul`调用TransposeQuantBatchMatMul算子，仅支持MX量化模式（MXFP8/MXFP4）。|
