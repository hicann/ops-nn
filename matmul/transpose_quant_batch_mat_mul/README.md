# TransposeQuantBatchMatMul


##  产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：完成张量x1与张量x2量化的矩阵乘计算，支持K-C[量化模式](../../../docs/zh/context/量化介绍.md)。仅支持三维的Tensor传入。Tensor支持转置，转置序列根据传入的序列进行变更。permX1代表张量x1的转置序列，支持[1,0,2]，permX2代表张量x2的转置序列[0,1,2]，permY表示矩阵乘输出矩阵的转置序列，当前仅支持[1,0,2]，序列值为0的是batch维度，其余两个维度做矩阵乘法。x1_scale和x2_scale表示输出矩阵的量化系数；bias为预留参数，当前暂不支持，详细约束条件可见约束说明或者[aclnnTransposeQuantBatchMatMul](docs/aclnnTransposeQuantBatchMatMul.md)调用说明文档。

- 示例：
  假设x1的shape是[M, B, K]，x2的shape是[B, K, N]，x1_scale和x2_scale不为None，batchSplitFactor等于1时，计算输出out的shape是[M, B, N]。

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
    <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>矩阵乘运算中的右矩阵。</td>
    <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>矩阵乘运算后累加的偏置。</td>
    <td>FLOAT32, FLOAT16, BF16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x1Scale</td>
    <td>输入</td>
    <td>量化参数的缩放因子。</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2Scale</td>
    <td>输入</td>
    <td>量化参数的缩放因子。</td>
    <td>FLOAT32</td>
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
    <td>用于指定量化分组大小，预留参数，当前仅支持配置为0，其他取值不生效。</td>
    <td>INT32</td>
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
    <td>用于指定矩阵乘输出矩阵中N维的切分大小。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>矩阵乘运算的计算结果。</td>
    <td>FLOAT16, BF16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
    - permX1和permY支持[1, 0, 2], 支持permX2支持输入[0, 1, 2]。
    - x1Scale和x2Scale为1维，并且x1Scale为(M,), x2Scale为(N,)。
    - out和dtype支持float16和bfloat16。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_quant_batch_mat_mul](examples/arch35/test_aclnn_transpose_quant_batch_mat_mul.cpp) | 通过<br>[aclnnTransposeQuantBatchMatMul](docs/aclnnTransposeQuantBatchMatMul.md)<br>等方式调用TransposeQuantBatchMatMul算子。|
