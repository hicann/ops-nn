# TransposeBatchMatMul

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR&950DT系列产品|√|
|Atlas A3系列产品|√|
|Atlas A2系列产品|√|
|Atlas 200I/500 A2推理产品|×|
|Atlas推理系列产品|×|
|Atlas训练系列产品|×|
|Kirin X90处理器系列产品|√|
|Kirin 9030处理器系列产品|√|

## 功能说明

- 算子功能：完成三维张量x1与x2的矩阵乘及可选bias计算。permX1/permX2指定输入轴位置；permY指定输出为(M,B,N)，Ascend 950PR&950DT系列产品还支持(B,M,N)。scale沿用原FLOAT16输入、INT8输出的量化语义。具体参数组合见[aclnnTransposeBatchMatMul](docs/aclnnTransposeBatchMatMul.md)。

- 示例：
  - x1的shape是(B, M, K)，x2的shape是(B, K, N)，scale为None，batchSplitFactor等于1时，计算输出out的shape是(M, B, N)。
  - x1的shape是(B, M, K)，x2的shape是(B, K, N)，scale不为None，batchSplitFactor等于1时，计算输出out的shape是(M, 1, B * N)。
  - x1的shape是(B, M, K)，x2的shape是(B, K, N)，scale为None，batchSplitFactor大于1时，计算输出out的shape是(batchSplitFactor, M, B * N / batchSplitFactor)。

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
    <td>FLOAT32, FLOAT16, BF16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>矩阵乘运算中的右矩阵。</td>
    <td>FLOAT32, FLOAT16, BF16</td>
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
    <td>scale</td>
    <td>输入</td>
    <td>量化参数的缩放因子。</td>
    <td>INT64, UINT64</td>
    <td>ND</td>
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
    <td>cubeMathType</td>
    <td>输入</td>
    <td>指定Cube单元的计算逻辑。</td>
    <td>INT8</td>
    <td>-</td>
  </tr>
  <tr>
    <td>batchSplitFactor</td>
    <td>输入</td>
    <td>用于指定矩阵乘输出矩阵中B维的切分大小。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>矩阵乘运算的计算结果。</td>
    <td>FLOAT32, FLOAT16, BF16, INT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

- Kirin X90/Kirin 9030处理器系列产品：不支持BFLOAT16。
- Atlas A2系列产品：只有输入x2支持FRACTAL_NZ格式。
- Atlas A3系列产品：只有输入x2支持FRACTAL_NZ格式。
- Ascend 950PR&950DT系列产品：只有输入x2支持FRACTAL_NZ格式。

## 约束说明

- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - 不支持空tensor。
    - 支持非连续tensor。
    - B的取值范围为[1, 65536)，N的取值范围为[1, 65536)。
    - 当x1的输入shape为(B, M, K)时，K <= 65535；当x1的输入shape为(M, B, K)时，B * K <= 65535。
    - 当scale不为空时，batchSplitFactor只能等于1，B与N的乘积小于65536，且仅支持输入为FLOAT16和输出为INT8的类型推导。
- <term>Ascend 950PR&950DT系列产品</term>：
    - 不支持空tensor。
    - 当scale不为空时，batchSplitFactor只能等于1，且仅支持输入为FLOAT16和输出为INT8的类型推导。
    - 支持可选ND bias，shape为(N,)或(1,N)，各batch和行共享；类型与x1相同或为FLOAT32。
    - A内轴转置[0,2,1]和ND KBN B[1,0,2]要求BMN输出、scale为空且batchSplitFactor=1。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_transpose_batch_mat_mul](examples/test_aclnn_transpose_batch_mat_mul.cpp) | 通过<br>- [aclnnTransposeBatchMatMul](docs/aclnnTransposeBatchMatMul.md)<br>- [aclnnTransposeBatchMatMulWeightNz](docs/aclnnTransposeBatchMatMulWeightNz.md)<br>等方式调用TransposeBatchMatMul算子。|
| torch接口  | [torchapi_transpose_batch_mat_mul](docs/torchapi_transpose_batch_mat_mul.md) | 通过`cann_ops_nn.transpose_batch_mat_mul`调用TransposeBatchMatMul算子，支持perm转置、batch_split_factor切分以及FLOAT16+scale的INT8量化输出，支持单算子模式与TorchAir图模式。|
