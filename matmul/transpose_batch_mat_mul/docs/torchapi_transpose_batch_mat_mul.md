# transpose_batch_mat_mul

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

- 接口功能：完成三维张量x1与x2的矩阵乘计算，底层封装`aclnnTransposeBatchMatMul`；当x2为FRACTAL_NZ格式时，封装`aclnnTransposeBatchMatMulWeightNz`。x1/x2支持按`perm_x1`/`perm_x2`指定的转置序列计算，输出按`perm_y`排列。
- 示例：x1的shape是(M, B, K)，x2的shape是(B, K, N)时：
  - scale为None、batch_split_factor等于1：输出y的shape为(M, B, N)，dtype与x1一致；
  - scale不为None、batch_split_factor等于1：输出y的shape为(M, 1, B * N)，dtype为INT8；
  - scale为None、batch_split_factor大于1：输出y的shape为(batch_split_factor, M, B * N / batch_split_factor)。

## 函数原型

```python
cann_ops_nn.transpose_batch_mat_mul(
    x1,
    x2,
    *,
    bias=None,
    scale=None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    enable_hf32=False,
    batch_split_factor=None,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x1 | Tensor | 必选 | 矩阵乘运算中的左矩阵。默认`perm_x1=[1, 0, 2]`，即按(M, B, K)解释。 | torch.float16 / torch.bfloat16 / torch.float32 | 3维 |
| x2 | Tensor | 必选 | 矩阵乘运算中的右矩阵。默认`perm_x2=[0, 1, 2]`，即按(B, K, N)解释；支持FRACTAL_NZ（WeightNZ），该格式下仅支持FLOAT16、BFLOAT16。 | 同x1 | 3维 |
| bias | Tensor | 可选 | 偏置，预留参数，当前暂不支持。 | - | - |
| scale | Tensor | 可选 | 输出量化系数。传入时表示FLOAT16输入、INT8输出，shape必须为(B * N,)。scale需由`aclnnTransQuantParamV2`将FLOAT32转换为INT64/UINT64后传入。 | torch.int64 / torch.uint64 | 1维，(B * N,) |
| perm_x1 | List[int] | 可选 | x1的转置序列，支持[0, 1, 2]、[1, 0, 2]，默认[1, 0, 2]。 | int64 | - |
| perm_x2 | List[int] | 可选 | x2的转置序列，支持[0, 1, 2]、[0, 2, 1]，默认[0, 1, 2]。 | int64 | - |
| perm_y | List[int] | 可选 | 输出的转置序列，仅支持[1, 0, 2]，默认[1, 0, 2]。 | int64 | - |
| enable_hf32 | bool | 可选 | 是否开启HF32计算，仅对FLOAT32输入生效。内部映射为aclnn `cubeMathType`（True=3/USE_HF32，False=0/KEEP_DTYPE）。 | bool | - |
| batch_split_factor | int | 可选 | 输出矩阵B维的切分大小，取值范围为[1, B]且B能被其整除，默认1。scale不为None时只能为1。 | int64 | - |

## 返回值说明

| 输出名 | 输出类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| y | Tensor | 矩阵乘的输出。 | scale为None时与x1一致；scale不为None时为torch.int8 | scale为None且batch_split_factor=1时为(M, B, N)；scale不为None时为(M, 1, B * N)；batch_split_factor>1时为(batch_split_factor, M, B * N / batch_split_factor) |

## 约束说明

- 仅支持3维Tensor，x1/x2的batch轴和K轴必须一致，不支持batch轴广播。
- x1/x2的数据类型必须一致且为FLOAT16、BFLOAT16或FLOAT32。
- perm_x1支持[0, 1, 2]、[1, 0, 2]；perm_x2支持[0, 1, 2]、[0, 2, 1]；perm_y仅支持[1, 0, 2]。
- scale仅支持FLOAT16输入 + INT8输出；传入scale时batch_split_factor必须为1。
- bias为预留参数，当前暂不支持，请传入None。
- x2支持FRACTAL_NZ格式（WeightNZ），可通过`torch_npu.npu_format_cast(x2, 29)`构造；该格式下x1/x2仅支持FLOAT16、BFLOAT16。
- 不支持空Tensor。
- 该接口支持单算子模式和TorchAir图模式调用；图模式下由GE Converter转换为`TransposeBatchMatMul`图算子。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用（eager，不带scale）：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, B, K, N = 64, 16, 512, 128
  x1 = torch.randn(M, B, K, dtype=torch.float16).npu()
  x2 = torch.randn(B, K, N, dtype=torch.float16).npu()

  y = cann_ops_nn.transpose_batch_mat_mul(x1, x2)
  print(y.shape, y.dtype)  # torch.Size([64, 16, 128]) torch.float16
  ```

- 单算子模式调用（fp16输入 + scale，输出INT8）：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, B, K, N = 64, 16, 512, 128
  x1 = torch.randn(M, B, K, dtype=torch.float16).npu()
  x2 = torch.randn(B, K, N, dtype=torch.float16).npu()
  # scale需由aclnnTransQuantParamV2从FLOAT32转换为INT64/UINT64，这里构造占位数据
  scale = torch.ones(B * N, dtype=torch.int64).npu()

  y = cann_ops_nn.transpose_batch_mat_mul(x1, x2, scale=scale)
  print(y.shape, y.dtype)  # torch.Size([64, 1, 2048]) torch.int8
  ```

- batch_split_factor示例：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, B, K, N = 64, 16, 512, 128
  x1 = torch.randn(M, B, K, dtype=torch.float16).npu()
  x2 = torch.randn(B, K, N, dtype=torch.float16).npu()
  y = cann_ops_nn.transpose_batch_mat_mul(x1, x2, batch_split_factor=2)
  print(y.shape)  # torch.Size([2, 64, 1024])
  ```

- x2为FRACTAL_NZ（WeightNZ）：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, B, K, N = 64, 16, 512, 128
  x1 = torch.randn(M, B, K, dtype=torch.float16).npu()
  x2 = torch.randn(B, K, N, dtype=torch.float16).npu()
  x2_nz = torch_npu.npu_format_cast(x2, 29)  # 29为NZ格式
  y = cann_ops_nn.transpose_batch_mat_mul(x1, x2_nz)
  print(y.shape, y.dtype)
  ```

- 图模式（torchair）调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  @torch.compile(backend="npu")
  def func(x1, x2):
      return cann_ops_nn.transpose_batch_mat_mul(x1, x2)

  M, B, K, N = 64, 16, 512, 128
  x1 = torch.randn(M, B, K, dtype=torch.float16).npu()
  x2 = torch.randn(B, K, N, dtype=torch.float16).npu()
  y = func(x1, x2)
  print(y.shape, y.dtype)
  ```
