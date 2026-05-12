# SparseTensorDenseMatMul

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|×|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|×|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|×|
|<term>Atlas 训练系列产品</term>|×|

## 功能说明

- 算子功能：稀疏矩阵乘法，将由 x1\_indices、x1\_values、x1\_shape 表示的稀疏矩阵 x1 乘以稠密矩阵 x2，得到结果矩阵 y。要求 x1、x2、y 的 rank 均为 2。

- 计算公式：

  $$ y = x1 \times x2 $$

  稀疏矩阵 x1 由三部分描述：非零元素索引 `x1_indices`（shape 为 [nnz, 2]）、非零元素值 `x1_values`（shape 为 [nnz]）、矩阵形状 `x1_shape`（shape 为 [2]，值为 [m, n]）。稠密矩阵 x2 的 shape 为 [n, p]。

  当 adjoint\_a=false、adjoint\_b=false 时：
  
  $$y_{i,j} = \sum_{k} x1_{i,k} \cdot x2_{k,j}$$

  其中 $x1_{i,k}$ 为 0（不在x1\_indices 中）或取 x1\_values 中对应位置的值。

  当 adjoint\_a=true 时，x1 先转置再参与矩阵乘；当 adjoint\_b=true 时，x2 先转置再参与矩阵乘。

## 参数说明

<table>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x1_indices</td>
      <td>输入</td>
      <td>稀疏矩阵中非零元素的索引，形状为 [nnz, 2]，其中 nnz 为非零元素个数。</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1_values</td>
      <td>输入</td>
      <td>稀疏矩阵中非零元素的值，形状为 [nnz]。</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1_shape</td>
      <td>输入</td>
      <td>稀疏矩阵的稠密形状，形状为 [2]，值为 [m, n] （若 x1 转置，即adjoint_a输入true，则值为 [n, m]）。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>稠密矩阵，形状为 [n, p]（若 x2 转置，则形状为 [p, n]）。</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出稠密矩阵，形状为 [m, p]，数据类型与 x1_values 一致。</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>adjoint_a</td>
      <td>属性</td>
      <td>是否对稀疏矩阵 x1 取转置，默认为 false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>adjoint_b</td>
      <td>属性</td>
      <td>是否对稠密矩阵 x2 取转置，默认为 false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- x1\_values 和 x2 的数据类型必须一致。
- x1\_indices 支持 INT32 和 INT64 两种类型，与 x1\_values 的类型自由组合。
- m、n、p、nnz、`m*p`、`n*p`、`nnz*p` 的值需在 `[0, INT32_MAX)` 范围内。
- nnz 的值需在 `[0, m*n]` 范围内。
- 不支持确定性计算。
- 不支持非连续Tensor。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用  | [test_geir_sparse_tensor_dense_mat_mul](./examples/test_geir_sparse_tensor_dense_mat_mul.cpp) | 通过[算子IR](op_graph/sparse_tensor_dense_mat_mul_proto.h)等方式调用SparseTensorDenseMatMul算子。 |
