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

- 算子功能：稀疏矩阵乘法，将由x1\_indices、x1\_values、x1\_shape表示的稀疏矩阵x1乘以稠密矩阵x2，得到结果矩阵y。要求x1、x2、y的rank均为2。

- 计算公式：

  $$ y = x1 \times x2 $$

  稀疏矩阵x1由三部分描述：非零元素索引`x1_indices`（shape为 [nnz, 2]）、非零元素值`x1_values`（shape为 [nnz]）、矩阵形状`x1_shape`（shape为 [2]，值为 [m, n]）。稠密矩阵x2的shape为 [n, p]。

  当adjoint\_a=false、adjoint\_b=false时：
  
  $$y_{i,j} = \sum_{k} x1_{i,k} \cdot x2_{k,j}$$

  其中 $x1_{i,k}$ 为0（不在x1\_indices中）或取x1\_values中对应位置的值。

  当adjoint\_a=true时，x1先转置再参与矩阵乘；当adjoint\_b=true时，x2先转置再参与矩阵乘。

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
      <td>稀疏矩阵中非零元素的索引，形状为 [nnz, 2]，其中nnz为非零元素个数。</td>
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
      <td>稀疏矩阵的稠密形状，形状为 [2]，值为 [m, n]（若x1转置，即adjoint_a输入true，则值为 [n, m]）。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>稠密矩阵，形状为 [n, p]（若x2转置，则形状为 [p, n]）。</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出稠密矩阵，形状为 [m, p]，数据类型与x1_values一致。</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>adjoint_a</td>
      <td>属性</td>
      <td>是否对稀疏矩阵x1取转置，默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>adjoint_b</td>
      <td>属性</td>
      <td>是否对稠密矩阵x2取转置，默认为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- x1\_values和x2的数据类型必须一致。
- x1\_indices支持INT32和INT64两种类型，与x1\_values的类型自由组合。
- m、n、p、nnz、`m*p`、`n*p`、`nnz*p`的值需在`[0, INT32_MAX)`范围内。
- nnz的值需在`[0, m*n]`范围内。
- 不支持确定性计算。
- 不支持非连续Tensor。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用  | [test_geir_sparse_tensor_dense_mat_mul](./examples/test_geir_sparse_tensor_dense_mat_mul.cpp) | 通过[算子IR](op_graph/sparse_tensor_dense_mat_mul_proto.h)等方式调用SparseTensorDenseMatMul算子。 |
