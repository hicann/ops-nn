# SparseToDense

##  产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |
## 功能说明

- 算子功能: 将一个稀疏表示（Sparse Representation）转换为一个稠密张亮。

- 示例：

  ```
  # If sparse_values is scalar
  dense[i] = (i == sparse_indices ? sparse_values : default_value)

  # If sparse_values is a vector, then for each i
  dense[sparse_indices[i]] = sparse_values[i]

  # If sparse_values is an n by d matrix, then for each i in [0, n)
  dense[sparse_indices[i][0], ... , sparse_indices[i][d-1]] = sparse_values[i]
  ```

  在计算时需要满足以下要求：
  - sparse_indices只能是0/1/2维。
  - sparse_values只能是0/1维。
  - 当sparse_values是1维时，sparse_values的size必须等于sparse_indices的0轴长度。
  - default_value的size必须为1。
  - sparse_indices每一行对应的索引不重复且升序排列。
  - sparse_indices每一行索引必须为有效索引。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>indices</td>
      <td>输入</td>
      <td>公式中的`sparse_indices`，Device侧的aclTensor。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_shape</td>
      <td>输入</td>
      <td>用来指定输出Tensor的形状和大小，数据类型与indices相同。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>values</td>
      <td>输入</td>
      <td>公式中的`sparse_values`，Device侧的aclTensor。</td>
      <td>UINT8、INT8、UINT16、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、BFLOAT16。</td>
      <td>ND</td>
    </tr>
      <td>default_value</td>
      <td>输入</td>
      <td>公式中的`default_value`，Device侧的aclTensor，数据类型与values相同。</td>
      <td>UINT8、INT8、UINT16、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>validate_indices</td>
      <td>输入</td>
      <td>Host侧的布尔型，判断是否校验indices有效性(预留，此参数当前不生效)。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出，数据类型与values相同。</td>
      <td>UINT8、INT8、UINT16、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir]   | 通过[算子IR](./op_graph/sparse_to_dense_proto.h)构图方式调用SparseToDense算子。 |
