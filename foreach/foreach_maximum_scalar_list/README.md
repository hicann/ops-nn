# ForeachMaximumScalarList

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：对张量列表x和标量列表scalars执行逐元素比较，计算每个元素对应的最大值。

- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \max(x_i, \text{scalars}_i) (i=0,1,...n-1)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x</td>
      <td>输入</td>
      <td>表示取最大值运算的输入张量列表，对应公式中的`x`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>输入</td>
      <td>表示取最大值运算的输入标量列表，对应公式中的`scalars`。元素个数与`x`中Tensor的个数相等。数据类型按`x`的dtype配对：当`x`为FLOAT32、FLOAT16、BFLOAT16时，`scalars`支持FLOAT32；当`x`为INT32时，`scalars`支持INT64。aclnn接口额外支持DOUBLE（`x`为浮点类型，内部转换为FLOAT32后参与计算）和INT32（`x`为INT32，内部转换为INT64后参与计算）。</td>
      <td>FLOAT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示取最大值运算的输出张量列表，对应公式中的`y`。数据类型和数据格式与入参`x`的数据类型和数据格式一致，shape size大于等于入参`x`的shape size。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：x和y的数据类型不支持BFLOAT16。

## 约束说明

- 输入`x`、输出`y`支持包含的最大Tensor个数均为256，且`y`的TensorList长度需要与`x`的TensorList长度相等。
- 输出`y`的shape size需大于等于`x`中对应位置Tensor的shape size。
- 输入`scalars`的ScalarList长度需要与`x`的TensorList长度相等，支持的最大元素个数为256。
- `scalars` 仅约束元素个数（ScalarList 长度）与 dtype，不限制其 Tensor rank；kernel 按扁平下标读取第 i 个元素。
- 支持空 TensorList：当 `x`、`y` 的 TensorList 长度均为 0 时，aclnn 接口与 kernel/图模式直调均直接返回成功（no-op）。
- 非连续Tensor支持情况：aclnn接口支持非连续的输入`x`和输出`y`（非连续输入先转换为连续Tensor参与计算，非连续输出由连续结果回写）；图模式由算子原型`AutoContiguous`在框架层处理；kernel直调仅支持连续Tensor。
- 输入`scalars`的DOUBLE数据类型仅在aclnn接口路径支持（内部转换为FLOAT32参与计算）；当`x`为INT32时，`scalars`的INT32数据类型也仅在aclnn接口路径支持（内部转换为INT64参与计算）；图模式（算子IR）下`scalars`仅支持FLOAT32、INT64。
- 上述长度、dtype、shape 等约束在 aclnn 接口路径下返回 ACLNN_ERR_PARAM_INVALID；图模式、算子直调等非 aclnn 路径下由 tiling 统一校验，失败时返回 tiling 错误（对应 ACLNN_ERR_INNER_TILING_ERROR 语义）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_maximum_scalar_list](examples/test_aclnn_foreach_maximum_scalar_list.cpp) | 通过[aclnnForeachMaximumScalarList](docs/aclnnForeachMaximumScalarList.md)接口方式调用ForeachMaximumScalarList算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_maximum_scalar_list_proto.h)构图方式调用ForeachMaximumScalarList算子。         |
