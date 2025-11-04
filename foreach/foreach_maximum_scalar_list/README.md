# ForeachMaximumScalarList

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：返回一个和输入张量列表同样形状大小的新张量列表, 对张量列表x和标量列表scalar执行逐元素比较，返回最大值的张量。

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
      <td>表示取最大值运算的输入张量列表，对应公式中的`x1`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>输入</td>
      <td>表示取最大值运算的输入标量列表，对应公式中的`scalars`。数据类型与输入`x`的数据类型具有一定对应关系：当入参`x`的数据类型为FLOAT32、FLOAT16、BFLOAT16时，`scalars`的数据类型支持FLOAT32。<1--、DOUBLE-->；当入参`x`的数据类型为INT32时，`scalars`的数据类型仅支持INT64。</td>
      <td>FLOAT32、INT64</td><!--aclnn支持DOUBLE,IR原型没有-->
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

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | - | 通过[aclnnForeachMaximumScalarList](docs/aclnnForeachMaximumScalarList.md)接口方式调用ForeachMaximumScalarList算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_maximum_scalar_list_proto.h)构图方式调用ForeachMaximumScalarList算子。         |

<!--[test_geir_foreach_maximum_scalar_list](examples/test_geir_foreach_maximum_scalar_list.cpp)-->
<!--[test_aclnn_foreach_maximum_scalar_list](examples/test_aclnn_foreach_maximum_scalar_list.cpp-->