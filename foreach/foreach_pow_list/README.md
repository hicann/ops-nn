# ForeachPowList

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行x2次方运算的结果。
- 计算公式：
  
  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}]\\
  x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x1_i^{x2_i}  (i=0,1,...n-1)
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
      <td>x1</td>
      <td>输入</td>
      <td>表示进行x2次方运算的底数，对应公式中的`x1`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示进行x2次方运算的指数，对应公式中的`x2`。该参数中所有Tensor的数据类型保持一致。数据类型和shape与输入`x1`的保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示进行x2次方运算的输出结果，对应公式中的`y`。该参数中所有Tensor的数据类型保持一致。数据类型与入参`x1`的数据类型一致，shapesize大于等于入参`x1`的shapesize。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_pow_list](examples/test_aclnn_foreach_pow_list.cpp) | 通过[aclnnForeachPowList](docs/aclnnForeachPowList.md)接口方式调用ForeachPowList算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_pow_list_proto.h)构图方式调用ForeachPowList算子。         |

<!--[test_geir_foreach_pow_list](examples/test_geir_foreach_pow_list.cpp)-->