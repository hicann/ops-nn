# ForeachAddScalar

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：将指定的张量值加到张量列表中的每个张量中。

- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {x}_{i}+{scalar} (i=0,1,...n-1)
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
      <td>表示加法运算的输入张量列表，对应公式中的`x`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>输入</td>
      <td>表示加法运算的输入张量，对应公式中的`scalar`。元素个数为1。</td>
      <td>FLOAT32、FLOAT16、INT32</td><!--V2多了double、int64-->
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示加法运算的输出张量列表，对应公式中的`y`。数据类型和数据格式与入参`x`的数据类型和数据格式一致，shape size大于等于入参`x`的shape size。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 当`x`的数据类型为FLOAT32、FLOAT16、INT32时，`scalar`数据类型与`x`的数据类型保持一致。
    - 当`x`的数据类型为BFLOAT16时，`scalar`数据类型支持FLOAT32。


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_add_scalar](examples/test_aclnn_foreach_add_scalar.cpp) | 通过[aclnnForeachAddScalar](docs/aclnnForeachAddScalar.md)接口方式调用ForeachAddScalar算子。 |
| aclnn接口  | [test_aclnn_foreach_add_scalar_v2](examples/test_aclnn_foreach_add_scalar_v2.cpp) | 通过[aclnnForeachAddScalarV2](docs/aclnnForeachAddScalarV2.md)接口方式调用ForeachAddScalar算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_add_scalar_proto.h)构图方式调用ForeachAddScalar算子。         |

<!--[test_geir_foreach_add_scalar](examples/test_geir_foreach_add_scalar.cpp)-->