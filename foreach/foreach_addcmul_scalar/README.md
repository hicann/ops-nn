# ForeachAddcmulScalar

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：返回一个和输入张量列表同样形状大小的新张量列表，对张量列表x2和张量列表x3执行逐元素乘法，将结果乘以张量scalar后将结果与张量列表x1执行逐元素加法。
- 计算公式：
  
  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_{i} + {\rm scalar} × x2_{i} × x3_{i} (i=0,1,...n-1)
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
      <td>表示混合运算中加法的第一个输入张量列表，对应公式中的`x1`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示混合运算中乘法的第二个输入张量列表，对应公式中的`x2`。数据类型、数据格式和shape与入参`x1`的数据类型、数据格式和shape一致，该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>输入</td>
      <td>表示混合运算中乘法的第三个输入张量列表，对应公式中的`x3`。数据类型、数据格式和shape与入参`x1`的数据类型、数据格式和shape一致，该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>输入</td>
      <td>表示混合运算中乘法的第一个输入张量，对应公式中的`scalar`。元素个数为1。</td>
      <td>FLOAT32、FLOAT16、INT32</td><!--v2的aclnn多了DOUBLE和INT64数据类型，待确认-->
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示混合运算的输出张量列表，对应公式中的`y`。数据类型和数据格式与入参`x1`的数据类型和数据格式一致，shape size大于等于入参`x1`的shape size。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 当`x1`的数据类型为FLOAT32、FLOAT16、INT32时，`scalar`数据类型与`x1`的数据类型保持一致。
    - 当`x1`的数据类型为BFLOAT16时，`scalar`数据类型支持FLOAT32。


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_addcmul_scalar](examples/test_aclnn_foreach_addcmul_scalar.cpp) | 通过[aclnnForeachAddcmulScalar](docs/aclnnForeachAddcmulScalar.md)接口方式调用ForeachAddcmulScalar算子。 |
| aclnn接口  | [test_aclnn_foreach_addcmul_scalar_v2](examples/test_aclnn_foreach_addcmul_scalar_v2.cpp) | 通过[aclnnForeachAddcmulScalarV2](docs/aclnnForeachAddcmulScalarV2.md)接口方式调用ForeachAddcmulScalar算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_addcmul_scalar_proto.h)构图方式调用ForeachAddcmulScalar算子。         |

<!--[test_geir_foreach_addcmul_scalar](examples/test_geir_foreach_addcmul_scalar.cpp)-->