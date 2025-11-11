# ForeachSubList

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对输入的两个张量列表执行逐元素相减运算，并可以通过alpha参数调整相减系数。

- 计算公式：
 
  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}]\\
  x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$
  
  $$
  y_i = x1_i-{x2_i}*alpha (i=0,1,...n-1)
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
      <td>表示进行相减运算的第一个输入，对应公式中的`x1`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示进行相减运算的第二个输入，对应公式中的`x2`。该参数中所有Tensor的数据类型保持一致。数据类型、数据格式和shape与入参`x1`的数据类型、数据格式和shape一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>表示进行相减运算的第二个输入的系数，对应公式中的`alpha`。元素个数为1。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16、DOUBLE、INT64</td><!--V2新增了DOUBLE、INT64；aclnn都没有BF16-->
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示进行相减运算的输出结果，对应公式中的`y`。该参数中所有Tensor的数据类型保持一致。数据类型和数据格式与入参`x1`的数据类型和数据格式一致，shape size大于等于入参`x1`的shape size。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_sub_list](examples/test_aclnn_foreach_sub_list.cpp) | 通过[aclnnForeachSubList](docs/aclnnForeachSubList.md)接口方式调用ForeachSubList算子。 |
| aclnn接口  | [test_aclnn_foreach_sub_list_v2](examples/test_aclnn_foreach_sub_list_v2.cpp) | 通过[aclnnForeachSubListV2](docs/aclnnForeachSubListV2.md)接口方式调用ForeachSubList算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_sub_list_proto.h)构图方式调用ForeachSubList算子。         |

<!--[test_geir_foreach_sub_list](examples/test_geir_foreach_sub_list.cpp)-->