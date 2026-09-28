# ForeachBinaryOp

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    ×    |
| <term>Atlas A2系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×    |
| <term>Atlas推理系列产品</term>                             |    ×    |
| <term>Atlas训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：对两个Tensor列表`x1`、`x2`逐Tensor、逐元素做二元运算，运算类型由属性`op_code`选择。该算子将多种二元foreach运算统一为一个算子，便于图融合。

- 计算公式：

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}],\ x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}],\ y = [{y_0}, {y_1}, ... {y_{n-1}}]
  $$

  $$
  y_i = x1_i \odot x2_i \quad (i=0,1,...,n-1)
  $$

  其中 $\odot$ 由`op_code`决定：

  | op_code | 运算 | 公式 |
  |:---:|:---:|:---|
  | 0 | add | $y_i = x1_i + x2_i$ |
  | 1 | sub | $y_i = x1_i - x2_i$ |
  | 2 | mul | $y_i = x1_i \times x2_i$ |
  | 3 | div | $y_i = x1_i / x2_i$ |

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
      <td>表示二元运算的第一个输入张量列表，对应公式中的`x1`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT16、FLOAT、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示二元运算的第二个输入张量列表，对应公式中的`x2`。数据类型、数据格式和shape与入参`x1`一致，该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT16、FLOAT、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>op_code</td>
      <td>属性</td>
      <td>表示二元运算的类型，对应公式中的$\odot$。取值范围为[0, 4)：0表示add、1表示sub、2表示mul、3表示div。取值为3（div）时，INT32的除数为0的元素结果置0，浮点的除数为0时遵循IEEE语义产生inf或nan。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示二元运算的输出张量列表，对应公式中的`y`。数据类型、数据格式和shape与入参`x1`一致，该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT16、FLOAT、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x1`、`x2`、`y`三个列表的Tensor个数一致，且一一对应的Tensor shape一致。
- 列表的Tensor个数上限为256。
- 支持空Tensor：总元素数为0时不做计算。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_foreach_binary_op](examples/arch35/test_geir_foreach_binary_op.cpp) | 通过[算子IR](op_graph/foreach_binary_op_proto.h)构图方式调用ForeachBinaryOp算子。 |
