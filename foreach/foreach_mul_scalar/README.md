# ForeachMulScalar

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |


## 功能说明

- 算子功能：对输入张量列表的每个张量与张量scalar执行相乘运算。

- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x_i * scalar (i=0,1,...n-1)
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
      <td>表示乘法运算的输入张量列表，对应公式中的`x`。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>输入</td>
      <td>表示乘法运算的输入张量，对应公式中的`scalar`。元素个数为1。</td>
      <td>FLOAT32、FLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示x乘以scalar的输出张量列表，对应公式中的`y`。数据类型和数据格式与入参`x`的数据类型和数据格式一致，shape size大于等于入参`x`的shape size。该参数中所有Tensor的数据类型保持一致。</td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 当`x`的数据类型为FLOAT32、FLOAT16、INT32时，`scalar`数据类型与`x`的数据类型保持一致。
  - 当`x`的数据类型为BFLOAT16时，`scalar`数据类型支持FLOAT32。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - 当`x`的数据类型为FLOAT32、INT32时，`scalar`数据类型与`x`的数据类型保持一致。
  - 当`x`的数据类型为BFLOAT16时，`scalar`数据类型支持FLOAT32。
  - 当`x`的数据类型为FLOAT16时，`scalar`数据类型支持FLOAT16、FLOAT32。
  - `x`支持包含的最大Tensor个数为50。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_mul_scalar](examples/test_aclnn_foreach_mul_scalar.cpp) | 通过[aclnnForeachMulScalar](docs/aclnnForeachMulScalar.md)接口方式调用ForeachMulScalar算子。 |
| aclnn接口  | [test_aclnn_foreach_mul_scalar_v2](examples/test_aclnn_foreach_mul_scalar_v2.cpp) | 通过[aclnnForeachMulScalarV2](docs/aclnnForeachMulScalarV2.md)接口方式调用ForeachMulScalar算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_mul_scalar_proto.h)构图方式调用ForeachMulScalar算子。         |