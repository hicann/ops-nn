# ForeachNonFiniteCheckAndUnscale

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 算子功能：遍历scaledGrads中的所有Tensor，检查是否存在Inf或NaN，如果存在则将foundInf设置为1.0，否则foundInf保持不变，并对scaledGrads中的所有Tensor进行反缩放。

- 计算公式：
  $$
  foundInf = \begin{cases}1.0, & 当 Inf \in  scaledGrads 或 NaN \in scaledGrads,\\
    foundInf, &其他.
  \end{cases}
  $$

  $$
   scaledGrads_i = {scaledGrads}_{i}*{invScale}.
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
      <td>scaled_grads</td>
      <td>输入/输出</td>
      <td>表示进行反缩放计算的输入和输出张量列表，对应公式中的`scaledGrads`。支持的最大长度为256个。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>found_inf</td>
      <td>输入</td>
      <td>表示用来标记输入`scaled_grads`中是否存在Inf或-Inf的张量，对应公式中的`foundInf`。仅包含一个元素。如果输入`scaled_grads`中存在Inf或-Inf的张量，将`found_inf`设置为1；否则，不对`found_inf`进行操作，最后将`scaled_grads`中的所有值乘以`inv_scale`并存储在`scaled_grads`中。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inv_scale</td>
      <td>输入</td>
      <td>表示进行反缩放计算的张量，对应公式中的`invScale`。仅包含一个元素。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_foreach_non_finite_check_and_unscale](examples/test_aclnn_foreach_non_finite_check_and_unscale.cpp) | 通过[aclnnForeachNonFiniteCheckAndUnscale](docs/aclnnForeachNonFiniteCheckAndUnscale.md)接口方式调用ForeachNonFiniteCheckAndUnscale算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/foreach_non_finite_check_and_unscale_proto.h)构图方式调用ForeachNonFiniteCheckAndUnscale算子。         |