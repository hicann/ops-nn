# L2NormalizeGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：前向L2Normalize（`y = x/max(sqrt(sum(x^2, dim)), eps)`）的反向算子。三输入`x`（前向输入）、`y`（前向输出，即归一化后的x）、`dy`（上游梯度），单输出`dx`（对x的梯度）。

- 计算公式（沿属性`dim`指定的归约轴集合归约，keepdims广播回全形状）：

  $$
  n = \max\left(\sqrt{\sum_{dim} x^2}, \ eps\right)
  $$

  $$
  s = \sum_{dim} (y \cdot dy)
  $$

  $$
  dx = \frac{dy - y \cdot s}{n}
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
      <td>表示正向算子的输入，对应公式中的<code>x</code>。支持1~8维。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>表示正向算子的输出，即归一化后的<code>x</code>，对应公式中的<code>y</code>。shape与数据类型与入参<code>x</code>一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>表示反向传回的梯度，对应公式中的<code>dy</code>。shape与数据类型与入参<code>x</code>一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>表示归一化轴集合，默认值为空数组<code>[]</code>。每个元素取值范围为[-<code>x</code>.dim(), <code>x</code>.dim()-1]。负值轴号自动转换为正索引（<code>dim + rank(x)</code>）。元素允许重复、乱序、正负混写，元素允许重复、乱序、正负混写，归约轴去重排序后须构成连续区间且逐维呈连续分布（如<code>[1]</code>、<code>[1,2]</code>），不支持轴号不相邻的轴集（如<code>[0,2]</code>）。数组长度不超过20。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>表示分母的下限，默认值为1e-4。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>表示对<code>x</code>的梯度，对应公式中的<code>dx</code>。shape与数据类型与入参<code>x</code>一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x`为空Tensor时，`dx`同为空Tensor。

- 入参`x`、`y`、`dy`三者的shape与数据类型必须完全一致，输出`dx`与之相同。

- 入参`x`支持1~8维；归约轴长度与张量元素数无额外上限（超出单次UB容量时由算子内部沿归约轴分块处理）。

- 属性`dim`的归约轴集合（折算负值并去重后）须构成连续区间，如`[1]`、`[1,2]`、`[0,1,2]`；不支持轴号不相邻的轴集，如`[0,2]`、`[1,3]`。

- 属性`dim`的数组长度不超过20；元素取值超出[-`x`.dim(), `x`.dim()-1]时报错。

- 属性`dim`不传或传空数组时不做归约，此时`dx`按逐元素公式计算：$dx = (dy - y \cdot y \cdot dy) / \max(|x|, eps)$。

## 调用说明

本算子为GE梯度图内部的反向算子，**不提供aclnn单算子接口**（无`aclnnL2NormalizeGrad`），仅通过GE图调用。



| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_l2_normalize_grad](examples/test_geir_l2_normalize_grad.cpp) | 通过[算子IR](op_graph/l2_normalize_grad_proto.h)构图方式调用L2NormalizeGrad算子。 |
