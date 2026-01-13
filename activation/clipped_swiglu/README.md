# ClippedSwiglu

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明
- 算子功能：带截断的Swish门控线性单元激活函数，实现x的SwiGlu计算。本算子相较于SwiGlu算子，新增了部分输入参数：groupIndex、alpha、limit、bias、interleaved，用于支持GPT-OSS模型使用的变体SwiGlu以及MoE模型使用的分组场景。

- 计算流程：

  对给定的输入张量 x ，其维度为[a,b,c,d,e,f,g…]，算子ClippedSwiglu对其进行以下计算：

  1. 将 x 基于输入参数 dim 进行合轴，合轴后维度为[pre,cut,after]。其中 cut 轴为合轴之后需要切分为两个张量的轴，切分方式分为前后切分或者奇偶切分；pre，after 可以等于1。例如当 dim 为3，合轴后 x 的维度为[a * b * c, d, e * f * g * …]。此外，由于after轴的元素为连续存放，且计算操作为逐元素的，因此将cut轴与after轴合并，得到x的维度为[pre,cut]。

  2. 根据输入参数 group_index, 对 x 的pre轴进行过滤处理，公式如下：

     $$
     sum = \text{Sum}(group\_index)
     $$

     $$
     x = x[ : sum, : ]
     $$

     其中sum表示group_index的所有元素之和。当不输入 group_index 时，跳过该步骤。

  3. 根据输入参数 interleaved，对 x 进行切分，公式如下：

     当 interleaved 为 true 时，表示奇偶切分：

     $$
     A = x[ : , : : 2]
     $$

     $$
     B = x[ : , 1 : : 2]
     $$

     当 interleaved 为 false 时，表示前后切分：

     $$
     h = x.shape[1] // 2
     $$

     $$
     A = x[ : , : h]
     $$

     $$
     B = x[ : , h : ]
     $$

  4. 根据输入参数 alpha、limit、bias 进行变体SwiGlu计算，公式如下：

     $$
     A = A.clamp(min=None, max=limit)
     $$

     $$
     B = B.clamp(min=-limit, max=limit)
     $$

     $$
     y\_glu = A * sigmoid(alpha * A)
     $$

     $$
     y = y\_glu * (B + bias)
     $$

  5. 重塑输出张量y的维度数量与合轴前的x的维度数量一致，dim轴上的大小为x的一半，其他维度与x相同。

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>公式中的输入x。维度必须大于0且必须在入参dim对应维度上是偶数。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td>公式中的输入group_index。维度必须是1维。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>可选属性</td>
      <td>公式中的输入dim，表示对x进行合轴以及切分的维度序号。取值范围为[-x.dim(), x.dim()-1]。默认为-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>可选属性</td>
      <td>公式中的输入alpha，表示变体SwiGlu使用的参数。默认为1.702。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>limit</td>
      <td>可选属性</td>
      <td>公式中的输入limit，表示变体SwiGlu使用的门限值。默认为7.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选属性</td>
      <td>公式中的输入bias，表示变体SwiGlu使用的偏差参数。默认为1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>interleaved</td>
      <td>可选属性</td>
      <td>公式中的输入interleaved，设置为true表示对x进行奇偶切分，设置为false表示对x进行前后切分。默认为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y。维度在入参dim对应维度上为x的一半，其他维度上与x一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_clipped_swiglu](./examples/test_aclnn_clipped_swiglu.cpp) | 通过[aclnnClippedSwiglu](./docs/aclnnClippedSwiglu.md)接口方式调用ClippedSwiglu算子。    |


