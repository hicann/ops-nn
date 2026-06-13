# EuclideanNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×   |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：

  沿给定的若干轴对输入张量计算欧几里得范数（L2范数），即先逐元素平方，再沿指定轴求和，最后开平方。等价于TensorFlow中的 `tf.math.reduce_euclidean_norm`。

- 计算公式：

  $$
  y = \sqrt{ \sum_{i \in axes} x_i^{2} }
  $$

  其中 $axes$ 为待reduce的维度集合；当属性 `keep_dims = true` 时，被reduce的维度在输出中保留为长度1，否则从输出shape中删除。

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
      <td>表示待求欧几里得范数的输入张量，公式中的x。支持任意维度，shape为 [D0, D1, ..., Dn-1]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>输入</td>
      <td><ul><li>表示需要reduce的维度，公式中的axes。1-D张量，元素取值范围为 [-rank(x), rank(x))，不允许重复。</li><li>该输入为值依赖输入：infershape与tiling都依赖其具体数值，图模式下需以Const算子注入。</li></ul></td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keep_dims</td>
      <td>可选属性</td>
      <td><ul><li>表示是否在输出中保留被reduce的维度。true时被reduce的维度保留为长度1，false时从输出shape中删除。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示欧几里得范数的计算结果，公式中的y。数据类型与x一致；shape由x.shape、axes与keep_dims共同决定。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- axes为值依赖输入，图模式调用时需以 `op::Const` 注入；运行时通过placeholder Data注入将导致编译期无法推导输出shape与tiling。
- axes元素必须落在[-rank(x), rank(x))范围内，且不允许重复。
- y的数据类型与x保持一致；当x为INT32时，输出仍为INT32（不做隐式类型提升），调用方需自行确保中间累加不溢出。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_euclidean_norm](./examples/arch35/test_geir_euclidean_norm.cpp)   | 通过[算子IR](./op_graph/euclidean_norm_proto.h)构图方式调用EuclideanNorm算子。 |
