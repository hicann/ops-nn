# SoftplusGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

接口功能：Softplus激活函数的反向传播，对齐TensorFlow `raw_ops.SoftplusGrad`。

计算公式：

$$
backprops = gradients \times sigmoid(features) = \frac{gradients}{1 + exp(-features)}
$$

支持 NumPy 风格广播：gradients 和 features 可以有不同的 shape，输出 shape 为广播后的结果。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
  <col style="width: 120px">
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
      <td>gradients</td>
      <td>输入</td>
      <td>表示梯度。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>features</td>
      <td>输入</td>
      <td>表示输入张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>backprops</td>
      <td>输出</td>
      <td>表示反向传播结果。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>：数据类型支持FLOAT、FLOAT16。

## 约束说明

- gradients、features、backprops的数据类型必须相同。
- 本交付数据类型仅支持FLOAT、FLOAT16、BFLOAT16，不支持DOUBLE（DOUBLE输入会在算子编译Tiling阶段报错；IR类型声明为对齐CANN内置注册的跨后端全集，含DOUBLE）。
- gradients和features支持NumPy风格广播。
- 支持0-8维输入。
- 支持动态shape：输入shape可含-1未知维度，推导时与另一输入的具体维度合并，双输入均为-1时输出保持-1。
- 支持未知秩[-2]：任一输入为未知秩时，输出shape保守推导为[-2]。
- sigmoid按输入数据类型使用TensorFlow公式`1 / (1 + exp(-features))`计算，极值输入遵循该数据类型的浮点溢出及NaN、Inf传播行为。

## 调用说明

| 调用方式   | 样例代码 | 说明 |
| ---------------- | -------- | ---- |
| GE图模式  | [test_geir_softplus_grad.cpp](examples/arch35/test_geir_softplus_grad.cpp) | 通过GE IR构建图方式调用SoftplusGrad算子，算子IR定义见[softplus_grad_proto.h](op_graph/softplus_grad_proto.h)。 |
