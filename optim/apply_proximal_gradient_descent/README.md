# ApplyProximalGradientDescent

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：完成带L1/L2正则的近端梯度下降（Proximal Gradient Descent）一步更新。给定待更新权重var、学习率alpha、L1正则系数l1、L2正则系数l2和梯度delta，按元素计算输出。
- 对标接口：TensorFlow `tf.raw_ops.ResourceApplyProximalGradientDescent`。
- 计算公式:

  $$
  prox\_v = var - alpha \times delta
  $$

  $$
  varOut = \dfrac{\operatorname{sign}(prox\_v)}{1 + alpha \times l2} \times \max\bigl(|prox\_v| - alpha \times l1,\ 0\bigr)
  $$

  其中`sign(0) = 0`，与TensorFlow语义一致。

  退化情形：

  | 条件            | 等价公式                                       |
  | --------------- | ---------------------------------------------- |
  | `l1 = 0, l2 = 0` | `varOut = var - alpha * delta`（标准SGD）      |
  | `l1 = 0`         | `varOut = (var - alpha * delta) / (1 + alpha * l2)` |

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>var</td>
      <td>输入</td>
      <td>待更新权重张量，支持1-8维tensor，支持非连续tensor。shape需要与delta的shape完全相同。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NC1HWC0、C1HWNCoC0、ND、FRACTAL_Z</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>学习率标量张量（0-D或shape=[1]，非负）。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l1</td>
      <td>输入</td>
      <td>L1正则系数标量张量（0-D或shape=[1]，非负）。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2</td>
      <td>输入</td>
      <td>L2正则系数标量张量（0-D或shape=[1]，非负）。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>delta</td>
      <td>输入</td>
      <td>梯度张量，shape与var完全相同，支持非连续tensor。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NC1HWC0、C1HWNCoC0、ND、FRACTAL_Z</td>
    </tr>
    <tr>
      <td>var（输出）</td>
      <td>输出</td>
      <td>输出张量，shape/dtype/format与输入var一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NC1HWC0、C1HWNCoC0、ND、FRACTAL_Z</td>
    </tr>
  </tbody></table>

## 约束说明

- 所有输入的数据类型必须一致，仅支持FLOAT16、FLOAT32。
- var和delta的shape必须完全相同，不支持广播。
- alpha / l1 / l2必须为0-D或shape=[1] 的标量张量。
- FP16输入在内部会提升至FP32精度计算，结果再转换回原始精度。
- 默认确定性实现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式  | [test_geir_apply_proximal_gradient_descent.cpp](examples/test_geir_apply_proximal_gradient_descent.cpp) | 通过图模式方式调用ApplyProximalGradientDescent算子。 |
