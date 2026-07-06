# ApplyAddSign

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- **算子功能**：执行AddSign优化器的单步参数更新。根据当前梯度、动量、学习率和符号衰减系数，计算参数更新量并**原地**更新权重参数（inplace 语义）。对标TensorFlow中`tf.raw_ops.ApplyAddSign` 接口的计算语义。

- **计算公式**：

  $$
  \begin{aligned}
  m_{t}   &= \beta \cdot m_{t-1} + (1 - \beta) \cdot grad \\
  update  &= (\alpha + sign\_decay \cdot \text{sign}(grad) \cdot \text{sign}(m_{t})) \cdot grad \\
  var_{t} &= var_{t-1} - lr \cdot update
  \end{aligned}
  $$

  其中`beta`为动量衰减系数（取值范围 [0, 1]），`alpha`为更新缩放系数，`sign_decay`为符号衰减系数，`lr`为学习率，`sign(\cdot)`为符号函数（NaN保留NaN）。

## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>var</td>
    <td>输入 / 输出 (inplace)</td>
    <td>待更新的权重参数，对应公式中的var。Kernel内inplace 更新，GE IR单输出视图与输入var共享Device内存。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>m</td>
    <td>输入 (inplace 更新)</td>
    <td>梯度一阶矩动量，对应公式中的m。shape/dtype 必须与var一致；Kernel内显式写回输入GM地址。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，对应公式中的lr。shape={1} 的 1 元素scalar Tensor，dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>输入</td>
    <td>更新缩放系数，对应公式中的alpha。shape={1} 的1元素 scalar Tensor。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sign_decay</td>
    <td>输入</td>
    <td>符号衰减系数，对应公式中的sign_decay。shape={1} 的 1 元素scalar Tensor。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>动量衰减系数，对应公式中的beta。shape={1} 的 1 元素scalar Tensor，取值范围 [0, 1]。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前梯度Tensor，对应公式中的grad。shape/dtype 必须与var一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>是否在更新时加锁。默认false。当前实现不强制互斥锁，仅作语义占位。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>输出</td>
    <td>更新后的var Tensor，与输入var共享Device内存（inplace）。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_apply_add_sign](./examples/test_geir_apply_add_sign.cpp) | 通过 [算子IR](./op_graph/apply_add_sign_proto.h) 构图方式调用 ApplyAddSign 算子。 |
