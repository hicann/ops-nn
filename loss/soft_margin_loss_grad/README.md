# SoftMarginLossGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：求SoftMarginLoss反向传播的梯度值。
- 计算公式：

  SoftMarginLoss的前向计算公式如下：

  $$
  loss_i = ln(1 + exp(-predict_i \cdot label_i))
  $$

  对 $predict$ 求偏导得到反向梯度：

  $$
  gradient = cof \cdot \frac{-label \cdot exp(-predict \cdot label)}{1 + exp(-predict \cdot label)} \cdot dout
  $$

  其中reduction为mean时 $cof = 1/N$，否则 $cof = 1.0$。

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>predict</td>
      <td>输入</td>
      <td>网络正向前一层的预测值。shape与label、dout相互broadcast。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>输入</td>
      <td>样本的标签值。shape与predict、dout相互broadcast。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dout</td>
      <td>输入</td>
      <td>网络反向传播前一步的梯度值。shape与predict、label相互broadcast。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradient</td>
      <td>输出</td>
      <td>存储梯度计算结果，数据类型与输入一致，shape为predict、label、dout三者broadcast后的shape。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>表示对反向求梯度计算结果做的reduce操作，取值为"none"/"mean"/"sum"，默认"mean"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 图模式支持0D~25D输入，aclnn接口支持0D~8D输入。

- 数据类型约束：通过GE图模式调用时，predict、label、dout与gradient的数据类型必须相同（同为FLOAT16、FLOAT32或BFLOAT16之一）；通过aclnn API调用时，输入数据类型需满足互转换关系，按共同数据类型规则转换后计算，输出gradient的数据类型与out一致。
- 支持shape broadcast：predict、label、dout三者的shape相互broadcast（低维在前补1，各维非1值需一致，否则不支持），输出shape为三者broadcast的结果。
- 图模式支持动态shape（-1）和未知rank（-2）。
- 空tensor：ACLNN接口支持空tensor输入；输入须满足broadcast规则，输出为空tensor且shape与broadcast结果一致。
- reduction为"mean"时，图模式的N取输入predict的元素数；aclnn接口的N取广播后的元素数。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_soft_margin_loss_grad.cpp](examples/arch35/test_aclnn_soft_margin_loss_grad.cpp) | 通过[aclnnSoftMarginLossBackward](docs/aclnnSoftMarginLossBackward.md)接口方式调用SoftMarginLossGrad算子。 |
| GE图模式  | [test_geir_soft_margin_loss_grad.cpp](examples/arch35/test_geir_soft_margin_loss_grad.cpp) | 编译与运行见[快速算子调用（GE图模式）](../../docs/zh/invocation/quick_op_invocation.md#ge图模式)。通过GE图模式调用算子，算子IR原型定义见[soft_margin_loss_grad_proto.h](op_graph/soft_margin_loss_grad_proto.h)。 |
