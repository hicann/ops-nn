# AdaLayerNormGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：[AdaLayerNormV2](../ada_layer_norm_v2/README.md)的反向传播。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。 
- 计算公式：
  
  正向：

  $$
  out = LayerNorm(x)\times(1+scale)+shift
  $$

  反向：

  $$
  z = (x - mean) \times rstd
  $$
  
  $$
  dy\_g = dy \times gamma \times (1 + scale)
  $$
  
  $$
  temp_1 = 1/N \times \sum_{reduce\_axis\_2} (dy \times gamma \times (1 + scale))
  $$
  
  $$
  temp_2 = 1/N \times (x - mean) \times rstd \times \sum_{reduce\_axis\_2}(dy \times gamma \times (1 + scale) \times (x - mean) \times rstd)
  $$

  $$
  pd\_x = (dy \times gamma \times (1 + scale) - (temp_1 + temp_2)) \times rstd
  $$
  
  $$
  pd\_scale =  \sum_{reduce\_axis\_1}(dy \times ((x - mean) \times rstd \times gamma + beta))
  $$

  $$
  pd\_shift =  \sum_{reduce\_axis\_1}dy
  $$

  $$
  pd\_gamma =  \sum_{reduce\_axis\_0,1}dy \times (1 + scale) \times (x - mean) \times rstd
  $$
  
  $$
  pd\_beta = \sum_{reduce\_axis\_0,1}dy \times (1 + scale)
  $$

  其中，N为进行归一化计算的轴的维度，即归一化轴维度的大小；'reduce_axis_0,1'表示第0维和第1维度进行归约求和。

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
      <td>dy</td>
      <td>输入</td>
      <td>反向计算的梯度张量，对应计算公式中的`dy`。与输入`x`的数据类型相同。shape与`x`的shape相等，为[B, S, H]，其中B支持0-6维。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>正向计算的首个输入，对应计算公式中的`x`。与输入`dy`的数据类型相同。shape与`dy`的shape相等，为[B, S, H]，其中B支持0-6维。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td>表示`x`的标准差的倒数，对应计算公式中的`rstd`。与输入`dy`的数据类型相同。shape与`mean`的shape相等，为[B, H, 1]，最后一维固定为1，其他维度与`x`一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示`x`的均值，对应计算公式中的`mean`。与输入`dy`的数据类型相同。shape与`rstd`的shape相等，为[B, H, 1]。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>scale</td>
      <td>输入</td>
      <td>表示自适应缩放张量，对应公式中的`scale`。数据类型与`dy`的数据类型一致。shape为[B, H]或[B, 1, H]，其中B支持0-6维，维度数量和大小与`dy`中的B保持一致，H与`dy`中H维一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示权重张量，对应公式中的`gamma`。数据类型支持FLOAT32、FLOAT16、BFLOAT16，当数据类型不为FLOAT32时，参数数据类型与`dy`相同。shape为[H]，H与`dy`中H维一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
        </tr>
        <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示偏置张量，对应公式中的`beta`。数据类型与`gamma`的数据类型一致。shape为[H]，H与`dy`中H维一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_x</td>
      <td>输出</td>
      <td>表示反向传播的输出梯度，对应计算公式中的`pd_x`。与输入`x`的数据类型相同，shape与`x`的shape相等。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_scale</td>
      <td>输出</td>
      <td>表示反向传播自适应缩放系数的梯度，对应计算公式中的`pd_scale`。与输入`scale`的数据类型相同，shape与`scale`的shape相等。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_shift</td>
      <td>输出</td>
      <td>表示反向传播自适应偏移系数的梯度，对应计算公式中的`pd_shift`。与输入`scale`的数据类型相同，shape与为[B, H]或[B, 1, H]，其中B支持0-6维，维度数量和大小与`dy`中的B保持一致，H与`dy`中H维一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_gamma</td>
      <td>输出</td>
      <td>表示反向传播权重的梯度，对应计算公式中的`pd_gamma`。与输入`gamma`的数据类型相同，shape与`gamma`的shape相等。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_beta</td>
      <td>输出</td>
      <td>表示反向传播偏置的梯度，对应计算公式中的`pd_beta`。与输入`gamma`的数据类型相同。shape与`pd_gamma`的shape相等。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_ada_layer_norm_grad](examples/test_aclnn_ada_layer_norm_grad.cpp) | 通过[aclnnAdaLayerNormBackward](docs/aclnnAdaLayerNormBackward.md)接口方式调用AdaLayerNormGrad算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/ada_layer_norm_grad_proto.h)构图方式调用AdaLayerNormGrad算子。         |
