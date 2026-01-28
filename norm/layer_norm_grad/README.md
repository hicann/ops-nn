# LayerNormGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 算子功能：[LayerNorm](../layer_norm/README.md)的反向传播。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。 
- 计算公式：
  
  $$
  rstd = 1/(variance + epsilon)
  $$

  $$
  res\_for\_gamma = (input - mean) \times rstd
  $$
  
  $$
  dy\_g = gradOut \times weight
  $$
  
  $$
  temp_1 = 1/N \times \sum_{reduce\_axis\_1} gradOut \times weight
  $$
  
  $$
  temp_2 = 1/N \times (input - mean) \times rstd \times \sum_{reduce\_axis\_1}(gradOut \times weight \times (input - mean) \times rstd)
  $$
 
  $$
  gradInputOut = (gradOut \times weight - (temp_1 + temp_2)) \times rstd
  $$
  
  $$
  gradWeightOut =  \sum_{reduce\_axis\_0}gradOut \times (input - mean) \times rstd
  $$
  
  $$
  gradBiasOut = \sum_{reduce\_axis\_0}gradOut
  $$

  其中，N为进行归一化计算的轴的维度，即归一化轴维度的大小。
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
      <td>反向计算的梯度张量，对应计算公式中的`gradOut`。与输入x的数据类型相同。shape与x的shape相等，为[A1,...,Ai,R1,...,Rj]。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>正向计算的首个输入，对应计算公式中的`input`。与输入`dy`的数据类型相同。shape与`dy`的shape相等，为[A1,...,Ai,R1,...,Rj]。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>输入</td>
      <td>正向计算的第三个输出，表示`x`的方差，对应计算公式中的`variance`。与输入mean的数据类型相同且位宽不低于输入`x`的数据类型位宽。shape与`mean`的shape相等，为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>正向计算的第二个输出，表示`x`的均值，对应计算公式中的`mean`。与输入`variance`的数据类型相同且位宽不低于输入`x`的数据类型位宽。shape与`variance`的shape相等，为[A1,...,Ai,1,...,1]，Ai后共有j个1，与需要norm的轴长度保持相同。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示权重张量，对应公式中的`weight`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_mask</td>
      <td>可选属性</td>
      <td><ul><li>表示输出的掩码，长度固定为3，取值为true时表示对应位置的输出非空。</li><li>默认值为{true, true, true}。</li></ul></td>
      <td>LISTBOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pd_x</td>
      <td>输出</td>
      <td>表示反向传播的输出梯度，由`output_mask`的第0个元素控制是否输出，对应计算公式中的`gradInputOut`。`output_mask`第0个元素为true时会进行输出，与输入x的数据类型相同，shape与`x`的shape相等，</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_gamma</td>
      <td>输出</td>
      <td>表示反向传播权重的梯度，由`output_mask`的第1个元素控制是否输出，对应计算公式中的`gradWeightOut`。`output_mask`第1个元素为true时会进行输出，与输入`gamma`的数据类型相同，shape与`pd_beta`的shape相等。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_beta</td>
      <td>输出</td>
      <td>表示反向传播偏置的梯度，由`output_mask`的第2个元素控制是否输出，对应计算公式中的`gradBiasOut`。`output_mask`第2个元素为true时会进行输出，与输入`gamma`的数据类型相同。shape与`pd_gamma`的shape相等。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | -  | 通过[算子IR](op_graph/layer_norm_grad_proto.h)构图方式调用LayerNormGrad算子。         |
