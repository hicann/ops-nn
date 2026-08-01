# AddRmsNormDynamicQuantV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicQuant算子则是为输入张量进行对称动态量化的算子。AddRmsNormDynamicQuantV2算子将RmsNorm前的Add算子和RmsNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。
- 计算公式：

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  yFP32=\begin{cases}
  cast(y) & outputMask[2]=True\ ||\ outputMask\ = null \\
  无效输出 & outputMask[2]=False
  \end{cases}
  $$

  $$
  y\_input=y+beta
  $$

  $$
  input1 =\begin{cases}
    y\_input \cdot smoothScale1Optional & \ \ smoothScale1Optional\ != null \\
    y\_input & \ \  smoothScale1Optional\ = null
    \end{cases}
  $$

  $$
  input2 =\begin{cases}
    y\_input \cdot smoothScale2Optional & \ \ smoothScale2Optional\ != null  \\
    y\_input & \ \ smoothScale2Optional\ = null
    \end{cases}
  $$

  $$
  scale1Out=\begin{cases}
    row\_max(abs(input1))/127 & outputMask[0]=True\ ||\ outputMask\ = null \\
    无效输出 & outputMask[0]=False
    \end{cases}
  $$

  $$
  y1Out=\begin{cases}
    round(input1/scale1Out) & outputMask[0]=True\ ||\ outputMask\ = null \\
    无效输出 & outputMask[0]=False
    \end{cases}
  $$

  $$
  scale2Out=\begin{cases}
    row\_max(abs(input2))/127 & outputMask[1]=True\ ||\ (outputMask\ = null\ \&\ smoothScale1Optional\ != null\ \&\ smoothScale2Optional\ != null) \\
    无效输出 & outputMask[1]=False\ ||\ (outputMask\ = null\ \&\ (smoothScale1Optional\ = null\ ||\ smoothScale2Optional\ = null))
    \end{cases}
  $$

  $$
  y2Out=\begin{cases}
    round(input2/scale2Out) & outputMask[1]=True\ ||\ (outputMask\ = null\ \&\ smoothScale1Optional\ != null\ \&\ smoothScale2Optional\ != null)\\
    无效输出 & outputMask[1]=False\ ||\ (outputMask\ = null\ \&\ (smoothScale1Optional\ = null\ ||\ smoothScale2Optional\ = null))
    \end{cases}
  $$

  其中row\_max代表每行求最大值，当outputMask[3]=False时，不输出y。

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
      <td>x1</td>
      <td>输入</td>
      <td><ul><li>支持空Tensor。</li><li>表示标准化过程中的源数据张量，对应公式中的`x1`。</li><li>当输出`y1`或`y2`的类型为INT4时，`x1`的尾轴必须能被2整除。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td><ul><li>支持空Tensor。</li><li>表示标准化过程中的源数据张量，对应公式中的`x2`，shape和数据类型与`x1`一致</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td><ul><li>支持空Tensor。</li><li>表示标准化过程中的权重张量，对应公式中的`gamma`，数据类型与`x1`一致，shape需要与`x1`最后一维一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale1</td>
      <td>可选输入</td>
      <td><ul><li>支持空Tensor。</li><li>表示量化过程中得到y1使用的smoothScale张量，对应公式中的`smoothScale1Optional`。</li><li>shape和数据类型需要与`gamma`保持一致。</li><li>具体约束详见约束说明。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale2</td>
      <td>可选输入</td>
      <td><ul><li>支持空Tensor。</li><li>表示量化过程中得到y2使用的smoothScale张量，对应公式中的`smoothScale2Optional`。</li><li>shape和数据类型需要与`gamma`保持一致。</li><li>具体约束详见约束说明。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选输入</td>
      <td><ul><li>支持空Tensor。</li><li>表示标准化过程中的偏置项，对应公式中的`beta`。</li><li>shape和数据类型需要与`gamma`保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>用于防止除0错误，对应公式中的`epsilon`。</li><li>默认值为1e-6。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_mask</td>
      <td>可选属性</td>
      <td><ul><li>表示输出的掩码，对应公式中的`outputMask`。只支持长度为0，或者长度为4的数组。</li><li>具体约束详见约束说明。</li><li>默认值为{}。</li></ul></td>
      <td>LISTBOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>指定`y1`和`y2`的输出数据类型。</li><li>输入范围为{2, 29, 34, 35, 36}，分别对应{INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN}。</li><li>默认值为2。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td><ul><li>支持空Tensor。</li><li>表示量化输出Tensor，对应公式中的`y1Out`。</li><li>如果`y1`为有效输出时，shape和数据类型需要与输入`x1`保持一致。</li><li>具体约束详见约束说明。</li></ul></td>
      <td>INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td><ul><li>支持空Tensor。</li><li>表示量化输出Tensor，对应公式中的`y2Out`。</li><li>如果`y2`为有效输出时，shape和数据类型需要与输入`x1`保持一致。</li><li>具体约束详见约束说明。</li></ul></td>
      <td>INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y3</td>
      <td>输出</td>
      <td><ul><li>支持空Tensor。</li><li>表示rmsNorm的FLOAT32类型输出Tensor，对应公式中的`yFP32`。</li><li>如果`y3`为有效输出时，shape需要与输入`x1`保持一致。</li><li>具体约束详见约束说明。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y4</td>
      <td>输出</td>
      <td><ul><li>支持空Tensor。</li><li>表示rmsNorm的原始输入类型输出Tensor，对应公式中的`y`。</li><li>如果`y4`为有效输出时，shape和数据类型需要与输入`x1`保持一致。</li><li>具体约束详见约束说明。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td><ul><li>支持空Tensor。</li><li>表示x1和x2的和，对应公式中的`x`。</li><li>shape和数据类型需要与输入`x1`保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale1</td>
      <td>输出</td>
      <td><ul><li>支持空Tensor。</li><li>第一路量化的输出，对应公式中的`scale1Out`。</li><li>如果此输出为有效输出，shape需要与`x1`除了最后一维后的shape一致。</li><li>具体约束详见约束说明。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale2</td>
      <td>输出</td>
      <td><ul><li>支持空Tensor。</li><li>第二路量化的输出，对应公式中的`scale2Out`。</li><li>如果此输出为有效输出，shape需要与`x1`除了最后一维后的shape一致。</li><li>具体约束详见约束说明。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas A2 训练系列产品/Atlas A2 推理系列产品/Kirin X90/Kirin 9030处理器系列产品：
  - x1、x2、gamma、smooth_scale1、smooth_scale2、y4和x的数据类型不支持BFLOAT16。
  - y1和y2的数据类型仅支持INT8。
  - beta、output_mask和dst_type的配置无效。
  - y1和y2输出情况仅与smooth_scale1和smooth_scale2的输入情况有关，且仅y2可不输出。

## 约束说明

- 当output_mask不为空时：
  - 参数smooth_scale1有值时，则output_mask[0]必须为True。参数smooth_scale2有值时，则output_mask[1]必须为True。
  - output_mask[0]和output_mask[1]不能同时为false。
  - 各输出有效性由output_mask统一控制，当对于output_mask位置为True时，y和scale为有效输出，对应为False时，y和scale为无效输出。

- 当output_mask为空时：
  - 参数smooth_scale2有值时，参数smooth_scale1不能为空。
  - y1、y3、y4和scale1始终为有效输出。
  - y2和scale2只有在smooth_scal1和smooth_scale2均有效时为有效输出，否则为无效输出。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_add_rms_norm_dynamic_quant_v2](examples/test_geir_add_rms_norm_dynamic_quant_v2.cpp)  | 通过[算子IR](op_graph/add_rms_norm_dynamic_quant_v2_proto.h)构图方式调用AddRmsNormDynamicQuantV2算子。         |
