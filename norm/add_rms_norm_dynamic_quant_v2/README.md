# AddRmsNormDynamicQuantV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |


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
  yFP32=cast(y)
  $$

  - 若smoothScale1Optional和smoothScale2Optional均不输入，则y2Out和scale2Out输出无实际意义。计算过程如下所示：

  $$
   scale1Out=row\_max(abs(y))/127
  $$

  $$
   y1Out=round(y/scale1Out)
  $$

  - 若仅输入smoothScale1Optional，则y2Out和scale2Out输出无实际意义。计算过程如下所示：
  $$
    input = y\cdot smoothScale1Optional
  $$
  $$
   scale1Out=row\_max(abs(input))/127
  $$

  $$
   y1Out=round(input/scale1Out)
  $$

  - 若smoothScale1Optional和smoothScale2Optional均输入，则算子的五个输出均为有效输出。计算过程如下所示：
  $$
    input1 = y\cdot smoothScale1Optional
  $$
  $$
    input2 = y\cdot smoothScale2Optional
  $$
  $$
   scale1Out=row\_max(abs(input1))/127
  $$
  $$
   scale2Out=row\_max(abs(input2))/127
  $$
  $$
   y1Out=round(input1/scale1Out)
  $$
  $$
   y2Out=round(input2/scale2Out)
  $$

  其中row\_max代表每行求最大值。
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
      <td>表示标准化过程中的源数据张量，对应公式中的`x1`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量，对应公式中的`x2`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量，对应公式中的`gamma`。shape需要与`x1`最后一维一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale1</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y1使用的smoothScale张量，对应公式中的`smoothScale1Optional`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale2</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y2使用的smoothScale张量，对应公式中的`smoothScale2Optional`。</td>
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
      <td>y1</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y1Out`。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y2Out`。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y3</td>
      <td>输出</td>
      <td>表示rmsNorm的FLOAT32类型输出Tensor，对应公式中的`yFP32`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y4</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>表示x1和x2的和，对应公式中的`x`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale1</td>
      <td>输出</td>
      <td>第一路量化的输出，对应公式中的`scale1Out`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale2</td>
      <td>输出</td>
      <td>第二路量化的输出，对应公式中的`scale2Out`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_add_rms_norm_dynamic_quant_v2](examples/test_geir_add_rms_norm_dynamic_quant_v2.cpp)  | 通过[算子IR](op_graph/add_rms_norm_dynamic_quant_v2_proto.h)构图方式调用AddRmsNormDynamicQuantV2算子。         |

<!--[test_geir_add_rms_norm_dynamic_quant_v2](examples/test_geir_add_rms_norm_dynamic_quant_v2.cpp)-->