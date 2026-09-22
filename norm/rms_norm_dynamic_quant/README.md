# RmsNormDynamicQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     ×    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term>       |     ×    |
|  <term>Kirin 9030处理器系列产品</term> | × |

## 功能说明

- 算子功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicQuant算子则是为输入张量进行对称动态量化的算子。RmsNormDynamicQuant算子将RmsNorm归一化和DynamicQuant动态量化融合起来，减少搬入搬出操作。

- 计算公式：

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma + beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - 若smooth_scales不输入，则直接对rmsnorm输出做量化：

  $$
   scaleOut=row\_max(abs(y))/max\_val
  $$

  $$
   yOut=round(y/scaleOut)
  $$

  - 若输入smooth_scales，则先做smooth缩放再量化：

  $$
    input = y\cdot smooth\_scales
  $$

  $$
   scaleOut=row\_max(abs(input))/max\_val
  $$

  $$
   yOut=round(input/scaleOut)
  $$

  其中row\_max代表每行求最大值。max\_val在INT8时为127。

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
      <td>表示标准化过程中的源数据张量。公式中的x。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量，公式中的gamma。shape需要与x最后一维一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>可选输入</td>
      <td>表示量化过程中使用的smoothScale张量，公式中的smooth_scales。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选输入</td>
      <td>表示标准化过程中的偏置项，公式中的beta。shape和dtype需要与gamma一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>用于防止除0错误，公式中的epsilon，必须大于零。</li><li>默认值为1e-6。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>表示输出y的数据类型枚举值。</li><li>默认值为DT_INT8。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示量化输出Tensor，公式中的yOut。shape与x一致。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输出</td>
      <td>量化的scale输出，公式中的scaleOut。shape为x去掉最后一维后的shape。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入shape限制：输入`x`的最后一维必须小于等于8192，否则可能出现精度问题。

- 输入值域限制：`x`不支持全0输入。

## 调用说明

| 调用方式   | 调用样例           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_rms_norm_dynamic_quant](examples/test_aclnn_rms_norm_dynamic_quant.cpp) | 通过[aclnnRmsNormDynamicQuant](docs/aclnnRmsNormDynamicQuant.md)接口方式调用RmsNormDynamicQuant算子 |
| GE图模式 | [test_geir_rms_norm_dynamic_quant](examples/test_geir_rms_norm_dynamic_quant.cpp) | 通过[算子IR](op_graph/rms_norm_dynamic_quant_proto.h)构图方式调用RmsNormDynamicQuant算子 |
| PyTorch API | - | 通过[cann_ops_nn.rms_norm_dynamic_quant](docs/torchapi_rms_norm_dynamic_quant.md)接口调用RmsNormDynamicQuant算子 |
