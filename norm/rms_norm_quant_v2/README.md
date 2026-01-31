# RmsNormQuantV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：RmsNorm是大模型常用的标准化操作，相比LayerNorm，其去掉了减去均值的部分。RmsNormQuantV2算子将RmsNorm算子以及RmsNorm归一化的输出给到1个或2个Quantize算子融合起来，减少搬入搬出操作。RmsNormQuantV2算子相较于AddRmsNormQuant在RmsNorm计算过程中增加了可选偏置项bias参数，即计算公式中的`bias`。

- 计算公式：

  $$
  y_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i + beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - divMode为True时：

    $$
    y1=round((y/scales1)+zero\_points1)
    $$

    $$
    y2=round((y/scales2)+zero\_points2)
    $$
  - divMode为False时：

    $$
    y1=round((y*scales1)+zero\_points1)
    $$

    $$
    y2=round((y*scales2)+zero\_points2)
    $$

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
      <td>表示标准化过程中的源数据张量，对应公式中的`x`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量，对应公式中的`g`。shape与x1需要norm（层归一化）的维度保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>输入</td>
      <td>表示量化过程中得到y1的scales张量，对应公式中的`scales1`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y2的scales张量，对应公式中的`scales2`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y1的offset张量，对应公式中的`zero_points1`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT32、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y2的offset张量，对应公式中的`zero_points2`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT32、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选输入</td>
      <td>表示标准化过程中的偏置项，公式中的`beta`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>用于防止除0错误，对应公式中的`epsilon`。</li><li>默认值为1e-6。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>可选属性</td>
      <td><ul><li>公式中决定量化公式是否使用除法的参数，对应公式中的`divMode`。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y1`。</td>
      <td>INT8、INT4、FLOAT8、HIF8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y2`。</td>
      <td>INT8、INT4、FLOAT8、HIF8</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

<term>Ascend 950PR/Ascend 950DT</term>：
- 当`x`的数据类型为FLOAT32时，`scale`和`offset`的数据类型只能为FLOAT32；当`x`的数据类型为FLOAT16或者BFLOAT16，并且`scale`的数据类型为FLOAT32时，`offset`的数据类型只能是FLOAT32或者INT32，否则`scale`的数据类型需要与`x`的数据类型保持一致同为FLOAT16或者BFLOAT16，`offset`的数据类型也需要与`x`的数据类型保持一致或者为INT8。
- 当`y`的数据类型为INT4时，`x`的最后一维、`gamma`以及`beta`的维度必须为偶数。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_rms_norm_quant](examples/test_aclnn_rms_norm_quant.cpp) | 通过[aclnnRmsNormQuant](docs/aclnnRmsNormQuant.md)接口方式调用RmsNormQuantV2算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/rms_norm_quant_v2_proto.h)构图方式调用RmsNormQuantV2算子。         |