# AddRmsNormCast

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：RmsNorm算子是大模型常用的归一化操作，AddRmsNormCast算子将AddRmsNorm后的Cast算子融合起来，减少搬入搬出操作。
- 计算公式：

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  y_2=\operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  $$
  y_1=float(y_2)
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
      <td>x1</td>
      <td>输入</td>
      <td>需要归一化的原始数据输入，公式中的输入`x1`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>需要归一化的原始数据输入，公式中的输入`x2`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>可选输入</td>
      <td>数据缩放因子，公式中的输入`g`。shape需要与`x1`后几维保持一致，后几维为x1需要norm的维度。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，用于防止除0错误，对应公式中的eps。</li><li>默认值为1e-6f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
      <td>y1</td>
      <td>输出</td>
      <td>归一化后经过类型转换的输出数据，公式中的输出`y1`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    </tr>
      <td>y2</td>
      <td>输出</td>
      <td>归一化后的输出数据，公式中的输出`y2`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>
      <td>x的标准差，公式中的输出`Rms(x)`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>归一化的数据和，公式中的输出`x`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_add_rms_norm_cast](examples/test_aclnn_add_rms_norm_cast.cpp) | 通过[aclnnAddRmsNormCast](docs/aclnnAddRmsNormCast.md)接口方式调用AddRmsNormCast算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/add_rms_norm_cast_proto.h)构图方式调用AddRmsNormCast算子。         |