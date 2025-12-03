# GemmaRmsNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：GemmaRmsNorm算子是大模型常用的归一化操作，相比RmsNorm算子，在计算时对gamma做了+1操作。
- 计算公式：

  $$
  \operatorname{GemmaRmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} (1 + g_i), \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
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
      <td><ul><li>需要归一化的数据输入，对应公式中的`x`。</li><li>shape支持1-8维度。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td><ul><li>数据缩放因子，对应公式中的`g`。</li><li>数据类型与输入`x`的数据类型保持一致。</li><li>shape支持1-8维度，且满足gamma_shape = x_shape[n:], n < x_shape.dims()。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，用于防止除0错误，对应公式中的`eps`。</li><li>默认值为1e-6f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>归一化后的输出数据，对应公式中的`GemmaRmsNorm(x)`。</li><li>数据类型、shape与输入`x`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>      
      <td><ul><li>x的标准差倒数，对应公式中`Rms(x)`的倒数。</li><li>shape支持1-8维度，shape与入参`x`的shape前几维保持一致，前几维指x的维度减去`gamma`的维度，表示不需要norm的维度。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_gemma_rms_norm](examples/test_aclnn_gemma_rms_norm.cpp) | 通过[aclnnGemmaRmsNorm](docs/aclnnGemmaRmsNorm.md)接口方式调用GemmaRmsNorm算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/gemma_rms_norm_proto.h)构图方式调用GemmaRmsNorm算子。         |

<!--[test_geir_xxx](examples/test_geir_xxx.cpp)-->