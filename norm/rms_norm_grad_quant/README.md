# RmsNormGradQuant

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

- 算子功能：RmsNormGrad是用于计算RmsNorm的梯度，即在反向传播过程中计算输入张量的梯度的算子。RmsNormGradQuant算子将RmsNormGrad和Quantize两个算子融合，RmsNormGrad计算完dx后进行quant计算，减少搬入搬出操作。

- 计算公式：

  $$
  dx_i= (dy_i * g_i - \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * \operatorname{Mean}(\mathbf{y})) * \frac{1} {\operatorname{Rms}(\mathbf{x})},  \quad \text { where } \operatorname{Mean}(\mathbf{y}) = \frac{1}{n}\sum_{i=1}^n (dy_i * g_i * x_i * \frac{1}{\operatorname{Rms}(\mathbf{x})})
  $$

  - div\_mode为True时：

    $$
    dx_i\_quant=round((dx_i / scales\_x) + offset\_x)
    $$

  - div\_mode为False时：

    $$
    dx_i\_quant=round((dx_i * scales\_x) + offset\_x)
    $$

  $$
  dg_i = \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * dy_i
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
      <td>dy</td>
      <td>输入</td>
      <td><ul><li>表示反向传回的梯度，对应公式中的<code>dy</code>。</li><li>shape支持2-8维。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示正向算子的输入，被标准化的数据，对应公式中的<code>x</code>。</li><li>shape和dtype与dy保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td><ul><li>表示正向算子的中间计算结果，对应公式中<code>Rms(x)</code>的倒数。</li><li>shape需要满足rstd_shape = x_shape[0:n]，n &lt; x_shape.dims()，n与gamma的n一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td><ul><li>表示正向算子进行归一化计算的缩放因子（权重），对应公式中的<code>g</code>。</li><li>shape需要满足gamma_shape = x_shape[n:]，n &lt; x_shape.dims()。</li><li>dtype与dy相同或为FLOAT32。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales_x</td>
      <td>输入</td>
      <td><ul><li>表示输入梯度量化缩放因子，对应公式中的<code>scales_x</code>。</li><li>shape为[1]，维度为1。</li><li>dtype与dy相同或为FLOAT32。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset_x</td>
      <td>可选输入</td>
      <td><ul><li>表示输入梯度量化零点，对应公式中的<code>offset_x</code>。</li><li>shape为[1]，维度为1。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td><ul><li>量化模式。</li><li>仅支持"static"，表示静态量化模式。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>属性</td>
      <td><ul><li>公式中决定量化公式是否使用除法的参数，对应公式中的<code>div_mode</code>。</li><li>支持True和False。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>表示指定数据转换后dx的类型。</li><li>输入范围为{2, 34}，分别对应{INT8, HIFLOAT8}。</li><li>默认值为2。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td><ul><li>表示输入<code>x</code>的量化梯度，对应公式中的<code>dx_quant</code>。</li><li>shape与入参<code>dy</code>的shape保持一致。</li></ul></td>
      <td>INT8、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>输出</td>
      <td><ul><li>表示<code>gamma</code>的梯度，对应公式中的<code>dg</code>。</li><li>shape与入参<code>gamma</code>的shape保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 各输入Tensor支持空Tensor。
  - dy、x、rstd、gamma支持非连续Tensor；scales\_x、offset\_x不支持非连续Tensor。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_rms_norm_grad_quant](examples/arch35/test_aclnn_rms_norm_grad_quant.cpp) | 通过[aclnnRmsNormGradQuant](docs/aclnnRmsNormGradQuant.md)接口方式调用RmsNormGradQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/rms_norm_grad_quant_proto.h)构图方式调用RmsNormGradQuant算子。         |
