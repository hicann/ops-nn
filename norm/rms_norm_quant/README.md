# RmsNormQuant

## 产品支持情况

| 产品 | 是否支持 |
| :---------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                                                |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                        |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                                         |    √    |
| <term>Atlas 推理系列产品</term>                                                |    √    |
| <term>Atlas 训练系列产品</term>                                                 |    ×    |


## 功能说明

- 算子功能：RmsNorm算子是大模型常用的标准化操作，相比LayerNorm算子，其去掉了减去均值的部分。RmsNormQuant算子将RmsNorm算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。
- 计算公式：

  $$
  quant\_in_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i + b_i, \quad \text { where }   \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$
  
  $$
  y=round((quant\_in*scale)+offset)
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
      <td>表示标准化过程中的源数据张量，公式中的`x`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的缩放张量，公式中的`g`。shape支持1-2维。数据类型需要与`x`保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示标准化过程中的偏移张量，公式中的`b`。shape和数据类型需要与`gamma`保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>表示量化过程中得到y进行的scale张量，公式中的`scale`。shape为1，维度为1。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>表示量化过程中得到y进行的offset张量，公式中的`offset`。shape需要与`scale`保持一致。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，对应公式中的`eps`。</li><li>默认值为0.1f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>high_precision_mode</td>
      <td>可选属性</td>
      <td><ul><li>是否使用高精度模式。预留参数，暂未使用。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gemma_mode</td>
      <td>可选属性</td>
      <td><ul><li>是否使用gemma模式。预留参数，暂未使用。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示最终量化输出Tensor，对应公式中的`y`。shape需要与输入`x`一致。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>：输入参数`x`、`gamma`、`beta`、`scale`的数据类型不支持BFLOAT16。

## 约束说明

- <term>Atlas 推理系列产品</term>：x、y的尾轴长度，以及gamma的尾轴长度必须大于等于32Bytes。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_rms_norm_quant](examples/test_aclnn_rms_norm_quant.cpp) | 通过[aclnnRmsNormQuant](docs/aclnnRmsNormQuant.md)接口方式调用RmsNormQuant算子。 |