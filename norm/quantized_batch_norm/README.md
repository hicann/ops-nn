# QuantizedBatchNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |


## 功能说明

- 算子功能：将输入Tensor执行一个反量化的计算，再根据输入的weight、bias、epsilon执行归一化，最后根据输出的outputScale以及outputZeroPoint执行量化。
- 计算公式：
  
  1.反量化计算：
  
  $$
  x' = (x - inputZeroPoint) * inputScale
  $$
  
  2.归一化计算：
  
  $$
  y =\frac{x' - mean}{\sqrt{var + ϵ}} * γ + β
  $$
  
  3.量化计算：
  
  $$
  out = round(\frac{y}{outputScale} + outputZeroPoint)
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
      <td>模型输入的量化后的数据，对应公式中的`x`。</td>
      <td>INT8、UINT8、INT32</td>
      <td>NCHW</td>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>模型输入数据的均值，对应公式中的`mean`。shape维度限制为1维，长度需与输入NCHW中的C相同。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输入</td>
      <td>模型输入数据的方差，对应公式中的`var`。shape维度限制为1维，长度需与输入NCHW中的C相同。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_scale</td>
      <td>输入</td>
      <td>输入标量，模型输入数据的缩放系数，对应公式中的`inputScale`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_zero_point</td>
      <td>输入</td>
      <td>输入标量，模型输入数据的偏置，对应公式中的`inputZeroPoint`。传入值不能超过input对应数据类型的上下边界，例如INT8上下边界为[-128,127]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_scale</td>
      <td>输入</td>
      <td>输入标量，模型输出数据的缩放系数，对应公式中的`outputScale`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_zero_point</td>
      <td>输入</td>
      <td>输入标量，模型输出数据的偏置，对应公式中的`outputZeroPoint`。传入值不能超过input对应数据类型的上下边界，例如INT8上下边界为[-128,127]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>进行批量归一化的权重，对应公式中的`γ`。shape维度限制为1维，长度需与输入参数input的数据格式NCHW中的C相同。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>进行批量归一化的偏置值，对应公式中的`β`。shape维度限制为1维，长度需与输入NCHW中的C相同。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到方差中的小值以避免除以零，对应公式中的`ε`。</li><li>默认值为空。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>模型输出的量化后的数据，对应公式中的`y`。数据类型、数据格式、shape与输入`x`保持一致。</td>
      <td>INT8、UINT8、INT32</td>
      <td>NCHW</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_quantized_batch_norm](examples/test_aclnn_quantized_batch_norm.cpp) | 通过[aclnnQuantizedBatchNorm](docs/aclnnQuantizedBatchNorm.md)接口方式调用QuantizedBatchNorm算子。 |