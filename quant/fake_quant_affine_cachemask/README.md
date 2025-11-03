# FakeQuantAffineCachemask

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
 

## 功能说明

- 算子功能：对于输入数据self，使用scale和zero_point对输入self在指定轴axis上进行伪量化处理，并根据quant_min和quant_max对伪量化输出进行值域更新，最终返回结果out及对应位置掩码mask。
- 计算公式：根据算子功能先计算临时变量qval，再计算得出out和mask。

  $$
  qval = Round(std::nearby\_int(self / scale) + zero\_point)
  $$

  $$
  out = (Min(quant\_max, Max(quant\_min, qval)) - zero\_point) * scale
  $$

  $$
  mask = (qval >= quant\_min)   \&  (qval <= quant\_max)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>self</td>
      <td>输入</td>
      <td>公式中的`self`。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>公式中的`scale`，表示输入伪量化的缩放系数。shape只支持1维。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zeroPoint</td>
      <td>输入</td>
      <td>公式中的`zeroPoint`，表示输入伪量化的零基准参数。shape只支持1维。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入</td>
      <td>表示计算维度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMin</td>
      <td>输入</td>
      <td>表示输入数据伪量化后的最小值。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMax</td>
      <td>输入</td>
      <td>表示输入数据伪量化后的最大值。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的`out`。维度和计算输入`self`一致。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输出</td>
      <td>公式中的`mask`。维度和计算输入`self`一致。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_fake_quant_per_channel_affine_cachemask.cpp](examples/test_aclnn_fake_quant_per_channel_affine_cachemask.cpp) | 通过[aclnnFakeQuantPerChannelAffineCachemask](docs/aclnnFakeQuantPerChannelAffineCachemask.md)接口方式调用FakeQuantAffineCachemask算子。 |
| aclnn接口  | [test_aclnn_fake_quant_per_tensor_affine_cachemask.cpp](examples/test_aclnn_fake_quant_per_tensor_affine_cachemask.cpp) | 通过[aclnnFakeQuantPerTensorAffineCachemask](docs/aclnnFakeQuantPerTensorAffineCachemask.md)接口方式调用FakeQuantAffineCachemask算子。 |
<!--| 图模式 | [test_geir_fake_quant_affine_cachemask.cpp](examples/test_geir_fake_quant_affine_cachemask.cpp)  | 通过[算子IR](op_graph/fake_quant_affine_cachemask_proto.h)构图方式调用FakeQuantAffineCachemask算子。         |-->
