# AdaLayerNormV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：AdaLayerNormV2算子将LayerNorm和下游的Add、Mul融合起来，通过自适应参数scale和shift来调整归一化过程。相比AdaLayerNorm算子，输出新增2个参数（输入的均值和输入的标准差的倒数）；weight和bias支持的数据类型增加对应约束。
- 计算公式：
  
  $$
  out = LayerNorm(x) * (1 + scale) + shift
  $$

  LayerNorm计算公式：
  
  $$
  mean = E(x)
  $$

  $$
  rstd = {1.0\over\sqrt {Var(x)+epsilon}}
  $$

  $$
  LayerNorm(x) = (x-mean) * rstd * weight + bias
  $$

  其中，E(x)表示输入的均值，Var(x)表示输入的方差。

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
      <td>表示计算的输入张量。对应公式中的`x`。shape为[B, S, H]，其中B支持0到6维。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>表示自适应缩放参数。对应公式中的`scale`。shape为[B, H]或[B, 1, H]，其中B支持0到6维，维度数量和大小与`x`中的B保持一致，H与`x`中H维一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>输入</td>
      <td>表示自适应偏移参数。对应公式中的`shift`。shape为[B, H]或[B, 1, H]，其中B支持0到6维，维度数量和大小与`x`中的B保持一致，H与`x`中H维一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td>表示归一化缩放参数。对应公式中的`weight`。shape为[H]，H与`x`中H维一致。数据类型与入参`x`的数据类型一致或为FLOAT32类型，且当`bias`存在时，`weight`与`bias`的数据类型相同。shape为[H]，H与`x`中H维一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>表示归一化偏移参数。对应公式中的`bias`。数据类型与入参`x`的数据类型一致或为FLOAT32类型，且当`bias`存在时，`weight`与`bias`的数据类型相同。shape为[H]，H与`x`中H维一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>表示添加到分母中的值，以确保数值稳定。对应公式中的`epsilon`。</li><li>默认值为1e-5f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示计算的输出张量。对应公式中的`out`。shape与`x`保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输出</td>
      <td>表示归一化后的均值。对应公式中的`mean`。shape为[B, S, 1]，最后一维固定为1，其他维度大小和`x`一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>
      <td>表示归一化后的标准差倒数。对应公式中的`rstd`。shape为[B, S, 1]，最后一维固定为1，其他维度大小和`x`一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_ada_layer_norm_v2](examples/test_aclnn_ada_layer_norm_v2.cpp) | 通过[aclnnAdaLayerNormV2](docs/aclnnAdaLayerNormV2.md)接口方式调用AdaLayerNormV2算子。 |
<!--
| 图模式 | -  | 通过[算子IR](op_graph/ada_layer_norm_proto.h)构图方式调用AdaLayerNormV2算子。         |-->

<!--[test_geir_ada_layer_norm](examples/test_geir_ada_layer_norm.cpp)-->