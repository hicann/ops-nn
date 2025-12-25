# AdaLayerNormQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：AdaLayerNormQuant算子将AdaLayerNorm和下游的量化（目前仅支持DynamicQuant）融合起来。该算子主要是用于执行自适应层归一化的量化操作，即将输入数据进行归一化处理，并将其量化为低精度整数，以提高计算效率和减少内存占用。

- 计算公式：
  
  1.先对输入x进行LayerNorm归一化处理：
  
    $$
    LayerNorm(x) = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * weight + bias
    $$

  2.再通过自适应参数scale和shift来调整归一化结果：
  
    $$
    y = LayerNorm(x) * (1 + scale) + shift
    $$

  3.若smooth_scales不为空，则：
  
    $$
    y = y \cdot smooth_scales
    $$

  4.然后对y计算最大绝对值并除以127以计算需量化为INT8格式的量化因子：
  
    $$
    quant_scale = row\_max(abs(y)) / 127
    $$

  5.最后y除以量化因子再四舍五入得到量化输出：
  
    $$
    out = round(y / quant_scale)
    $$

  其中，E(x)表示输入的均值，Var(x)表示输入的方差，row_max代表每行求最大值。

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
      <td>表示输入待处理的数据。对应公式中的`x`。shape为[B, S, H]，其中B支持0到6维。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>表示自适应缩放参数。对应公式中的`scale`。shape为[B, H]或[B, 1, H]，其中B支持0到6维，维度数量和大小与`x`中的B保持一致，H与`x`中H维一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>输入</td>
      <td>表示自适应偏移参数。对应公式中的`shift`。shape为[B, H]或[B, 1, H]，其中B支持0到6维，维度数量和大小与`x`中的B保持一致，H与`x`中H维一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td>表示归一化缩放参数。对应公式中的`weight`。shape为[H]，H与`x`中H维一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>表示归一化偏移参数。对应公式中的`bias`。shape为[H]，H与`x`中H维一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>可选输入</td>
      <td>表示量化的平滑权重。对应公式中的`smooth_scales`。shape为[H]，H与`x`中H维一致。</td>
      <td>FLOAT16、BFLOAT16</td>
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
      <td>表示量化输出张量。对应公式中的`out`。shape与入参`x`的shape保持一致。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_scale</td>
      <td>输出</td>
      <td>表示量化系数。对应公式中的`quant_scale`。shape为[B, S]，其中B支持0到6维，维度数量和大小与`x`中的B保持一致，S与`x`中S维一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_ada_layer_norm_quant](examples/test_aclnn_ada_layer_norm_quant.cpp) | 通过[aclnnAdaLayerNormQuant](docs/aclnnAdaLayerNormQuant.md)接口方式调用AdaLayerNormQuant算子。 |

<!--
| 图模式 | -  | 通过[算子IR](op_graph/ada_layer_norm_quant_proto.h)构图方式调用AdaLayerNormQuant算子。         |
-->

<!--[test_geir_ada_layer_norm_quant](examples/test_geir_ada_layer_norm_quant.cpp)-->
<!--[test_aclnn_ada_layer_norm_quant](examples/test_aclnn_ada_layer_norm_quant.cpp)-->