# MaskedSoftmaxWithRelPosBias

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：替换在swinTransformer中使用window attention计算softmax的部分。
- 计算公式：

  $$
  out=\operatorname{softmax}(scaleValue*x+attenMaskOptional+relativePosBias)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
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
      <td>对应公式中的x。支持shape为4维(B*W, N, S1, S2)和5维(B, W, N, S1, S2)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>可选输入</td>
      <td>对应公式中的attenMaskOptional。当不需要时为空指针。支持shape为3维(W, S1, S2)、4维(W, 1, S1, S2)和5维(1, W, 1, S1, S2)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>relativePosBias</td>
      <td>输入</td>
      <td>对应公式中的relativePosBias。支持shape为3维(N, S1, S2)、4维(1, N, S1, S2)和5维(1, 1, N, S1, S2)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>可选属性</td>
      <td>对应公式中的scaleValue。</td>
      <td>DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>innerPrecisionMode</td>
      <td>可选属性</td>
      <td>精度模式参数。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>对应公式中的out。支持shape为4维(B*W, N, S1, S2)和5维(B, W, N, S1, S2)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_masked_softmax_with_rel_pos_bias](examples/test_aclnn_masked_softmax_with_rel_pos_bias.cpp) | 通过[aclnnMaskedSoftmaxWithRelPosBias](docs/aclnnMaskedSoftmaxWithRelPosBias.md)接口方式调用MaskedSoftmaxWithRelPosBias算子。 |
| 图模式 | [test_aclnn_masked_softmax_with_rel_pos_bias](examples/test_aclnn_masked_softmax_with_rel_pos_bias.cpp)  | 通过[算子IR](op_graph/masked_softmax_with_rel_pos_bias_proto.h)构图方式调用MaskedSoftmaxWithRelPosBias算子。         |
