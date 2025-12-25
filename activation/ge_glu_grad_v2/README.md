# GeGluGradV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

算子功能：完成[aclnnGeGlu](../ge_glu_v2/docs/aclnnGeGlu.md)和[aclnnGeGluV3](../ge_glu_v2/docs/aclnnGeGluV3.md)的反向。

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>公式中的输入gradOutput。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>公式中的输入self。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gelu</td>
      <td>输入</td>
      <td>公式中的输入gelu。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>取值范围为[-self.dim(), self.dim()-1]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>approximate</td>
      <td>属性</td>
      <td>取值范围是0('none')、1('tanh')。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
      <tr>
      <td>activateLeft</td>
      <td>属性</td>
      <td><ul><li>激活函数操作数据块的方向。</li><li>false表示对右边做activate。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>公式中的gradInput。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- aclnnGeGluBackward接口不包含activateLeft参数。
## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_ge_glu_backward](./examples/test_aclnn_ge_glu_backward.cpp) | 通过[aclnnGeGluBackward](./docs/aclnnGeGluBackward.md)接口方式调用GeGluGradV2算子。    |
| aclnn调用 | [test_aclnn_ge_glu_v3_backward](./examples/test_aclnn_ge_glu_v3_backward.cpp) | 通过[aclnnGeGluV3Backward](./docs/aclnnGeGluV3Backward.md)接口方式调用GeGluGradV2算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/ge_glu_grad_v2_proto.h)构图方式调用GeGluGradV2算子。 |
