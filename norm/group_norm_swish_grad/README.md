# GroupNormSwishGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：[aclnnGroupNormSwish](../../norm/group_norm_swish/docs/aclnnGroupNormSwish.md)的反向操作。


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
      <td>反向计算的梯度。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>正向计算的第二个输出，表示input分组后每个组的均值。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td>正向计算的第三个输出，表示input分组后每个组的标准差倒数。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>正向的输入x。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>每个channel的缩放系数。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示每个channel的偏移系数。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numGroups</td>
      <td>属性</td>
      <td>表示将输入gradOut的C维度分为group组。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dataFormatOptional</td>
      <td>属性</td>
      <td>数据格式。</td>
      <td>Char*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>swishScale</td>
      <td>属性</td>
      <td>Swish计算公式中的系数。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dgammaIsRequire</td>
      <td>属性</td>
      <td>是否需要输出dgamma。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dbetaIsRequire</td>
      <td>属性</td>
      <td>是否需要输出dbeta。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dxOut</td>
      <td>输出</td>
      <td>x的输出梯度。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dgammaOut</td>
      <td>输出</td>
      <td>gamma的输出梯度。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dbetaOut</td>
      <td>输出</td>
      <td>beta的输出梯度。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_group_norm_swish_grad.cpp](examples/test_aclnn_group_norm_swish_grad.cpp) | 通过[aclnnGroupNormSwishGrad](docs/aclnnGroupNormSwishGrad.md)接口方式调用GroupNormSwishGrad算子。 |