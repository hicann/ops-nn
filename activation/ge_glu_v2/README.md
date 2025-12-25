# GeGluV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：高斯误差线性单元激活门函数，针对aclnnGeGlu，扩充了设置激活函数操作数据块方向的功能。
- 计算公式：
  若activateLeft为true，表示对$self$的左半部分做activate

    $$
    out_{i}=GeGlu(self_{i}) = Gelu(A) \cdot B
    $$

  若activateLeft为false，表示对$self$的右半部分做activate

    $$
    out_{i}=GeGlu(self_{i}) = A \cdot Gelu(B)
    $$

  其中，$A$表示$self$的左半部分，$B$表示$self$的右半部分。
  
## 参数说明

<table style="undefined;table-layout: fixed; width: 919px"><colgroup>
  <col style="width: 130px">
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
      <td>self</td>
      <td>输入</td>
      <td>公式中的输入self。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>可选属性</td>
      <td>设定的slice轴，需要对self对应的轴进行对半切，同时dim对应的self的轴必须是双数。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>approximate</td>
      <td>可选属性</td>
      <td>GeGlu计算使用的激活函数索引，0表示使用“none”，1表示使用“tanh”。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateLeft</td>
      <td>属性</td>
      <td>表示激活函数操作数据块的方向，false表示对右边做activate，true表示对左边做activate。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的dx。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
     <td>outGelu</td>
      <td>输出</td>
      <td>公式中的outGelu。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- GeGlu不包含activateLeft参数。
## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_ge_glu](./examples/test_aclnn_ge_glu.cpp) | 通过[aclnnGeGlu](./docs/aclnnGeGlu.md)接口方式调用GeGluV2算子。    |
| aclnn调用 | [test_aclnn_ge_glu_v3](./examples/test_aclnn_ge_glu_v3.cpp) | 通过[aclnnGeGluV3](./docs/aclnnGeGluV3.md)接口方式调用GeGluV2算子。    |
| 图模式调用 | -  | 通过[算子IR](./op_graph/ge_glu_v2_proto.h)构图方式调用GeGluV2算子。 |
