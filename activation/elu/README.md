# Elu

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能: 对输入张量self中的每个元素x调用指数线性单元激活函数ELU，并将得到的结果存入输出张量out中。
- 计算公式：

  $$
  ELU(x) =
  \begin{cases}
  scale \ast x, \quad x > 0\\
  \alpha \ast scale \ast (exp(x \ast inputScale)-1), \quad x \leq 0
  \end{cases}
  $$

- **参数说明：**
  
  <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示ELU激活函数的输入，公式中的x。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>表示ELU激活函数的激活系数，公式中的α。</td>
      <td>数据类型需要是可转换为FLOAT的数据类型。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>scale</td>
      <td>输入</td>
      <td>表示ELU激活函数的缩放系数，公式中的scale。</td>
      <td>数据类型需要是可转换为FLOAT的数据类型。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>inputScale</td>
      <td>输入</td>
      <td>表示ELU激活函数的输入的缩放系数，公式中的inputScale。</td>
      <td>数据类型需要是可转换为FLOAT的数据类型。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示ELU激活函数的输出。</td>
      <td>shape需要与self一致。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
  </tbody>
  </table>
  
   - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_elu](./examples/test_aclnn_elu.cpp) | 通过[aclnnElu&aclnnInplaceElu](./docs/aclnnElu&aclnnInplaceElu.md)接口方式调用Elu算子。 |
| aclnn调用 | [test_aclnn_inplace_elu](./examples/test_aclnn_inplace_elu.cpp) | 通过[aclnnElu&aclnnInplaceElu](./docs/aclnnElu&aclnnInplaceElu.md)接口方式调用Elu算子。 |