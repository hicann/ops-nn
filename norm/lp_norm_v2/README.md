# LpNormV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |


## 功能说明

- 接口功能：返回给定张量的矩阵范数或者向量范数。

- 计算公式：支持1/2范数、无穷范数以及其他`p`为float类型的范数计算。
  - 1-范数：  
  
    $$
    \Vert x \Vert = \sum_{i=1}^{N}{\vert x_i \vert}
    $$

  - 2-范数（默认值）：  

    $$
    \Vert x \Vert_2 = (\sum_{i=1}^{N}{\vert x_i \vert^2})^{\frac{1}{2}}
    $$

  - 无穷范数：
  
    $$
    \Vert x \Vert_\infty = \max\limits_{i}{\vert x_i \vert}
    $$
  
    $$
    \Vert x \Vert_{-\infty} = \min\limits_{i}{\vert x_i \vert}
    $$

  - p范数：
  
    $$
    \Vert x \Vert_p = (\sum_{i=1}^{N}{\vert x_i \vert^p})^{\frac{1}{p}}
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
      <td>self</td>
      <td>输入</td>
      <td>公式中的x。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pScalar</td>
      <td>输入</td>
      <td>表示范数的类型，公式中的p。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>计算self范数的维度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>输入</td>
      <td>决定输出张量是否保留dim参数指定的轴。</td>
      <td>BOOL </td>
      <td>-</td>
    </tr>
    <tr>
      <td>relType</td>
      <td>输入</td>
      <td>预留参数，暂不支持使用。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的out。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>：数据类型不支持BFLOAT16。

## 约束说明
无

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_norm](./examples/test_aclnn_norm.cpp) | 通过[aclnnNorm](docs/aclnnNorm.md)接口方式调用LpNormV2算子。 |
| aclnn接口 | [test_aclnn_linalg_vector_norm](./examples/test_aclnn_LinalgVectorNorm.cpp) | 通过[aclnnLinalgVectorNorm](docs/aclnnLinalgVectorNorm.md)接口方式调用LpNormV2算子。 |