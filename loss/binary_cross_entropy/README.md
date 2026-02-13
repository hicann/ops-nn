# BinaryCrossEntropy

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算x和y的二元交叉熵。
- 计算公式:
  
  当reduction为None时：

  $$
  \ell(x, y)= L = \{l_{1},...,l_{n}\}^{T}, \ell_{n} = -  weight_{n}[y_{n}·log(x_{n}) + (1 - y_{n}·log(1-x_{n}))]
  $$

  当reduction不为None时：

  $$
  \ell(x, y)
  \begin{cases}
  mean(L), & if\ reduction = mean \\
  sum(L), & if\ reduction = sum \\
  \end{cases}
  $$


## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示预测的概率值，公式中的输入`x`。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>表示目标张量，公式中的输入`y`。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示权重张量，公式中的输入`weight`。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>表示计算输出</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>表示规约方式，公式中的输入`reduction`，输出规约的枚举值。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>

  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_binary_cross_entropy.cpp](examples/test_aclnn_binary_cross_entropy.cpp) | 通过[aclnnBinaryCrossEntropy](docs/aclnnBinaryCrossEntropy.md)接口方式调用BinaryCrossEntropy算子。 |