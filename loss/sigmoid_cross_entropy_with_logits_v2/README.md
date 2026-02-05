# SigmoidCrossEntropyWithLogitsV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 接口功能：计算输入logits与标签target之间的BCELoss损失。

- 计算公式：

  - 单标签场景：

    $$
    \ell(self, target) = L = \{l_{1},..., l_{n}\}^{T}
    $$

    $$
    \ell_{n} = -weight_{n}[target_{n} \cdot log(\sigma(self_{n})) + (1 - target_{n}) \cdot log(1 - \sigma(self_{n}))]
    $$

    $$
    \ell(self, target) =
    \begin{cases}
    L, & if\ reduction = none\\
    mean(L), & if\ reduction = mean\\
    sum(L), & if\ reduction = sum\\
    \end{cases}
    $$

  - 多标签场景：

    $$
    \ell_c(self, target) = L_c = \{l_{1,c},..., l_{n,c}\}^{T}
    $$

    $$
    \ell_{n,c} = -weight_{n,c}[pos\_weight_{n,c} \cdot target_{n,c} \cdot log(\sigma(self_{n,c})) + (1 - target_{n,c}) \cdot log(1 - \sigma(self_{n,c}))]
    $$

- **参数说明：**

   <table style="undefined;table-layout: fixed; width: 1290px"><colgroup>
    <col style="width: 183px">
    <col style="width: 120px">
    <col style="width: 263px">
    <col style="width: 222px">
    <col style="width: 152px">
    <col style="width: 101px">
    <col style="width: 104px">
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
        <td>连接层输出。</td>
        <td>-</td>
        <td>FLOAT16、FLOAT、BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>target</td>
        <td>输入</td>
        <td>label标签值。</td>
        <td>-</td>
        <td>与“self”一致</td>
        <td>ND</td>
        <td>与“self”一致</td>
        <td>√</td>
      </tr>
      <tr>
      <td>weightOptional</td>
      <td>输入</td>
      <td>二分交叉熵权重。</td>
      <td>shape需要能够broadcast到“target”</td>
      <td>与“self”一致</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
      </tr>
      <tr>
        <td>posWeightOptional</td>
        <td>输入</td>
        <td>各类的正类权重。</td>
        <td>shape需要能够broadcast到“target”</td>
        <td>与“self”一致</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>输入</td>
        <td>输出结果计算方式。</td>
        <td>直调算子当前仅支持none值：<ul><li>none表示不做任何操作</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>输出误差。</td>
        <td>如果reduction = None，shape与self一致，其他情况shape为[1]</td>
        <td>与“target”一致</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
    </tbody></table>

  - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [aclnnBinaryCrossEntropyWithLogits](./docs/aclnnBinaryCrossEntropyWithLogits.md) | 通过[aclnnBinaryCrossEntropyWithLogits](./docs/aclnnBinaryCrossEntropyWithLogits.md)接口方式调用SigmoidCrossEntropyWithLogitsV2算子。 |