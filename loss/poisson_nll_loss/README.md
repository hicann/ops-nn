# PoissonNllLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算泊松分布的负对数似然损失（Poisson negative log likelihood loss），常用于计数数据回归，对标PyTorch的`torch.nn.PoissonNLLLoss`。逐元素计算损失后按reduction归约。
- 计算公式：

  当log_input为true时：

  $$
  loss = e^{input\_x} - target \cdot input\_x
  $$

  当log_input为false时：

  $$
  loss = input\_x - target \cdot ln(input\_x + eps)
  $$

  当full为true时，对 $target > 1$ 的元素额外添加斯特林近似项（ $target \leq 1$ 时该项为0）：

  $$
  loss = loss + \left(target \cdot ln(target) - target + 0.5 \cdot ln(2\pi \cdot target)\right)
  $$

  其中reduction为none时输出逐元素损失（shape与input_x相同）；为sum时输出所有元素之和（标量）；为mean时输出所有元素均值（标量，即除以元素总数N）。

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
      <td>input_x</td>
      <td>输入</td>
      <td>预测值张量，公式中的input_x。数据类型需要与target一致，shape需要与target相同。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>目标值张量，公式中的target。数据类型需要与input_x一致，shape需要与input_x相同。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>log_input</td>
      <td>属性</td>
      <td>控制是否对输入取指数，取值为true/false，默认true，具体计算见计算公式。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>full</td>
      <td>属性</td>
      <td>控制是否添加斯特林近似项，取值为true/false，默认false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>log_input为false时防止ln(0)的小常数，默认1e-8，不能为0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>表示对逐元素损失做的reduce操作，取值为"none"/"mean"/"sum"，默认"mean"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>输出</td>
      <td>损失计算结果，公式中的loss。数据类型与输入一致，reduction为none时shape与input_x相同，为mean/sum时为标量。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- input_x、target的数据类型必须相同，同为FLOAT16或FLOAT32。
- input_x与target的shape必须完全相同（不支持broadcast）；输出loss的数据类型与输入一致。
- eps不能为0。
- reduction取值仅支持"none"/"mean"/"sum"。
- 支持空Tensor（元素总数为0）输入：reduction=none时输出空Tensor；sum时输出0；mean时输出nan（0/0）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_poisson_nll_loss.cpp](examples/arch35/test_geir_poisson_nll_loss.cpp) | 通过算子IR构图方式调用PoissonNllLoss算子。 |
