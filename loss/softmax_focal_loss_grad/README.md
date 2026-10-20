# SoftmaxFocalLossGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                              |    ×     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算SoftmaxFocalLoss的反向梯度。输入pred为已完成softmax的概率，输出对pred的梯度。

- 计算公式：

  设pred为$p$（shape为$[N, C]$），target为$t$，dout为$\text{d}$，weight为$w$（缺省时按全1处理），对第$n$个样本先求四个行标量：

  $$
  \text{WF}_n = \sum_c \alpha (1-p_{n,c})^{\gamma} t_{n,c}, \quad
  \text{WB}_n = \sum_c \alpha (1-p_{n,c})^{\gamma-1} t_{n,c}
  $$

  $$
  \text{CE}_n = \sum_c -\log(p_{n,c}) t_{n,c} w_{n,c}, \quad
  \text{W}_n = \sum_c w_{n,c} t_{n,c}
  $$

  再逐元素组合：

  $$
  \text{dce}_{n,c} = p_{n,c}\text{W}_n - t_{n,c}w_{n,c}
  $$

  $$
  \text{dwf}_{n,c} = -\gamma \cdot p_{n,c} \cdot \big[(\text{WF}_n - \text{WB}_n) + \alpha (1-p_{n,c})^{\gamma-1} t_{n,c}\big]
  $$

  $$
  \text{grad}_{n,c} = \big(\text{dce}_{n,c}\text{WF}_n + \text{dwf}_{n,c}\text{CE}_n\big) \cdot \text{d}_{n,c} \cdot \text{coef}
  $$

  其中：

  - $N$为batch数，$C$为类别数，归约轴为最后一维。
  - $\alpha$、$\gamma$分别为属性alpha、gamma。
  - reduction取"mean"时$\text{coef} = 1/(N \times C)$，取"none"或"sum"时$\text{coef} = 1$。
  - $(1-p)^{\gamma}$按$\exp(\gamma \log(1-p))$求值。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 40px">
  <col style="width: 60px">
  <col style="width: 266px">
  <col style="width: 60px">
  <col style="width: 40px">
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
      <td>pred</td>
      <td>输入</td>
      <td>前级softmax输出的概率，公式中的p。取值落在(0, 1)开区间，各维长度须大于0（不支持空Tensor），最后一维为类别维、其余维为样本维。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>one-hot标签，公式中的t，shape与pred一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dout</td>
      <td>输入</td>
      <td>上游传入的梯度，公式中的d，shape与数据类型均与pred一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td>逐元素权重，公式中的w，shape与数据类型均与pred一致。不传入时按全1处理，该行为仅<term>Ascend 950PR&950DT系列产品</term>支持，其余产品需显式传入weight。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>属性</td>
      <td>调制因子的权重系数，公式中的alpha，缺省值为0.25。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>属性</td>
      <td>调制因子的指数，公式中的gamma，缺省值为2.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>归约方式，支持'none'、'mean'、'sum'（大小写不敏感），缺省值为'mean'。取'mean'时梯度乘以1/numel(pred)，'none'与'sum'不缩放。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输出</td>
      <td>pred的梯度，shape与数据类型均与pred一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| 图模式调用 | [test_geir_softmax_focal_loss_grad.cpp](examples/test_geir_softmax_focal_loss_grad.cpp) | 通过[算子IR](./op_graph/softmax_focal_loss_grad_proto.h)构图方式调用SoftmaxFocalLossGrad算子。 |
