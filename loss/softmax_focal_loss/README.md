# SoftmaxFocalLoss

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    √     |
| <term>Atlas推理系列产品</term>                              |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算多分类场景下的Softmax Focal Loss。Focal Loss在交叉熵的基础上引入调制因子$(1-p)^\gamma$，降低易分样本的损失权重，使训练更关注难分样本。输入pred为已完成softmax的概率，本算子不再内置softmax。

- 计算公式：

  设pred为$p$（shape为$[N, C]$），target为$t$（与$p$同shape的one-hot标签），weight为$w$（与$p$同shape，缺省时按全1处理），对第$n$个样本：

  $$
  \text{CE}_n = \sum_{c=0}^{C-1} -\log(p_{n,c}) \cdot t_{n,c} \cdot w_{n,c}
  $$

  $$
  \text{FW}_n = \sum_{c=0}^{C-1} \alpha \cdot \exp\big(\gamma \cdot \log(1 - p_{n,c})\big) \cdot t_{n,c}
  $$

  $$
  y_{n,c} = \text{CE}_n \cdot \text{FW}_n
  $$

  其中：

  - $N$为batch数，$C$为类别数，归约轴为最后一维。
  - $\alpha$、$\gamma$分别为属性alpha、gamma。
  - $y$与$p$同shape，同一行内各元素取值相同。

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
      <td>前级softmax输出的概率，公式中的p。取值落在(0, 1)开区间，各维长度须大于0（不支持空Tensor），最后一维为类别维、其余维为样本维。<term>Ascend 950PR&950DT系列产品</term>：支持rank≥1；其余产品：仅支持(batch_size, num_classes)二维。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>one-hot标签，公式中的t，shape与pred一致，取值须为0或1。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td>逐元素权重，公式中的w，shape与pred一致；不传入时按全1处理。数据类型可与pred不同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>属性</td>
      <td>调制因子的指数，公式中的gamma，缺省值为2.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>属性</td>
      <td>调制因子的权重系数，公式中的alpha，缺省值为0.25。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>属性</td>
      <td>归约方式，支持'none'、'mean'、'sum'，缺省值为'mean'。输出y的shape随该属性：'none'时与pred同形，'mean'、'sum'时为标量。当前仅支持'none'，须显式传入。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>损失值，shape与pred一致，同一行内各元素取值相同。数据类型与pred一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| 图模式调用 | [test_geir_softmax_focal_loss.cpp](examples/test_geir_softmax_focal_loss.cpp) | 通过[算子IR](./op_graph/softmax_focal_loss_proto.h)构图方式调用SoftmaxFocalLoss算子。 |
