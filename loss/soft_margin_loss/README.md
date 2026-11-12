# SoftMarginLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算输入self和目标target的二分类逻辑损失函数。
- 计算公式:

当`reduction`为'none'时：

$$
\text{out}(self,target)=L=log(1+\exp(-target[i]*self[i]))
$$

如果`reduction`为`mean`或`sum`时，

$$
\text{out}(self,target)=\begin{cases}
mean(L), & \text{if reduction} = \text{'mean'}\\
sum(L), & \text{if reduction} = \text{'sum'}
\end{cases}
$$

- 其中：
- self: 输入张量
- target: 真实标签

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
    <td>self</td>
    <td>输入</td>
    <td>公式中的'self'，表示输入张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>target</td>
    <td>输入</td>
    <td>公式中的'target'，表示真实标签。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>loss</td>
    <td>输出</td>
    <td>表示self与target经soft_margin_loss计算得到的损失值输出</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>reduction</td>
    <td>属性</td>
    <td>表示对逐元素损失值进行均值/求和/无归约的归约参数，支持'none'、'mean'、'sum'</td>
    <td>STRING</td>
    <td>-</td>
    </tr>

</tbody></table>

## 约束说明

- 形状与维度：self/target 的维度不超过 8 维（0 维标量亦可）。
- 空 tensor：aclnn 接口支持空输入。
- 数据格式：仅支持 ND。
- 数值边界（算子在float32精度下计算，FLOAT16/BFLOAT16输入同样适用）：
  - 溢出：当逐元素满足 $-target \times self > \ln(FLT\_MAX) \approx 88.72$ 时，$\exp(-target \times self)$ 溢出为+inf，输出 $log(1+inf)=inf$；`reduction`为'mean'或'sum'时，任一元素为inf会使标量输出为inf。典型场景为预测方向严重错误（如target=1且self取较大的负值）。
  - 下溢：实现使用 log1p 稳定计算 $\log(1+\exp(-target \times self))$。当 $target \times self$ 较大时，输出约为 $\exp(-target \times self)$，不会因 $1+\exp(-target \times self)$ 舍入为1而提前变为0。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_soft_margin_loss.cpp](examples/test_aclnn_soft_margin_loss.cpp) | 通过[aclnnSoftMarginLoss](docs/aclnnSoftMarginLoss.md)接口方式调用SoftMarginLoss算子。 |
| 图模式（GEIR） | [test_geir_soft_margin_loss.cpp](examples/arch35/test_geir_soft_margin_loss.cpp) | 通过[算子IR](op_graph/soft_margin_loss_proto.h)构图方式调用SoftMarginLoss算子。 |
