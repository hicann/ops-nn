# SigmoidFocalLossGrad

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

SigmoidFocalLossGrad计算Sigmoid Focal Loss对前向logits `pred`的raw selector反向梯度，适用于目标检测等使用Focal Loss的训练反向过程。`target`是前向dense target的补集，不能直接当作高层框架的普通正类标签。

记`p=sigmoid(pred)`、`t=target`，且`weight`缺省时按全1处理，则：

$$
dpos = \alpha\gamma p(1-p)^\gamma\log(p)-\alpha(1-p)^{\gamma+1}
$$

$$
dneg = \gamma(\alpha-1)p^\gamma(1-p)\log(1-p)+(1-\alpha)p^{\gamma+1}
$$

$$
grad = (dpos(1-t)+dneg\,t)\times dout\times weight
$$

`reduction=mean`时逐元素结果再除以`pred`的元素数；`sum`和`none`不缩放，输出shape均与`pred`一致。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead><tr><th>参数名</th><th>输入/输出/属性</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr></thead>
<tbody>
<tr><td>pred</td><td>输入</td><td>前向logits，对应公式中pred。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
<tr><td>target</td><td>输入</td><td>raw backward selector，对应公式中t。</td><td>INT32</td><td>ND</td></tr>
<tr><td>dout</td><td>输入</td><td>上游梯度，对应公式中dout。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
<tr><td>weight</td><td>可选输入</td><td>逐元素样本权重，对应公式中weight；缺省时等价于全1。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
<tr><td>grad</td><td>输出</td><td>`pred`的梯度，数据类型和shape均跟随`pred`。</td><td>FLOAT16、FLOAT</td><td>ND</td></tr>
<tr><td>alpha</td><td>可选属性</td><td>类别平衡权重，对应公式中alpha，默认值为0.25。</td><td>FLOAT</td><td>-</td></tr>
<tr><td>gamma</td><td>可选属性</td><td>聚焦指数，对应公式中gamma，默认值为2.0。</td><td>FLOAT</td><td>-</td></tr>
<tr><td>reduction</td><td>可选属性</td><td>结果缩放方式，可取mean、sum或none，默认值为mean。</td><td>STRING</td><td>-</td></tr>
</tbody>
</table>

## 约束说明

- `pred`、`target`、`dout`、`grad`与存在时的`weight`必须是非空二维ND Tensor，shape完全相同，不支持广播。
- `target`只支持INT32，业务标签语义为0或1；IR和Kernel不在运行前逐元素检查Device侧数值。
- `pred`、`dout`和`weight`可分别使用FLOAT16或FLOAT，`grad`的数据类型必须与`pred`相同。
- `alpha`和`gamma`必须为有限数；`reduction`只能为mean、sum或none。
- Ascend 950图模式支持动态Shape（-1未知维），不支持动态Rank（-2未知秩）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | - | 通过[SigmoidFocalLossGrad IR定义](op_graph/sigmoid_focal_loss_grad_proto.h)构建算子图。 |
