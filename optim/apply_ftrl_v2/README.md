# ApplyFtrlV2

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

ApplyFtrlV2 实现 FTRL（Follow The Regularized Leader）优化算法的 V2 版本参数更新规则。该算子是纯逐元素计算算子，在模型训练的每步迭代中根据梯度、学习率和正则化参数就地更新权重变量（var）、梯度平方累加器（accum）和线性累加器（linear）。适用于大规模稀疏特征的在线学习场景，如广告点击率（CTR）预估、推荐系统等。

**V2 与 V1（ApplyFtrl）的区别**：V2 新增 `l2Shrinkage` 参数，支持 shrinkage-type L2 正则化。V2 在梯度上叠加 `2 * l2Shrinkage * var` 的修正量，等价于对损失函数直接添加 L2 惩罚项（`Loss += l2Shrinkage * ||w||²`）。当 `l2Shrinkage = 0` 时，V2 退化为 V1。

**计算公式**（7 步更新）：

$$
\begin{aligned}
\hat{g} &= g + 2 \cdot \lambda_s \cdot w \\
n' &= n + g^2 \\
\sigma &= \frac{{n'}^{-p} - n^{-p}}{\eta} \\
z &\leftarrow z + \hat{g} - \sigma \cdot w \\
x &= \lambda_1 \cdot \text{sign}(z) - z \\
y &= \frac{{n'}^{-p}}{\eta} + 2 \cdot \lambda_2 \\
w &\leftarrow \begin{cases} x / y & \text{if } |z| > \lambda_1 \\ 0 & \text{otherwise} \end{cases} \\
n &\leftarrow n'
\end{aligned}
$$

其中：$w$=var（权重），$n$=accum（梯度平方累积），$z$=linear（线性累加器），$g$=grad（梯度），$\eta$=lr（学习率），$\lambda_1$=l1，$\lambda_2$=l2，$\lambda_s$=l2Shrinkage，$p$=lrPower。


## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>var</td>
    <td>输入 / 输出(inplace)</td>
    <td>模型权重张量，对应公式中 $w$。Kernel内inplace更新，GE IR单输出视图与输入var共享Device内存。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输入(inplace更新)</td>
    <td>梯度平方累加器，对应公式中 $n$。shape/dtype必须与var一致；Kernel内显式写回输入GM地址。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>linear</td>
    <td>输入(inplace更新)</td>
    <td>线性累加器，对应公式中 $z$。shape/dtype必须与var一致；Kernel内显式写回输入GM地址。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前步梯度，对应公式中 $g$。shape/dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率（0-d tensor），对应公式中 $\eta$。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l1</td>
    <td>输入</td>
    <td>L1 正则化强度（0-d tensor），对应公式中 $\lambda_1$。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2</td>
    <td>输入</td>
    <td>L2 正则化强度（0-d tensor），对应公式中 $\lambda_2$。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2Shrinkage</td>
    <td>输入</td>
    <td>L2 shrinkage 强度（0-d tensor），对应公式中 $\lambda_s$（V2 新增）。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lrPower</td>
    <td>输入</td>
    <td>学习率衰减幂次（0-d tensor），对应公式中 $p$。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>是否在更新时加锁。默认false。当前实现不强制互斥锁，仅作语义占位。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>输出</td>
    <td>更新后的var Tensor，与输入var共享Device内存（inplace）。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

> **注**：accum、linear 为 in-place 更新（与 TensorFlow 一致），不作为显式输出返回，计算结果直接写回输入地址。

## 约束说明

- **数据类型一致性**：var、accum、linear、grad 四个 tensor 的数据类型必须一致；lr、l1、l2、l2Shrinkage、lrPower 五个 scalar 的数据类型必须与 tensor 一致。
- **Shape 约束**：var、accum、linear、grad 的 shape 必须完全相同；lr、l1、l2、l2Shrinkage、lrPower 必须为 scalar（0-d tensor）；tensor 维度范围 0-8 维。
- **参数值域**：lr > 0（学习率必须为正数）、l1 >= 0、l2 >= 0、l2Shrinkage >= 0、lrPower <= 0。
- **In-place 语义**：var、accum、linear 三个参数为就地更新，执行后原始数据被覆盖。
- **FP16/BF16 计算**：FP16 和 BF16 输入在 kernel 内部提升到 FP32 计算，结果 cast 回原始类型输出。
- **确定性**：默认确定性实现。该算子为纯逐元素操作，无归约或排序，计算完全确定性。
- **不支持 `multiply_linear_by_lr` 属性**：以 TensorFlow ApplyFtrlV2 标准路径为准（`multiply_linear_by_lr=false`）。
- **空 Tensor**：支持空 Tensor（numel=0），kernel 跳过计算。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:---------|:---------|:-----|
| 图模式 | [test_geir_apply_ftrl_v2](./examples/arch35/test_geir_apply_ftrl_v2.cpp) | 通过 [算子IR](./op_graph/apply_ftrl_v2_proto.h) 构图方式调用 ApplyFtrlV2 算子。 |
