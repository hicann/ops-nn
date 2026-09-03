# InplaceApplyAddSign

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

InplaceApplyAddSign是TensorFlow `ResourceApplyAddSign`在Ascend上的等价实现，属于训练优化器算子族。以AddSign算法对可训练变量var与一阶动量m进行原地（in-place）更新，每个训练step根据当前梯度grad计算符号一致性自适应步长并更新var与m，更新后的值原地写回。适用于CV、NLP、推荐系统、生成模型等使用AddSign优化器的训练场景。

计算公式：

$$
m_{out} = beta \times m + (1 - beta) \times grad
$$

$$
var_{out} = var - lr \times (alpha + sign\_decay \times sign(grad) \times sign(m_{out})) \times grad
$$

其中`sign(x)`遵循numpy.sign语义：x>0返回+1，x<0返回-1，x==0返回0，NaN返回0。var与m原地写回，lr、alpha、sign_decay、beta为rank=0标量输入，按numpy广播规则参与计算。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
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
</tr>
</thead>
<tbody>
<tr>
<td>var</td>
<td>输入</td>
<td>待更新的可训练变量，对应公式中var，原地写回更新后的值。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>m</td>
<td>输入</td>
<td>一阶动量缓冲区，对应公式中m，原地写回更新后的值。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>输入</td>
<td>学习率，对应公式中lr，rank=0标量。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>alpha</td>
<td>输入</td>
<td>步长增量基准值，对应公式中alpha，rank=0标量。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>sign_decay</td>
<td>输入</td>
<td>sign一致性的衰减权重，对应公式中sign_decay，rank=0标量。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>beta</td>
<td>输入</td>
<td>动量衰减系数，对应公式中beta，rank=0标量。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>输入</td>
<td>当前step梯度，对应公式中grad。</td>
<td>BFLOAT16、FLOAT16、FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>可选属性</td>
<td>是否对var与m的更新加锁，默认false，Ascend单算子执行模式下true与false行为一致。</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- use_locking属性在Ascend单算子执行模式下true与false行为一致，仅用于接口对齐TensorFlow。
- var、m、grad三个张量输入必须具有相同的shape与数据类型。
- lr、alpha、sign_decay、beta必须为标量：接受rank=0标量，也接受shape为(1,)的标量张量。
- 张量输入的维度范围为0-8。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| 图模式调用 | [test_geir_inplace_apply_add_sign](./examples/arch35/test_geir_inplace_apply_add_sign.cpp) | 通过[算子IR](./op_graph/inplace_apply_add_sign_proto.h)构图方式调用InplaceApplyAddSign算子，覆盖静态Shape场景。 |
