# LpNormReduce

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                   |    √     |
| <term>Atlas A3系列产品</term> |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                  |    √     |
| <term>Atlas推理系列产品</term>                          |    √     |
| <term>Atlas训练系列产品</term>                          |    √     |

## 功能说明

- 算子功能：计算Lp范数的归约段。对输入`x`逐元素取绝对值后，沿属性`axes`指定的轴做归约，归约算子由属性`p`决定。本算子只做归约，不做最后的$\frac{1}{p}$次开方。完整的Lp范数由LpNormReduce与匹配的LpNormUpdate两个算子串联得到。当前旧链的LpNormUpdate只支持FLOAT16和FLOAT，因此BFLOAT16仅是本算子的单节点能力。
- 计算公式：记$a_i = \lvert x_i \rvert$，$S$为规范化后的归约轴集合（`axes`为空表示全轴，负轴按`axis + rank`折算，重复轴去重）。

  当`p`为0时，统计非零元素个数：

  $$
  y = \sum_{i \in S}{[a_i \neq 0]}
  $$

  当`p`为1时：

  $$
  y = \sum_{i \in S}{a_i}
  $$

  当`p`为其余非负有限整数时：

  $$
  y = \sum_{i \in S}{a_i^p}
  $$

  当`p`为2147483647（表示$+\infty$）时：

  $$
  y = \max\limits_{i \in S}{a_i}
  $$

  当`p`为-2147483648（表示$-\infty$）时：

  $$
  y = \min\limits_{i \in S}{a_i}
  $$

## 参数说明

| 参数名  | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------- | -------------- | ---- | -------- | -------- |
| x       | 必选输入       | 公式中的`x`，表示待归约张量。rank范围为0到8，支持动态shape与动态rank。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| y       | 必选输出       | 公式中的输出`y`。数据类型与`x`一致，shape由`axes`与`keepdim`决定。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| p       | 可选属性       | 公式中的`p`，表示范数的整数阶。默认值为2。2147483647表示正无穷，-2147483648表示负无穷；除此之外`p`须为非负整数。 | INT | - |
| axes    | 可选属性       | 公式中的归约轴集合$S$。默认值为{}。空列表表示对所有轴归约；取值范围为[-rank(`x`), rank(`x`))，支持负轴与重复轴。 | LISTINT | - |
| keepdim | 可选属性       | 决定输出张量是否保留归约轴。为true时归约轴保留为长度1；为false时删除归约轴，全轴归约时输出为0维标量。默认值为false。 | BOOL | - |
| epsilon | 可选属性       | 为与旧ABI保持一致而保留的属性，在本算子中不参与计算，由后续LpNormUpdate消费。默认值为1e-12。 | FLOAT | - |

### 产品差异说明

| 产品 | 实现与数据类型 | shape/rank能力及限制 |
| ---- | -------------- | -------------------- |
| <term>Ascend 950PR&950DT系列产品</term> | 使用本仓Ascend C实现；支持FLOAT16、FLOAT、BFLOAT16和ND格式。 | 支持静态shape、动态shape和动态rank，rank范围为0到8。除两个无穷哨兵外，支持完整的非负INT64 `p`范围，有限阶幂运算保持整数原值。BFLOAT16只适用于LpNormReduce单节点；当前旧链LpNormUpdate不支持BFLOAT16。 |
| <term>Atlas A3系列产品</term><br><term>Atlas A2系列产品</term><br><term>Atlas 200I/500 A2推理产品</term><br><term>Atlas推理系列产品</term><br><term>Atlas训练系列产品</term> | 使用CANN内置TBE实现；支持FLOAT16、FLOAT和ND格式，不支持BFLOAT16。 | 支持静态shape、动态shape和动态rank，rank范围为0到8。 |

## 约束说明

- 本算子是面向旧GEIR固化图的兼容算子，新增场景请使用LpNormReduceV2算子。
- 本算子只输出Lp范数的归约结果，不做$\frac{1}{p}$次开方；FLOAT16/FLOAT完整范数可再串联匹配的LpNormUpdate算子。当前旧链LpNormUpdate不支持BFLOAT16。
- 以下输入边界描述<term>Ascend 950PR&950DT系列产品</term>实现。
- 输入rank范围为0到8。rank-0输入只允许空`axes`；已知rank下越界轴会报错，重复轴按首次出现去重。
- 图推导阶段允许unknown rank和取值为-1的动态维，其他负维非法。执行期`x`的逻辑shape与storage shape都必须具体；非标量的rank及每一维必须完全相同，rank-0标量允许storage shape为`[]`或`[1]`。执行期`y`的逻辑shape与storage shape必须与`axes`、`keepdim`推导结果逐维一致且具体；标量输出的逻辑/storage shape兼容`[]`和`[1]`两种物化形式。为保证归约分段规模可表示，`x`所有非零维度的乘积不得超过INT64_MAX（含空Tensor）。
- 有限`p`的归约域为空时输出0；若零长度维仅位于非归约轴，输出为空Tensor。正负无穷哨兵遇到空归约域时因`max`/`min`无定义而报错。
- 除-2147483648哨兵外，有限负`p`会报错。2147483648及更大的非负值仍按有限整数阶处理。
- FLOAT16和BFLOAT16输入的绝对值、幂与归约累加中间过程以FLOAT完成，结束后转换回输入数据类型；输出数据类型始终与输入一致。
- 当`axes`指定到`x`中长度为1的轴时，计算结果可能存在精度差异。
- 仅提供GEIR/Kernel兼容路径，不新增aclnn接口。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| 图模式   | -        | 通过[LpNormReduce算子IR入口](op_graph/lp_norm_reduce_proto.h)构图；该入口转发到仓库公共原型，未提供aclnn接口。 |
