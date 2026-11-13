# L2Loss

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

- 算子功能：计算输入x中所有元素平方和的一半，即L2正则化损失。输出y为标量。

- 计算公式：

  $$
  y = \frac{1}{2} \sum_{i} x_i^2
  $$

  其中$x$是输入`x`，$i$遍历$x$中的所有元素。

- 计算实现：为规避平方运算的中间溢出，实现上先将输入乘以$\frac{1}{\sqrt{2}}$再做平方累加，数学上等价于$\frac{1}{2}\sum_i x_i^2$。输入为FLOAT16或BFLOAT16时，片上提升到FLOAT32完成乘法与累加后再回写为输入数据类型。

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
<tr><th>参数名</th><th>输入/输出/属性</th><th>描述</th><th>数据类型</th><th>数据格式</th></tr>
</thead>
<tbody>
<tr><td>x</td><td>输入</td><td>公式中的输入x，待计算L2损失的张量，元素总数可为0。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
<tr><td>y</td><td>输出</td><td>公式中的输出y，0维标量，数据类型与x一致。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
</tbody>
</table>

## 约束说明

- 数据类型与格式：`x`与`y`的数据类型必须一致，数据格式仅支持ND。本算子不校验越界数据类型与格式，调用方须自行保证入参合法。
- 输出形状：`y`恒为0维标量，与`x`的shape无关；`y`的数据在图执行完成后可读。
- 输入Rank与逐维边界：支持动态shape与动态rank；Rank上限由共享reduce模板与UB容量决定，本仓测试用例覆盖Rank 1~5。`x`元素总数为0时，输出为0。
- 非连续与原地语义：本算子不提供非连续Tensor语义，GE图模式下Tensor的连续性由图编译器负责转换；`y`为独立输出，不支持与`x`原地复用或内存重叠。
- 特殊值与精度：`x`中的inf与nan按IEEE 754规则传播到输出。FLOAT16与BFLOAT16虽在片上以FLOAT32累加，但回写时受自身表示范围限制，大量值或大幅值输入可能溢出为inf。
- 确定性：归约的分核与累加次序由tiling依据输入shape与芯片核数决定，不保证在不同shape、不同芯片配置间的逐bit一致。
- 0维输入与1维空Tensor：在CANN 9.3.0的const-shape（cst）与dynamic（dyn）编译模式下，受编译框架对0维标量输出的代码生成限制，Rank为0的输入与shape为(0,)的输入编译失败；binary（bin）模式不受影响。
- 调用通路：本算子仅提供GE图模式调用，不提供aclnn接口，也不提供PyTorch或TensorFlow端到端绑定，详见“调用说明”。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---- | ---- | ---- |
| GE图模式 | [test_geir_l2_loss.cpp](examples/test_geir_l2_loss.cpp) | 通过[算子IR](op_graph/l2_loss_proto.h)构图方式调用L2Loss算子。 |

其他通路的当前支持情况：

- aclnn API：不提供。本算子在[CMakeLists.txt](CMakeLists.txt)中以`ACLNNTYPE aclnn_exclude`声明排除aclnn接口生成，目录下无`op_api`实现，调用`aclnnL2Loss`将返回接口不存在。
- PyTorch API：不提供端到端绑定。
- TensorFlow：[framework/l2_loss_tf_plugin.cpp](framework/l2_loss_tf_plugin.cpp)仅提供TensorFlow 1.x（npu_bridge）风格的GE图映射注册，用于把TensorFlow图中的L2Loss节点映射到本算子，不构成可独立运行的端到端通路，且依赖对应版本的TensorFlow与npu_bridge运行环境。
