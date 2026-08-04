# SGD

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

> 上表写的是SGD在各产品形态上的**可得性**，不是本次交付的架构范围。本仓的Ascend C实现只适配 <term>Ascend 950PR/Ascend 950DT</term>（`sgd_def.cpp`中仅`AddConfig("ascend950")`）；其余产品形态上的SGD由CANN内置的TBE实现提供，语义一致，但不由本算子承载。

## 功能说明

- **算子功能**：带动量的随机梯度下降（SGD）优化器更新算子，训练迭代中就地更新一组权重。

- **计算公式**：

  记$d$为`dampening`、$wd$为`weightDecay`、$lr$为`learningRate[0]`、$m$为`momentum[0]`，逐元素计算：

  **步骤一** 权重衰减（仅$wd \neq 0$时执行，否则$grad = gradient$）：

  $$
  grad = gradient + parameters \times wd
  $$

  **步骤二** 动量累积（**无条件执行**）：

  $$
  accum_t = accum \times m + grad
  $$

  **步骤三** 阻尼修正（仅$d \neq 0$时执行）。$stat$是**逐元素**的首步标记，取值1表示该元素处于首步、不施加阻尼：

  $$
  accum_t = accum_t - grad \times (1 - stat) \times d
  $$

  **步骤四** 权重更新（**无条件写出**）：

  $$
  parameters_{out} =
  \begin{cases}
  parameters - (grad \times lr + accum_t \times m \times lr), & nesterov = true \\
  parameters - accum_t \times lr, & nesterov = false
  \end{cases}
  $$

  **步骤五** 动量与标记回写，受$m \neq 0$掩码控制：

  $$
  accum_{out}, stat_{out} =
  \begin{cases}
  accum_t,\ 0, & m \neq 0 \\
  \text{保持输入原值（不回写）}, & m = 0
  \end{cases}
  $$

- **计算精度**：中间计算在float32域进行，结果按就近偶数舍入（round-half-to-even）回目标数据类型。`learningRate`与`momentum`与`parameters`同数据类型，故float16/bfloat16下这两个标量本身已被量化。

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
    <td>parameters</td>
    <td>输入 / 输出（原地）</td>
    <td>待更新的权重。<b>无条件被改写</b>。维度数（rank）须在<code>[1, 8]</code>内；不支持空tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>gradient</td>
    <td>输入</td>
    <td>梯度。shape与数据类型须与parameters一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>learning_rate</td>
    <td>输入</td>
    <td>学习率。<b>shape须为<code>[1]</code>或0维标量</b>，数据类型须与parameters一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>accum</td>
    <td>输入 / 输出（原地）</td>
    <td>动量累积量。<b>仅momentum ≠ 0时被改写</b>；momentum = 0时逐位保持原值。shape与数据类型须与parameters一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>momentum</td>
    <td>输入</td>
    <td>动量因子。<b>shape须为<code>[1]</code>或0维标量</b>，数据类型须与parameters一致。取值为0（含<code>-0.0</code>）时触发“不回写”语义。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>stat</td>
    <td>输入 / 输出（原地）</td>
    <td>逐元素首步标记，取值1表示该元素处于首步、不施加阻尼。<b>仅momentum ≠ 0时被改写为0</b>；momentum = 0时逐位保持原值。shape与数据类型须与parameters一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>dampening</td>
    <td>属性</td>
    <td>动量阻尼系数，默认值0.0。<b>nesterov为true时必须为0</b>。</td>
    <td>FLOAT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weight_decay</td>
    <td>属性</td>
    <td>权重衰减系数，默认值0.0。<b>必须大于或等于0</b>。</td>
    <td>FLOAT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>nesterov</td>
    <td>属性</td>
    <td>是否启用Nesterov动量，默认值false。</td>
    <td>BOOL</td>
    <td>-</td>
    </tr>
    <tr>
    <td>parameters</td>
    <td>输出</td>
    <td>更新后的权重，与输入parameters为同一块内存。shape、数据类型、数据格式均与输入parameters一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
    </tr>
</tbody>
</table>

## 约束说明

- **三路原地回写，但图上仅声明1个输出**：算子实际就地更新`parameters`、`accum`、`stat`三个张量，而图原型只声明`parameters`一个输出，`accum`与`stat`通过覆写其输入内存返回。调用方必须把这三者都视为可写。此形态与CANN内置实现一致。

- **momentum = 0时的回写语义**：`momentum`为0（含`-0.0`）时，`accum`与`stat` **完全不被写入**，逐位保持输入原值（包括NaN的具体位模式、`±inf`、`-0.0`）；`parameters`不受该掩码影响，任何`momentum`取值下都照常计算并写出。`momentum`为极小非零值（如`1e-8`、`1e-30`）时按非零处理，正常回写。

- **⚠️ 从PyTorch迁移的差异告警**：本算子`momentum = 0`时的“不回写”方向与`torch.optim.SGD`一致（PyTorch在`momentum == 0`时整块跳过动量更新）。但 **PyTorch的`dampening`施加在该判断之内，本算子（与CANN内置实现一致）施加在判断之外**。因此当`momentum = 0`且`dampening > 0`且`stat = 0`时，`parameters`的更新量与PyTorch相差$(1 - dampening)$倍。仅当`dampening = 0`或`stat = 1`时两者一致。从PyTorch迁移的调用方须感知此差异。

- **rank与空tensor**：`parameters`的维度数须在`[1, 8]`内，**0维标量被拒绝**。**不支持空tensor** —— 任意一轴或多轴为0均判为非法并返回错误码，不存在“空进空出”语义（`accum`/`stat`的原地回写在元素数为0时无定义）。

- **属性取值**：`nesterov = true`时`dampening`必须为0；`weight_decay`必须大于或等于0。违反者返回参数非法错误码。

- **inf/NaN**：按IEEE 754语义传播，不做钳制或特判。特别地，`accum`含`±inf`且`momentum = 0`时，$accum \times momentum$产生的NaN会按IEEE语义传播进`parameters`。

- **确定性**：输出逐位可复现。算子为纯逐元素计算、无跨元素累加，多核切分不改变任一元素的计算顺序。

- **张量连续性**：所有输入须为连续张量。本算子不提供aclnn接口，无接口层做转连续/回填，非连续视图由调用方（GE图编译期）负责处理。

## 调用说明

> **不提供aclnn单算子接口。** SGD在CANN上游本就是纯图模式算子：CANN 9.1.0的
> `include/aclnnop/`下无`aclnn_sgd.h`（只有语义不同的`aclnn_fused_sgd.h`），
> `libopapi.so`未导出`aclnnSgd*`符号，canndev全仓亦无`aclnn_sgd`定义。
> 本算子与之对齐，只支持GE图模式下发。

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_sgd.cpp](examples/test_geir_sgd.cpp) | 通过GE图方式调用SGD算子。 |

## 参考资源

- [《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)：算子开发的概念原理与编程模型。
- [算子列表](../../docs/zh/op_list.md)：本项目全部算子的分类、调用方式与功能说明。
- [算子调用快速入门](../../docs/zh/invocation/quick_op_invocation.md)：算子样例的编译与运行步骤。
- [apply_momentum](../apply_momentum/README.md)：同族的动量优化器算子，与本算子结构最相近，可对照阅读。
- [fused_sgd](../fused_sgd/README.md)：语义**不同**的另一个算子（多TensorList融合、`dampening`施加在`momentum`分支之内、无`stat`）。名称相近，请勿混用。
