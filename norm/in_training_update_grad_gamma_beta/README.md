# INTrainingUpdateGradGammaBeta

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

本算子用于InstanceNorm训练反向传播的参数梯度归约。它分别将`res_gamma`和`res_beta`
沿第0维求和并保留规约维，得到`pd_gamma`和`pd_beta`：

$$
pd\_gamma = \sum_{n=0}^{N-1} res\_gamma[n, \ldots], \qquad
pd\_beta = \sum_{n=0}^{N-1} res\_beta[n, \ldots].
$$

输出shape与输入shape相同，但第0维固定为1。

## 参数说明

<table>
<col style="width: 170px"><col style="width: 170px"><col style="width: 200px"><col style="width: 200px"><col style="width: 170px">
<tr><td>参数名</td><td>输入/输出/属性</td><td>描述</td><td>数据类型</td><td>数据格式</td></tr>
<tr><td>res_gamma</td><td>输入</td><td>`gamma`梯度的部分和，规约轴为第0维。</td><td>FLOAT</td><td>NCHW、NHWC、NCDHW、NDHWC、ND、NDC1HWC0</td></tr>
<tr><td>res_beta</td><td>输入</td><td>`beta`梯度的部分和，shape、数据类型和数据格式须与`res_gamma`完全相同。</td><td>FLOAT</td><td>与`res_gamma`相同</td></tr>
<tr><td>pd_gamma</td><td>输出</td><td>`res_gamma`沿第0维求和并保留规约维的结果。</td><td>FLOAT</td><td>与输入相同</td></tr>
<tr><td>pd_beta</td><td>输出</td><td>`res_beta`沿第0维求和并保留规约维的结果，shape与`pd_gamma`相同。</td><td>FLOAT</td><td>与输入相同</td></tr>
</table>

### 产品差异说明

| 产品 | 参数或场景 | 静态shape能力 | 动态shape能力 | shape/rank及组合限制 |
|:---|:---|:---|:---|:---|
| <term>Ascend 950PR&950DT系列产品</term> | 输入、输出均为FLOAT | NCHW、NHWC、NCDHW、NDHWC、ND | NCHW、NHWC、NCDHW、NDHWC、ND，支持动态rank | NCHW、NHWC对应4D；NCDHW、NDHWC对应5D；ND支持4D或5D；支持下文所述空Tensor语义；不支持NDC1HWC0。 |
| <term>Atlas A3系列产品</term><br><term>Atlas A2系列产品</term><br><term>Atlas 200I/500 A2推理产品</term><br><term>Atlas推理系列产品</term><br><term>Atlas训练系列产品</term> | 公共GE IR逻辑接口为FLOAT、4D NCHW/NHWC；CANN built-in内部执行采用NDC1HWC0 | 内部执行/存储格式为NDC1HWC0 | 支持动态shape的存量产品，其unknown-shape内部执行/存储格式为NDC1HWC0 | NDC1HWC0是built-in内部格式，不是<term>Ascend 950PR&950DT系列产品</term>的用户输入格式；本目录新增的5D、ND和空Tensor扩展不改变存量产品行为。 |

## 约束说明

- 本算子无属性，规约轴固定为第0维并保留规约维。
- 两个输入必须具有相同的shape、数据类型和数据格式，不支持广播。
- <term>Ascend 950PR&950DT系列产品</term>
  - 输入rank恒为4或5，对全部数据格式生效（ND亦不豁免）：NCHW、NHWC对应4维输入，NCDHW、NDHWC对应5维输入；ND用作稠密张量执行视图（与四种公有格式的稠密布局逐元素一致），rank同样须为4或5。rank为0、1、2、3或大于等于6的输入一律拒绝。
  - 各维取值无额外上限限制：各维为非负int64（实际上限由设备内存与全shape元素总数约束），不设逐维数值上限；第0维与其他维的边界行为一致（0维触发空集/空张量语义，见下）。
  - 数值语义：使用fp32累加并按第0维顺序求和，内部可根据输入选择不同补偿策略，对外统一按fp32sum精度容差验收。输入含NaN的列输出NaN，同列同时含+Inf与-Inf时输出NaN，有限真值超出FLOAT范围时输出对应符号的Inf。本算子不承诺与其他产品实现或不同tiling配置的位级一致。
  - 支持动态shape与动态rank：shape/dtype推导阶段对unknown-rank输入原样透传；对含-1未知维的shape，非规约维原样拷贝，第0维仍按规约语义置1（如输入[-1,64,3,5]推导输出为[1,64,3,5]）。执行阶段具体化后的shape仍须满足上述rank约束且各维非负（任何负维拒绝）。
  - 扩展的空Tensor语义：第0维为0时执行空集求和，输出元素为0；非规约维为0时输出为空张量。
  - 仅支持参数表中列出的稠密数据格式，不支持`NDC1HWC0`或`NC1HWC0`。
- 输入与输出均须为连续稠密张量（调用方内存义务）：不支持转置、切片、步长变化等非连续视图，调用方持有此类视图时须先转换为连续内存再构图传入；带存储偏移的连续视图等价于以数据起始地址为基址的稠密张量。输出必须为连续稠密张量，不支持非连续输出。
- 输入仅读、输出独立写（调用方内存义务）：不支持输入与输出内存重叠（alias）与原地修改，调用后输入内容保持不变。
- 以下约束由算子在GE图编译阶段（shape/dtype推导或tiling校验）拒绝并返回失败，图执行不启动：数据类型与数据格式（含两输入一致性）、rank与shape一致性、origin数据格式与rank的配对（unknown-rank输入在推导阶段暂缓、具体化后在tiling复核）、动态维负值。各拒绝分支均产出算子侧诊断日志，失败原因可通过GE运行日志（plog）或`GEGetErrorMsgV2`获取；本算子不承诺公开数值错误码，错误语义以日志信息为准。
- 输出shape与dtype由推导决定：GE图模式下输出tensor由框架按推导结果分配（shape为输入shape第0维置1、dtype为float32）。若执行通路保留了调用方声明的具体输出shape或dtype，算子会校验其与推导结果一致。
- 连续性与内存不重叠不在图编译阶段校验之列：shape/dtype推导与tiling校验没有内存地址与步长输入通道，违反连续性或内存重叠义务的调用不会被编译期拒绝，行为未定义。
- 本目录不提供aclnn单算子接口，也不提供PyTorch/TensorFlow扩展通路，唯一调用方式为GE图模式。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|:---|:---|:---|
| GE图模式 | [构图样例](examples/test_geir_in_training_update_grad_gamma_beta.cpp) | 通过[算子IR](op_graph/in_training_update_grad_gamma_beta_proto.h)创建`INTrainingUpdateGradGammaBeta`节点。 |
