# MVNV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：对4-D NCHW输入张量沿指定axes计算均值与标准差并执行归一化。理想数学语义下输出在规约轴上均值为0；当标准差显著大于eps时，方差接近1。

- 计算公式：

$$
y = \frac{x - mean}{\sqrt{var} + eps}
$$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 4-D NCHW输入张量，对应公式中x。 | FLOAT16、FLOAT | NCHW |
| y | 输出 | 归一化输出张量，shape与dtype均与x相同，对应公式中y。 | FLOAT16、FLOAT | NCHW |
| eps | 可选属性 | 加在std上（std + eps）的浮点值，默认1e-9；负值和非有限值按IEEE浮点语义传播。 | FLOAT | - |
| axes | 可选属性 | 规约轴列表，元素取值范围为[-4, 4)，负值按axis + 4归一化；未传入或传入空列表时取默认值[0, 2, 3]；重复轴去重，顺序不影响结果。 | INT64 | - |

## 约束说明

- 输入必须为4-D张量；图描述允许使用动态维标记-1，InferShape会将其传播到输出，但进入运行时tiling前storage shape必须已具体化为4-D，tiling阶段拒绝任何非4-D输入与负值维度。
- axes元素经负值归一化后须在[0, 4)内，越界会在tiling阶段被拒绝；未传入axes或传入空列表时使用默认值[0, 2, 3]。非空axes按集合语义处理，重复轴幂等且顺序不影响结果。
- eps只约束为FLOAT类型。负值、NaN和正负Inf为兼容输入，按字面公式与IEEE浮点语义传播；真正的FP32次正规eps走不缩放兼容路径，具体次正规执行仍受平台FTZ模式影响。
- 分母严格按公式计算为`sqrt(var) + eps`，不额外做最小正规数clamp；方差为0且eps为0时按浮点除零语义传播。
- FLOAT16输入会提升为FLOAT进行均值和方差统计，FLOAT输入使用FLOAT中间计算；OpDef关闭精度降级，禁止FLOAT图路径被自动转换为FLOAT16；长规约路径使用补偿累加。
- 有限常数规约组在eps大于0时输出0；NaN与Inf不按常数组处理，按浮点运算传播。次正规输入和中间结果遵循当前昇腾平台的浮点执行模式，精度验收使用精度2.1的cross_check L0标准。
- 空Tensor输入直接返回同shape的空Tensor输出，不执行kernel计算。
- 数据类型仅支持FLOAT16与FLOAT，输出shape和数据类型必须与输入一致；冲突描述若到达tiling会被拒绝。
- 输入输出逻辑格式为NCHW；OpDef为输入输出声明AutoContiguous，由框架在需要时完成连续化；tiling接受NCHW或GE派生的dense ND storage，裸Kernel仍要求连续storage。
- 4-D shape乘积必须可由INT64表示，arch35 DMA行stride必须小于2^40字节，否则tiling拒绝。实际可执行规模还受图框架张量描述能力、设备可用内存和运行时资源限制。实现内部使用INT64表示shape、stride和元素偏移。
- rank、维度、shape乘积、输入输出描述、DMA stride、UB预算和axes等非法场景会在GE图校验、编译或operator tiling中的最早可达阶段被拒绝。到达tiling时返回GRAPH_FAILED，运行时通常呈现EZ9999；框架提前拒绝时错误码由对应GE阶段定义。算子自身日志使用算子名作为前缀并包含非法参数与实际取值。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_mvn_v2](./examples/test_geir_mvn_v2.cpp) | 通过算子IR定义[op_graph/mvn_v2_proto.h](./op_graph/mvn_v2_proto.h)接入Graph Engine图模式。 |
