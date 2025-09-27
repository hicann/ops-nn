# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[目录结构](../../README.md#目录结构)。
> - **算子执行位置**：大部分算子运行在AI Core上，少部分算子运行在AI CPU上。默认情况下，项目中提到的算子一般指AI Core算子。
> - 关于AI Core和AI CPU详细介绍请参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。

项目提供的所有算子分类和算子列表如下：

|  算子分类 |  算子目录   |    算子执行位置   |     说明     |
| -------- |--------------|-------|-----------------|
|	activation	|	[fatrelu_mul](../../activation/fatrelu_mul/README.md)	|		AI Core	|	将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行Threshold计算，将计算结果与x2相乘。	|
|	activation	|	[gelu_mul](../../activation/gelu_mul/README.md)	|		AI Core	|	将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行GELU计算，将计算结果与x2相乘。	|
|	activation	|	[gelu_quant](../../activation/gelu_quant/README.md)	|		AI Core	|	将GeluV2与DynamicQuant/AscendQuantV2进行融合，对输入的数据self进行GELU激活后，对激活的结果进行量化，输出量化后的结果。	|
|	activation	|	[ge_glu_grad_v2](../../activation/ge_glu_grad_v2/README.md)	|		AI Core	|	ge_glu_v2的反向传播。	|
|	activation	|	[ge_glu_v2](../../activation/ge_glu_v2/README.md)	|		AI Core	|	高斯误差线性单元激活门函数，针对aclnnGeGlu，扩充了设置激活函数操作数据块方向的功能。	|
|	activation	|	[heaviside](../../activation/heaviside/README.md)	|		AI Core	|	计算输入input中每个元素的Heaviside阶跃函数，作为模型的激活函数。	|
|	activation	|	[squared_relu](../../activation/squared_relu/README.md)	|		AI Core	|	基于标准Relu函数的变体，其主要特点是对Relu函数的输出进行平方，常作为模型的激活函数。	|
|	activation	|	[swi_glu](../../activation/swi_glu/README.md)	|		AI Core	|	Swish门控线性单元激活函数，实现x的SwiGlu计算。 	|
|	activation	|	[swi_glu_grad](../../activation/swi_glu_grad/README.md)	|		AI Core	|	swi_glu的反向传播，完成x的SwiGlu反向梯度计算。	|
|	control	|	[assert](../../control/assert/README.md)	|		AI Core	|	断言给定条件为 true，如果输入张量 input_condition 判定为 false，则打印 input_data 中的张量列表。	|
|	conv	|	[conv3d_backprop_filter_v2](../../conv/conv3d_backprop_filter_v2/README.md)	|		AI Core	|	计算三维卷积核权重张量$w$的梯度 $\frac{\partial L}{\partial w}$。	|
|	conv	|	[conv3d_backprop_input_v2](../../conv/conv3d_backprop_input_v2/README.md)	|		AI Core	|	计算三维卷积正向的输入张量$x$对损失函数$L$的梯度 $\frac{\partial L}{\partial x}$。	|
|	conv	|	[conv3d_transpose_v2](../../conv/conv3d_transpose_v2/README.md)	|		AI Core	|	计算三维卷积的转置（反卷积）相对于输入的梯度	|
|	conv	|	[conv3d_v2](../../conv/conv3d_v2/README.md)	|		AI Core	|	实现 3D 卷积功能。	|
|	foreach	|	[foreach_abs](../../foreach/foreach_abs/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的张量的绝对值。	|
|	foreach	|	[foreach_acos](../../foreach/foreach_acos/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，其中每个张量的元素都是原始张量对应元素的反余弦值。	|
|	foreach	|	[foreach_addcdiv_list](../../foreach/foreach_addcdiv_list/README.md)	|		AI Core	|	对多个张量进行逐元素加、乘、除操作，并返回一个新的张量列表。	|
|	foreach	|	[foreach_addcdiv_scalar](../../foreach/foreach_addcdiv_scalar/README.md)	|		AI Core	|	对多个张量进行逐元素加、乘、除操作，返回一个和输入张量列表同样形状大小的新张量列表，$x2_{i}$和$x3_{i}$进行逐元素相除，并将结果乘以scalar，再与$x1_{i}$相加。	|
|	foreach	|	[foreach_addcdiv_scalar_list](../../foreach/foreach_addcdiv_scalar_list/README.md)	|		AI Core	|	对多个张量进行逐元素加、乘、除操作，返回一个和输入张量列表同样形状大小的新张量列表，$x2_{i}$和$x3_{i}$进行逐元素相除，并将结果乘以$scalars_{i}$，再与$x1_{i}$相加。	|
|	foreach	|	[foreach_addcmul_list](../../foreach/foreach_addcmul_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，对张量列表x2和张量列表x3执行逐元素乘法，将结果乘以张量scalars后将结果与张量列表x1执行逐元素加法。	|
|	foreach	|	[foreach_addcmul_scalar](../../foreach/foreach_addcmul_scalar/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，对张量列表x2和张量列表x3执行逐元素乘法，将结果乘以标量值scalar后将结果与张量列表x1执行逐元素加法。	|
|	foreach	|	[foreach_addcmul_scalar_list](../../foreach/foreach_addcmul_scalar_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，对张量列表x2和张量列表x3执行逐元素乘法，将结果与标量列表scalars进行逐元素乘法，最后将结果与张量列表x1执行逐元素加法。	|
|	foreach	|	[foreach_add_list](../../foreach/foreach_add_list/README.md)	|		AI Core	|	两个Tensor列表中的元素逐个相加，并返回一个新的Tensor列表。可以通过设置alpha参数来调整相加的系数。	|
|	foreach	|	[foreach_add_scalar](../../foreach/foreach_add_scalar/README.md)	|		AI Core	|	将指定的标量值加到张量列表中的每个张量中，并返回更新后的张量列表。	|
|	foreach	|	[foreach_add_scalar_list](../../foreach/foreach_add_scalar_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行scalar相加运算的结果。	|
|	foreach	|	[foreach_asin](../../foreach/foreach_asin/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，按元素做反正弦函数运算。	|
|	foreach	|	[foreach_atan](../../foreach/foreach_atan/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，按元素做反正切函数运算。	|
|	foreach	|	[foreach_copy](../../foreach/foreach_copy/README.md)	|		AI Core	|	用于实现两个张量列表内容的复制，要求输入和输出两个张量列表形状相同。	|
|	foreach	|	[foreach_cos](../../foreach/foreach_cos/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，按元素做余弦函数运算。	|
|	foreach	|	[foreach_cosh](../../foreach/foreach_cosh/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，按元素做双曲余弦函数运算。	|
|	foreach	|	[foreach_div_list](../../foreach/foreach_div_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表, 对张量x1和张量x2执行逐元素除法。	|
|	foreach	|	[foreach_div_scalar](../../foreach/foreach_div_scalar/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表, 将张量x除以标量值scalar。	|
|	foreach	|	[foreach_div_scalar_list](../../foreach/foreach_div_scalar_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，对张量x和标量列表scalars执行逐元素除法。	|
|	foreach	|	[foreach_erf](../../foreach/foreach_erf/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，按元素做误差函数运算（也称之为高斯误差函数，error function or Gaussian error function）。	|
|	foreach	|	[foreach_erfc](../../foreach/foreach_erfc/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，按元素做误差函数运算（也称之为高斯误差函数，error function or Gaussian error function）。	|
|	foreach	|	[foreach_exp](../../foreach/foreach_exp/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行指数运算的结果。	|
|	foreach	|	[foreach_expm1](../../foreach/foreach_expm1/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行指数运算然后减1的结果。	|
|	foreach	|	[foreach_lerp_list](../../foreach/foreach_lerp_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，其中每个元素都是张量列表x1和张量列表x2对应位置上元素的线性插值结果，其中张量weight是插值系数。	|
|	foreach	|	[foreach_lerp_scalar](../../foreach/foreach_lerp_scalar/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，其中每个元素都是张量列表x1和张量列表x2对应位置上元素的线性插值结果，其中标量值weight是插值系数。	|
|	foreach	|	[foreach_log](../../foreach/foreach_log/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，按元素以e为底做对数函数运算。	|
|	foreach	|	[foreach_log10](../../foreach/foreach_log10/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，按元素以10为底做对数函数运算。	|
|	foreach	|	[foreach_log1p](../../foreach/foreach_log1p/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它对每一个元素先加一再进行以e为底的对数函数运算。	|
|	foreach	|	[foreach_log2](../../foreach/foreach_log2/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，按元素以2为底做对数函数运算。	|
|	foreach	|	[foreach_maximum_list](../../foreach/foreach_maximum_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，对张量列表x1和张量列表x2执行逐元素比较，返回最大值的张量。	|
|	foreach	|	[foreach_maximum_scalar](../../foreach/foreach_maximum_scalar/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表, 对张量列表和标量值scalar执行逐元素比较，返回最大值的张量。	|
|	foreach	|	[foreach_maximum_scalar_list](../../foreach/foreach_maximum_scalar_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表, 对张量列表x和标量列表scalar执行逐元素比较，返回最大值的张量。	|
|	foreach	|	[foreach_minimum_list](../../foreach/foreach_minimum_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，对张量列表x1和张量列表x2执行逐元素比较，返回最小值的张量。	|
|	foreach	|	[foreach_minimum_scalar](../../foreach/foreach_minimum_scalar/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，对张量列表x和标量值scalar执行逐元素比较，返回最小值的张量。	|
|	foreach	|	[foreach_minimum_scalar_list](../../foreach/foreach_minimum_scalar_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表, 对张量列表x和标量列表scalars执行逐元素比较，返回最小值的张量。	|
|	foreach	|	[foreach_mul_list](../../foreach/foreach_mul_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入两个张量列表的每个张量进行相乘运算的结果。	|
|	foreach	|	[foreach_mul_scalar](../../foreach/foreach_mul_scalar/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行scalar相乘运算的结果。|
|	foreach	|	[foreach_mul_scalar_list](../../foreach/foreach_mul_scalar_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行scalar相乘运算的结果。	|
|	foreach	|	[foreach_neg](../../foreach/foreach_neg/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表中每个张量的相反数。	|
|	foreach	|	[foreach_non_finite_check_and_unscale](../../foreach/foreach_non_finite_check_and_unscale/README.md)	|		AI Core	|	遍历scaledGrads中的所有Tensor，检查是否存在inf或NaN，如果存在则将foundInf设置为1.0，否则foundInf保持不变，并对scaledGrads中的所有Tensor进行反缩放。	|
|	foreach	|	[foreach_norm](../../foreach/foreach_norm/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行范数运算的结果。	|
|	foreach	|	[foreach_pow_list](../../foreach/foreach_pow_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行x2次方运算的结果。	|
|	foreach	|	[foreach_pow_scalar](../../foreach/foreach_pow_scalar/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行n次方运算的结果。	|
|	foreach	|	[foreach_pow_scalar_and_tensor](../../foreach/foreach_pow_scalar_and_tensor/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行x次方运算的结果。	|
|	foreach	|	[foreach_pow_scalar_list](../../foreach/foreach_pow_scalar_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行n次方运算的结果。	|
|	foreach	|	[foreach_reciprocal](../../foreach/foreach_reciprocal/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行倒数运算的结果。	|
|	foreach	|	[foreach_round_off_number](../../foreach/foreach_round_off_number/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行四舍五入到指定的roundMode小数位数运算的结果。	|
|	foreach	|	[foreach_sigmoid](../../foreach/foreach_sigmoid/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行Sigmoid函数运算的结果。	|
|	foreach	|	[foreach_sign](../../foreach/foreach_sign/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表中张量的符号值。	|
|	foreach	|	[foreach_sin](../../foreach/foreach_sin/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行正弦函数运算的结果。	|
|	foreach	|	[foreach_sinh](../../foreach/foreach_sinh/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行双曲正弦函数运算的结果。	|
|	foreach	|	[foreach_sqrt](../../foreach/foreach_sqrt/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行平方根运算的结果。	|
|	foreach	|	[foreach_sub_list](../../foreach/foreach_sub_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入的两个张量列表的相减运算的结果。	|
|	foreach	|	[foreach_sub_scalar](../../foreach/foreach_sub_scalar/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行scalar相减运算的结果。	|
|	foreach	|	[foreach_sub_scalar_list](../../foreach/foreach_sub_scalar_list/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行scalars相减运算的结果。	|
|	foreach	|	[foreach_tan](../../foreach/foreach_tan/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行正切函数运算的结果。	|
|	foreach	|	[foreach_tanh](../../foreach/foreach_tanh/README.md)	|		AI Core	|	返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行双曲正切函数运算的结果。	|
|	foreach	|	[foreach_zero_inplace](../../foreach/foreach_zero_inplace/README.md)	|		AI Core	|	原地更新输入张量列表，输入张量列表的每个张量置为0。	|
|	index	|	[apply_top_k_top_p_with_sorted](../../index/apply_top_k_top_p_with_sorted/README.md)	|		AI Core	|	对原始输入logits进行top-k和top-p采样过滤。	|
|	index	|	[embedding_bag](../../index/embedding_bag/README.md)	|		AI Core	|	根据indices从weight中获得一组被聚合的数，然后根据offsets的偏移和mode指定的聚合模式对获取的数进行max、sum、mean聚合。	|
|	index	|	[embedding_dense_grad_v2](../../index/embedding_dense_grad_v2/README.md)	|		AI Core	|	实现Embedding的反向计算，将相同索引indices对应grad的一行累加到out上。	|
|	index	|	[gather_elements_v2](../../index/gather_elements_v2/README.md)	|		AI Core	|	对输入tensor中指定的维度dim进行数据聚集。	|
|	index	|	[inplace_index_add_with_sorted](../../index/inplace_index_add_with_sorted/README.md)	|		AI Core	|	将alpha和value相乘，并按照sorted_indices中的顺序，累加到var张量的指定axis维度中。	|
|	index	|	[inplace_scatter_add](../../index/inplace_scatter_add/README.md)	|		AI Core	|	根据给定的indices，将updates中的值加到输入张量var的第一维度上。	|
|	index	|	[linear_index](../../index/linear_index/README.md)	|		AI Core	|	将多维数组 / 矩阵的元素位置映射为单一整数的索引方式。	|
|	index	|	[linear_index_v2](../../index/linear_index_v2/README.md)	|		AI Core	|	将多维数组 / 矩阵的元素位置映射为单一整数的索引方式。	|
|	index	|	[repeat_interleave_grad](../../index/repeat_interleave_grad/README.md)	|		AI Core	|	repeat_interleave_grad的反向传播, 将y_grad tensor的axis维度按repeats进行ReduceSum。	|
|	index	|	[scatter_add_with_sorted](../../index/scatter_add_with_sorted/README.md)	|		AI Core	|	将tensor src中的值按指定的轴和方向和对应的位置关系逐个替换/累加/累乘至tensor self中。	|
|	index	|	[scatter_elements_v2](../../index/scatter_elements_v2/README.md)	|		AI Core	|	将tensor src中的值按指定的轴和方向和对应的位置关系逐个替换/累加/累乘至tensor self中。	|
|	index	|	[scatter_list](../../index/scatter_list/README.md)	|		AI Core	|	将稀疏矩阵更新应用到变量引用中。	|
|	loss	|	[chamfer_distance_grad](../../loss/chamfer_distance_grad/README.md)	|		AI Core	|	ChamferDistance(倒角距离)的反向算子，根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。	|
|	loss	|	[cross_entropy_loss](../../loss/cross_entropy_loss/README.md)	|		AI Core	|	计算输入的交叉熵损失。	|
|	loss	|	[cross_entropy_loss_grad](../../loss/cross_entropy_loss_grad/README.md)	|		AI Core	|	cross_entropy_loss的反向传播。	|
|	loss	|	[cross_v2](../../loss/cross_v2/README.md)	|		AI Core	|	用于计算x1和x2张量在维度dim上的向量叉积。	|
|	loss	|	[ctc_loss_v3](../../loss/ctc_loss_v3/README.md)	|		AI Core	|	计算连续（未分段）时间序列与目标序列之间的损失。	|
|	loss	|	[ctc_loss_v3_grad](../../loss/ctc_loss_v3_grad/README.md)	|		AI Core	|	计算连续（未分段）时间序列与目标序列之间的损失的反向。	|
|	loss	|	[logit](../../loss/logit/README.md)	|		AI Core	|	该算子是概率到对数几率（log-odds）转换的一个数学运算，常用于概率值的反变换。它可以将一个介于0和1之间的概率值转换为一个实数值。	|
|	loss	|	[logit_grad](../../loss/logit_grad/README.md)	|		AI Core	|	logit的反向传播。	|
|	loss	|	[mse_loss_grad_v2](../../loss/mse_loss_grad_v2/README.md)	|		AI Core	|	均方误差函数mse_loss_v2的反向传播。	|
|	loss	|	[mse_loss_v2](../../loss/mse_loss_v2/README.md)	|		AI Core	|	计算输入x和目标y中每个元素之间的均方误差。	|
|	matmul	|	[batch_mat_mul_v3](../../matmul/batch_mat_mul_v3/README.md)	|		AI Core	|	完成带batch的矩阵乘计算。	|
|	matmul	|	[gemm_v2](../../matmul/gemm_v2/README.md)	|		AI Core	|	计算α乘以A与B的乘积，再与β和input C的乘积求和。	|
|	matmul	|	[mat_mul_v3](../../matmul/mat_mul_v3/README.md)	|		AI Core	|	完成通用矩阵乘计算。	|
|	matmul	|	[quant_batch_matmul_v3](../../matmul/quant_batch_matmul_v3/README.md)	|		AI Core	|	完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。	|
|	matmul	|	[quant_batch_matmul_v4](../../matmul/quant_batch_matmul_v4/README.md)	|		AI Core	|	完成量化的矩阵乘计算。	|
|	matmul	|	[quant_matmul_reduce_sum](../../matmul/quant_matmul_reduce_sum/README.md)	|		AI Core	|	完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出。	|
|	matmul	|	[transpose_batch_mat_mul](../../matmul/transpose_batch_mat_mul/README.md)	|		AI Core	|	完成张量x1与张量x2的矩阵乘计算。	|
|	matmul	|	[weight_quant_batch_matmul_v2](../../matmul/weight_quant_batch_matmul_v2/README.md)	|		AI Core	|	完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。	|
|	norm	|	[add_layer_norm](../../norm/add_layer_norm/README.md)	|		AI Core	|	实现AddLayerNorm功能。	|
|	norm	|	[add_layer_norm_quant](../../norm/add_layer_norm_quant/README.md)	|		AI Core	|	LayerNorm下游的量化算子：Quantize、AscendQuantV2或DynamicQuant算子。	|
|	norm	|	[add_rms_norm](../../norm/add_rms_norm/README.md)	|		AI Core	|	将RmsNorm前的Add算子融合起来，减少搬入搬出操作。	|
|	norm	|	[add_rms_norm_cast](../../norm/add_rms_norm_cast/README.md)	|		AI Core	|	将AddRmsNorm后的Cast算子融合起来，减少搬入搬出操作。	|
|	norm	|	[add_rms_norm_dynamic_quant](../../norm/add_rms_norm_dynamic_quant/README.md)	|		AI Core	|	将RmsNorm前的Add算子和RmsNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。	|
|	norm	|	[add_rms_norm_dynamic_quant_v2](../../norm/add_rms_norm_dynamic_quant_v2/README.md)	|		AI Core	|	将RmsNorm前的Add算子和RmsNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。	|
|	norm	|	[add_rms_norm_quant](../../norm/add_rms_norm_quant/README.md)	|		AI Core	|	将RmsNorm前的Add算子以及RmsNorm归一化的输出给到1个或2个Quantize算子融合起来，减少搬入搬出操作。	|
|	norm	|	[batch_norm_v3](../../norm/batch_norm_v3/README.md)	|		AI Core	|	对一个批次的数据做正则化处理，正则化之后生成的数据的统计结果为0均值、1标准差。	|
|	norm	|	[deep_norm](../../norm/deep_norm/README.md)	|		AI Core	|	对输入张量x的元素进行深度归一化，通过计算其均值和标准差，将每个元素标准化为具有零均值和单位方差的输出张量。	|
|	norm	|	[deep_norm_grad](../../norm/deep_norm_grad/README.md)	|		AI Core	|	deep_norm的反向传播，完成张量x、张量gx、张量gamma的梯度计算，以及张量dy的求和计算。	|
|	norm	|	[dua_quantize_add_layer_norm](../../norm/dua_quantize_add_layer_norm/README.md)	|		AI Core	|	DuaQuantizeAddLayerNorm结合了量化、加法和层归一化（Layer Normalization）操作。	|
|	norm	|	[gemma_rms_norm](../../norm/gemma_rms_norm/README.md)	|		AI Core	|	GemmaRmsNorm归一化操作，相比RmsNorm算子，在计算时对gamma做了+1操作。	|
|	norm	|	[group_norm_grad](../../norm/group_norm_grad/README.md)	|		AI Core	|		group_norm的反向传播。	|
|	norm	|	[group_norm_silu](../../norm/group_norm_silu/README.md)	|		AI Core	|	计算输入x的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及silu的输出。	|
|	norm	|	[group_norm_swish](../../norm/group_norm_swish/README.md)	|		AI Core	|	计算输入x的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及swish的输出。	|
|	norm	|	[group_norm_swish_grad](../../norm/group_norm_swish_grad/README.md)	|		AI Core	|	group_norm_swish的反向传播。	|
|	norm	|	[inplace_add_layer_norm](../../norm/inplace_add_layer_norm/README.md)	|		AI Core	|	一种结合了原位加法和层归一化的算法。	|
|	norm	|	[inplace_add_rms_norm](../../norm/inplace_add_rms_norm/README.md)	|		AI Core	|	一种结合了原位加法和RMS归一化的操作。	|
|	norm	|	[kv_rms_norm_rope_cache](../../norm/kv_rms_norm_rope_cache/README.md)	|		AI Core	|	对输入张量(kv)的尾轴，拆分出左半边用于rms_norm计算，右半边用于rope计算，再将计算结果分别scatter到两块cache中。	|
|	norm	|	[layer_norm_grad_v3](../../norm/layer_norm_grad_v3/README.md)	|		AI Core	|	layer_norm_v4的反向传播。  	|
|	norm	|	[layer_norm_v4](../../norm/layer_norm_v4/README.md)	|		AI Core	|	对指定层进行均值为0、标准差为1的归一化计算。	|
|	norm	|	[masked_softmax_with_rel_pos_bias](../../norm/masked_softmax_with_rel_pos_bias/README.md)	|		AI Core	|	替换在swinTransformer中使用window attention计算softmax的部分。	|
|	norm	|	[quantized_batch_norm](../../norm/quantized_batch_norm/README.md)	|		AI Core	|	将输入Tensor做一个反量化的计算，再根据输入的weight、bias、epsilon做归一化，最后根据输出的outputScale以及outputZeroPoint做量化。	|
|	norm	|	[quantize_add_layer_norm](../../norm/quantize_add_layer_norm/README.md)	|		AI Core	|	结合了量化、加法和层归一化的操作。	|
|	norm	|	[rms_norm](../../norm/rms_norm/README.md)	|		AI Core	|	RmsNorm归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。	|
|	norm	|	[rms_norm_grad](../../norm/rms_norm_grad/README.md)	|		AI Core	|	rms_norm的反向计算。用于计算RMSNorm的梯度，即在反向传播过程中计算输入张量的梯度。	|
|	norm	|	[rms_norm_quant](../../norm/rms_norm_quant/README.md)	|		AI Core	| 将RmsNorm算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。	|
|	optim	|	[advance_step](../../optim/advance_step/README.md)	|		AI Core	|	推进推理步骤，在每个生成步骤中更新模型的状态并生成新的inputTokens、inputPositions、seqLens和slotMapping，为vLLM的推理提升效率。	|
|	optim	|	[apply_fused_ema_adam](../../optim/apply_fused_ema_adam/README.md)	|		AI Core	|	实现FusedEmaAdam融合优化器功能，结合了Adam优化器和指数移动平均（EMA）技术。	|
|	pooling	|	[adaptive_avg_pool3d](../../pooling/adaptive_avg_pool3d/README.md)	|		AI Core	|	在指定三维输出shape信息的情况下，完成输入张量的3D自适应平均池化计算。	|
|	pooling	|	[adaptive_avg_pool3d_grad](../../pooling/adaptive_avg_pool3d_grad/README.md)	|		AI Core	|	adaptive_avg_pool3d的反向传播。	|
|	pooling	|	[adaptive_max_pool3d](../../pooling/adaptive_max_pool3d/README.md)	|		AI Core	|	根据输入的outputSize计算每次kernel的大小，对输入self进行3维最大池化操作，输出池化后的值outputOut和索引indicesOut。	|
|	pooling	|	[adaptive_max_pool3d_grad](../../pooling/adaptive_max_pool3d_grad/README.md)	|		AI Core	|	adaptive_max_pool3d的反向传播，将梯度回填到每个自适应窗口最大值的坐标处，相同坐标处累加。	|
|	pooling	|	[avg_pool3_d](../../pooling/avg_pool3_d/README.md)	|		AI Core	|	对输入Tensor进行窗口为$kD * kH * kW$、步长为$sD * sH * sW$的三维平均池化操作，其中$k$为kernelSize，表示池化窗口的大小，$s$为stride，表示池化操作的步长。	|
|	pooling	|	[avg_pool3_d_grad](../../pooling/avg_pool3_d_grad/README.md)	|		AI Core	|	avg_pool3_d的反向传播，计算三维平均池化正向传播的输入梯度。	|
|	pooling	|	[max_pool3d_grad_with_argmax](../../pooling/max_pool3d_grad_with_argmax/README.md)	|		AI Core	|	正向最大池化的反向传播，将梯度回填到每个窗口最大值的坐标处，相同坐标处累加。	|
|	pooling	|	[max_pool3d_with_argmax_v2](../../pooling/max_pool3d_with_argmax_v2/README.md)	|		AI Core	|	对于输入信号的输入通道，提供3维最大池化（Max pooling）操作，输出池化后的值out和索引indices。	|
|	quant	|	[ascend_quant_v2](../../quant/ascend_quant_v2/README.md)	|		AI Core	|	对输入x进行量化操作，支持设置axis以指定scale和offset对应的轴，scale和offset的shape需要满足和axis指定x的轴相等或1。axis当前支持设置最后两个维度。	|
|	quant	|	[dequant_bias](../../quant/dequant_bias/README.md)	|		AI Core	|	对输入x反量化操作，将输入的int32的数据转化为FLOAT16/BFLOAT16输出。	|
|	quant	|	[dequant_swiglu_quant](../../quant/dequant_swiglu_quant/README.md)	|		AI Core	|	在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。  	|
|	quant	|	[dynamic_quant](../../quant/dynamic_quant/README.md)	|		AI Core	|	为输入张量进行per-token对称动态量化。	|
|	quant	|	[dynamic_quant_update_scatter](../../quant/dynamic_quant_update_scatter/README.md)	|		AI Core	|	将DynamicQuantV2和ScatterUpdate单算子自动融合为DynamicQuantUpdateScatterV2融合算子，以实现INT4类型的非对称量化。	|
|	quant	|	[dynamic_quant_update_scatter_v2](../../quant/dynamic_quant_update_scatter_v2/README.md)	|		AI Core	|	将DynamicQuantV2和ScatterUpdate单算子自动融合为DynamicQuantUpdateScatterV2融合算子，以实现INT4类型的非对称量化。	|
|	quant	|	[dynamic_quant_v2](../../quant/dynamic_quant_v2/README.md)	|		AI Core	|	为输入张量进行per-token对称/非对称动态量化。	|
|	quant	|	[fake_quant_affine_cachemask](../../quant/fake_quant_affine_cachemask/README.md)	|		AI Core	|	对于输入数据self，使用scale和zero_point对输入self在指定轴axis上进行伪量化处理，并根据quant_min和quant_max对伪量化输出进行值域更新，最终返回结果out及对应位置掩码mask。	|
|	quant	|	[flat_quant](../../quant/flat_quant/README.md)	|		AI Core	|	该融合算子为输入矩阵x一次进行两次小矩阵乘法，即右乘输入矩阵kroneckerP2，左乘输入矩阵kroneckerP1，然后针对矩阵乘的结果进行per-token量化处理。	|
|	quant	|	[group_quant](../../quant/group_quant/README.md)	|		AI Core	|	对输入x进行分组量化操作。	|
|	quant	|	[swi_glu_quant](../../quant/swi_glu_quant/README.md)	|		AI Core	|	在SwiGlu激活函数后添加quant操作，实现输入x的SwiGluQuant计算，支持int8或int4量化输出。	|
|	quant	|	[trans_quant_param_v2](../../quant/trans_quant_param_v2/README.md)	|		AI Core	|	完成量化计算参数scale数据类型的转换，将Float32的数据类型转换为硬件需要的UINT64类型。	|
|	rnn	|	[dynamic_rnn](../../rnn/dynamic_rnn/README.md)	|		AI Core	|	基础循环神经网络 (Recurrent Neural Network) 算子，用于处理序列数据。它通过隐藏状态传递时序信息，适合处理具有时间/顺序依赖性的数据。	|
|	rnn	|	[dynamic_rnnv2](../../rnn/dynamic_rnnv2/README.md)	|		AI Core	|	基础循环神经网络 (Recurrent Neural Network) 算子，用于处理序列数据。它通过隐藏状态传递时序信息，适合处理具有时间/顺序依赖性的数据， 仅支持单层RNN。	|
|	vfusion	|	[multi_scale_deformable_attn_function](../../vfusion/multi_scale_deformable_attn_function/README.md)	|		AI Core	|	通过采样位置（sample location）、注意力权重（attention weights）、映射后的value特征、多尺度特征起始索引位置、多尺度特征图的空间大小（便于将采样位置由归一化的值变成绝对位置）等参数来遍历不同尺寸特征图的不同采样点。	|
|	vfusion	|	[multi_scale_deformable_attention_grad](../../vfusion/multi_scale_deformable_attention_grad/README.md)	|		AI Core	|	multi_scale_deformable_attn_function的反向传播。	|
|	vfusion	|	[scaled_masked_softmax_grad_v2](../../vfusion/scaled_masked_softmax_grad_v2/README.md)	|		AI Core	|	scaled_masked_softmax_v2的反向传播，并对结果进行缩放以及掩码。	|
|	vfusion	|	[scaled_masked_softmax_v2](../../vfusion/scaled_masked_softmax_v2/README.md)	|		AI Core	|	将输入的数据x先进行scale缩放和mask，然后执行softmax的输出。	|
|	activation	|	[celu_v2](../../activation/celu_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[elu](../../activation/elu/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[elu_grad_v2](../../activation/elu_grad_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[erfinv](../../activation/erfinv/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[fast_gelu](../../activation/fast_gelu/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[fast_gelu_grad](../../activation/fast_gelu_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[gelu](../../activation/gelu/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[gelu_grad](../../activation/gelu_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[gelu_grad_v2](../../activation/gelu_grad_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[gelu_v2](../../activation/gelu_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[hardtanh_grad](../../activation/hardtanh_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[hard_shrink](../../activation/hard_shrink/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[hard_shrink_grad](../../activation/hard_shrink_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[hard_sigmoid](../../activation/hard_sigmoid/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[hard_sigmoid_grad](../../activation/hard_sigmoid_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[hard_swish](../../activation/hard_swish/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[hard_swish_grad](../../activation/hard_swish_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[leaky_relu](../../activation/leaky_relu/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[leaky_relu_grad](../../activation/leaky_relu_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[logsigmoid](../../activation/logsigmoid/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[logsigmoid_grad](../../activation/logsigmoid_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[logsoftmax_grad](../../activation/logsoftmax_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[logsoftmax_v2](../../activation/logsoftmax_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[mish](../../activation/mish/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[mish_grad](../../activation/mish_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[prelu](../../activation/prelu/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[prelu_grad_reduce](../../activation/prelu_grad_reduce/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[prelu_grad_update](../../activation/prelu_grad_update/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[relu](../../activation/relu/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[relu_grad](../../activation/relu_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[selu](../../activation/selu/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[selu_grad](../../activation/selu_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[shrink](../../activation/shrink/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[sigmoid](../../activation/sigmoid/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[sigmoid_grad](../../activation/sigmoid_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[silu_grad](../../activation/silu_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[softmax_grad](../../activation/softmax_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[softmax_v2](../../activation/softmax_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[softplus_v2](../../activation/softplus_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[softplus_v2_grad](../../activation/softplus_v2_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[softshrink](../../activation/softshrink/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[softshrink_grad](../../activation/softshrink_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[swish](../../activation/swish/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[swish_grad](../../activation/swish_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[threshold](../../activation/threshold/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	activation	|	[threshold_grad_v2_d](../../activation/threshold_grad_v2_d/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	control	|	[identity](../../control/identity/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	control	|	[identity_n](../../control/identity_n/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	control	|	[rank](../../control/rank/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	control	|	[shape](../../control/shape/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	control	|	[shape_n](../../control/shape_n/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[gather_elements](../../index/gather_elements/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[gather_nd](../../index/gather_nd/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[gather_v2](../../index/gather_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[index](../../index/index/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[index_fill_d](../../index/index_fill_d/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[index_put_v2](../../index/index_put_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[matrix_inverse](../../index/matrix_inverse/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[non_zero](../../index/non_zero/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[repeat_interleave](../../index/repeat_interleave/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[reverse_v2](../../index/reverse_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[scatter_elements](../../index/scatter_elements/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[scatter_nd_update](../../index/scatter_nd_update/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[scatter_update](../../index/scatter_update/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[unique_consecutive](../../index/unique_consecutive/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	index	|	[unique_with_counts_ext2](../../index/unique_with_counts_ext2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[binary_cross_entropy](../../loss/binary_cross_entropy/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[binary_cross_entropy_grad](../../loss/binary_cross_entropy_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[ctc_loss_v2](../../loss/ctc_loss_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[ctc_loss_v2_grad](../../loss/ctc_loss_v2_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[kl_div_loss_grad](../../loss/kl_div_loss_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[l1_loss_grad](../../loss/l1_loss_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[lp_loss](../../loss/lp_loss/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[mse_loss](../../loss/mse_loss/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[multilabel_margin_loss](../../loss/multilabel_margin_loss/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[nll_loss](../../loss/nll_loss/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[nll_loss_grad](../../loss/nll_loss_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[sigmoid_cross_entropy_with_logits_grad_v2](../../loss/sigmoid_cross_entropy_with_logits_grad_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[sigmoid_cross_entropy_with_logits_v2](../../loss/sigmoid_cross_entropy_with_logits_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[smooth_l1_loss_grad_v2](../../loss/smooth_l1_loss_grad_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[smooth_l1_loss_v2](../../loss/smooth_l1_loss_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[soft_margin_loss](../../loss/soft_margin_loss/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	loss	|	[soft_margin_loss_grad](../../loss/soft_margin_loss_grad/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	matmul	|	[addmv](../../matmul/addmv/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	matmul	|	[gemm](../../matmul/gemm/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	matmul	|	[mv](../../matmul/mv/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	norm	|	[batch_norm_elemt](../../norm/batch_norm_elemt/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	norm	|	[batch_norm_grad_v3](../../norm/batch_norm_grad_v3/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	norm	|	[group_norm](../../norm/group_norm/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	norm	|	[instance_norm_v3](../../norm/instance_norm_v3/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	norm	|	[lp_norm_v2](../../norm/lp_norm_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	norm	|	[renorm](../../norm/renorm/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	norm	|	[sync_batch_norm_backward_elemt](../../norm/sync_batch_norm_backward_elemt/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	norm	|	[sync_batch_norm_backward_reduce](../../norm/sync_batch_norm_backward_reduce/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	norm	|	[sync_batch_norm_gather_stats_with_counts](../../norm/sync_batch_norm_gather_stats_with_counts/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	quant	|	[ascend_anti_quant_v2](../../quant/ascend_anti_quant_v2/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|
|	quant	|	[quantize](../../quant/quantize/README.md)	|		AI Core	|	该算子暂无AscendC代码实现，欢迎各位开发者补充贡献，贡献方式可以参考[贡献指南](../../CONTRIBUTING.md)。	|