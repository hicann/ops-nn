# 算子接口（aclnn）

## 使用说明

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

调用算子API时，需引用依赖的头文件和库文件，一般头文件默认在```${INSTALL_DIR}/include/aclnnop```，库文件默认在```${INSTALL_DIR}/lib64```，具体文件如下：

- 依赖的头文件：①方式1 （推荐）：引用算子总头文件aclnn\_ops\_\$\{ops\_project\}.h。②方式2：按需引用单算子API头文件aclnn\_\*.h。
- 依赖的库文件：按需引用算子总库文件libopapi\_\$\{ops\_project\}.so。

其中${INSTALL_DIR}表示CANN安装后文件路径；\$\{ops\_project\}表示算子仓（如math、nn、cv、transformer），请配置为实际算子仓名。

## 接口列表

> **确定性简介**：
>
> - 配置说明：因CANN或NPU型号不同等原因，可能无法保证同一个算子多次运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分算子接口可通过`aclrtCtxSetSysParamOpt`（参见[《acl API（C）》](https://hiascend.com/document/redirect/CannCommunityCppApi)）开启确定性算法，使多次运行结果一致。
> - 性能说明：同一个算子采用确定性计算通常比非确定性慢，因此模型单次运行性能可能会下降。但在实验、调试调测等需要保证多次运行结果相同来定位问题的场景，确定性计算可以提升效率。
> - 线程说明：同一线程中只能设置一次确定性状态，多次设置以最后一次有效设置为准。有效设置是指设置确定性状态后，真正执行了一次算子任务下发。如果仅设置，没有算子下发，只能是确定性变量开启但未下发给算子，因此不执行算子。
>   解决方案：暂不推荐一个线程多次设置确定性。该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。

算子接口列表如下：

|    接口名   |      说明     |    确定性说明（A2/A3）    |
|-----------|------------|------------|
| [aclnnAdaLayerNorm](../../norm/ada_layer_norm/docs/aclnnAdaLayerNorm.md)|将LayerNorm和下游的Add、Mul融合起来，通过自适应参数scale和shift来调整归一化过程。|默认确定性实现|
| [aclnnAdaLayerNormQuant](../../norm/ada_layer_norm_quant/docs/aclnnAdaLayerNormQuant.md)|算子将AdaLayerNorm和下游的量化（目前仅支持DynamicQuant）融合起来。该算子主要是用于执行自适应层归一化的量化操作，即将输入数据进行归一化处理，并将其量化为低精度整数，以提高计算效率和减少内存占用。|默认确定性实现|
| [aclnnAdaLayerNormV2](../../norm/ada_layer_norm_v2/docs/aclnnAdaLayerNormV2.md)|算子将LayerNorm和下游的Add、Mul融合起来，通过自适应参数scale和shift来调整归一化过程。相比AdaLayerNorm算子，输出新增2个参数（输入的均值和输入的标准差的倒数）；weight和bias支持的数据类型增加对应约束。|默认确定性实现|
| [aclnnAdaptiveAvgPool2d](../../pooling/adaptive_avg_pool3d/docs/aclnnAdaptiveAvgPool2d.md)|在指定二维输出shape信息（outputSize）的情况下，完成张量self的2D自适应平均池化计算。|默认确定性实现|
| [aclnnAdaptiveAvgPool2dBackward](../../pooling/adaptive_avg_pool3d_grad/docs/aclnnAdaptiveAvgPool2dBackward.md)|[aclnnAdaptiveAvgPool2d](../../pooling/adaptive_avg_pool3d/docs/aclnnAdaptiveAvgPool2d.md) 的反向计算。|默认非确定性实现，支持配置开启|
| [aclnnAdaptiveAvgPool3d](../../pooling/adaptive_avg_pool3d/docs/aclnnAdaptiveAvgPool3d.md)|在指定三维输出shape信息（outputSize）的情况下，完成张量self的3D自适应平均池化计算。|默认确定性实现|
| [aclnnAdaptiveAvgPool3dBackward]()|[aclnnAdaptiveAvgPool3d](../../pooling/adaptive_avg_pool3d/docs/aclnnAdaptiveAvgPool3d.md)的反向计算。|默认非确定性实现，支持配置开启|
| [aclnnAdaptiveMaxPool2d](../../pooling/adaptive_max_pool3d/docs/aclnnAdaptiveMaxPool2d.md)|根据输入的outputSize计算每次kernel的大小，对输入self进行2维最大池化操作。|默认确定性实现|
| [aclnnAdaptiveMaxPool3d](../../pooling/adaptive_max_pool3d/docs/aclnnAdaptiveMaxPool3d.md)|根据输入的outputSize计算每次kernel的大小，对输入self进行3维最大池化操作。|默认确定性实现|
| [aclnnAdaptiveMaxPool3dBackward](../../pooling/adaptive_max_pool3d_grad/docs/aclnnAdaptiveMaxPool3dBackward.md)|正向自适应最大池化的反向传播，将梯度回填到每个自适应窗口最大值的坐标处，相同坐标处累加。|默认非确定性实现，支持配置开启|
| [aclnnAddbmm&aclnnInplaceAddbmm](../../matmul/batch_mat_mul_v3/docs/aclnnAddbmm&aclnnInplaceAddbmm.md) |首先进行batch1、batch3的矩阵乘计算，然后将该结果按照第一维（batch维度）批处理相加，将三维向量压缩为二维向量（shape大小为后两维的shape），然后该结果与α作乘积计算，再与β和self的乘积求和得到结果。|默认确定性实现|
| [aclnnAddmm](../../matmul/mat_mul_v3/docs/aclnnAddmm&aclnnInplaceAddmm.md) |计算α 乘以mat1与mat2的乘积，再与β和self的乘积求和。|默认非确定性实现，支持配置开启。|
| [aclnnAddmmWeightNz](../../matmul/mat_mul_v3/docs/aclnnAddmmWeightNz.md)|计算α 乘以mat1与mat2的乘积，再与β和self的乘积求和。相较于原有addmm接口，新接口mat2支持nz格式。|默认确定性实现|
| [aclnnAddmv](../../matmul/addmv/docs/aclnnAddmv.md) | 完成矩阵乘计算，然后和向量相加。|默认非确定性实现，支持配置开启。|
| [aclnnAddLayerNorm](../../norm/add_layer_norm/docs/aclnnAddLayerNorm.md)|实现AddLayerNorm功能。|默认确定性实现|
| [aclnnAddLayerNormGrad](../../norm/add_layer_norm_grad/docs/aclnnAddLayerNormGrad.md)|LayerNorm是一种归一化方法，可以将网络层输入数据归一化到[0, 1]之间。|默认非确定性实现，支持配置开启|
| [aclnnAddLayerNormQuant](../../norm/add_layer_norm_quant/docs/aclnnAddLayerNormQuant.md)|LayerNorm算子是大模型常用的归一化操作。|默认确定性实现|
| [aclnnAddRmsNormCast](../../norm/add_rms_norm_cast/docs/aclnnAddRmsNormCast.md)|RmsNorm算子是大模型常用的归一化操作，AddRmsNormCast算子将AddRmsNorm后的Cast算子融合起来，减少搬入搬出操作。|默认确定性实现|
| [aclnnAddRmsNormDynamicQuant](../../norm/add_rms_norm_dynamic_quant/docs/aclnnAddRmsNormDynamicQuant.md)|将RmsNorm前的Add算子和RmsNorm归一化输出给到的1个或2个DynamicQuant算子融合起来，减少搬入搬出操作。|默认确定性实现|
| [aclnnAddRmsNormDynamicQuantV2](../../norm/add_rms_norm_dynamic_quant/docs/aclnnAddRmsNormDynamicQuantV2.md)|RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。|默认确定性实现|
| [aclnnAddRmsNorm](../../norm/add_rms_norm/docs/aclnnAddRmsNorm.md)|RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。|默认确定性实现|
| [aclnnAddRmsNormQuant](../../norm/add_rms_norm_quant/docs/aclnnAddRmsNormQuant.md)|将RmsNorm前的Add算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。|默认确定性实现|
| [aclnnAddRmsNormQuantV2](../../norm/add_rms_norm_quant_v2/docs/aclnnAddRmsNormQuantV2.md)|RmsNorm是大模型常用的标准化操作，相比LayerNorm，其去掉了减去均值的部分。|默认确定性实现|
| [aclnnAdvanceStep](../../optim/advance_step/docs/aclnnAdvanceStep.md)|推进推理步骤，即在每个生成步骤中更新模型的状态并生成新的inputTokens、inputPositions、seqLens和slotMapping，为vLLM的推理提升效率。|默认确定性实现|
| [aclnnAdvanceStepV2](../../optim/advance_step/docs/aclnnAdvanceStepV2.md)|推进推理步骤，即在每个生成步骤中更新模型的状态并生成新的inputTokens、inputPositions、seqLens和slotMapping，为vLLM的推理提升效率。|默认确定性实现|
| [aclnnApplyAdamW](../../optim/apply_adam_w/docs/aclnnApplyAdamW.md)|实现adamW优化器功能。|默认确定性实现|
| [aclnnApplyAdamWV2](../../optim/apply_adam_w_v2/docs/aclnnApplyAdamWV2.md)|实现adamW优化器功能。|默认确定性实现|
| [aclnnApplyFusedEmaAdam](../../optim/apply_fused_ema_adam/docs/aclnnApplyFusedEmaAdam.md)|实现FusedEmaAdam融合优化器功能。|默认确定性实现|
| [aclnnApplyTopKTopP](../../index/apply_top_k_top_p_with_sorted/docs/aclnnApplyTopKTopP.md) |对原始输入logits进行top-k和top-p采样过滤。|默认确定性实现|
| [aclnnAscendAntiQuant](../../quant/ascend_anti_quant_v2/docs/aclnnAscendAntiQuant.md)|根据输入的sacle和offset对输入x进行反量化。|默认确定性实现|
| [aclnnAscendQuant](../../quant/ascend_quant_v2/docs/aclnnAscendQuant.md)|根据输入的sacle和offset对输入x进行量化，且scale和offset的size需要是x的最后一维或1。|默认确定性实现|
| [aclnnAscendQuantV3](../../quant/ascend_quant_v2/docs/aclnnAscendQuantV3.md)|对输入x进行量化操作，支持设置axis以指定scale和offset对应的轴，scale和offset的shape需要满足和axis指定x的轴相等或1。|默认确定性实现|
| [aclnnAvgPool2d](../../pooling/avg_pool3_d/docs/aclnnAvgPool2d.md)|对输入Tensor进行窗口为$kH * kW$、步长为$sH * sW$的二维平均池化操作。|默认确定性实现|
| [aclnnAvgPool2dBackward](../../pooling/avg_pool3_d_grad/docs/aclnnAvgPool2dBackward.md)|二维平均池化的反向传播，计算二维平均池化正向传播的输入梯度。|默认非确定性实现，支持配置开启|
| [aclnnAvgPool3d](../../pooling/avg_pool3_d/docs/aclnnAvgPool3d.md)|对输入Tensor进行窗口为$kD * kH * kW$、步长为$sD * sH * sW$的三维平均池化操作。|默认确定性实现|
| [aclnnAvgPool3dBackward](../../pooling/avg_pool3_d_grad/docs/aclnnAvgPool3dBackward.md)|三维平均池化的反向传播，计算三维平均池化正向传播的输入梯度。|默认非确定性实现，支持配置开启|
| [aclnnBaddbmm&aclnnInplaceBaddbmm](../../matmul/batch_mat_mul_v3/docs/aclnnBaddbmm&aclnnInplaceBaddbmm.md) |计算α与batch1、batch2的矩阵乘结果的乘积，再与β和self的乘积求和。|默认确定性实现|
| [aclnnBatchMatMul](../../matmul/batch_mat_mul_v3/docs/aclnnBatchMatMul.md) |完成张量self与张量mat2的矩阵乘计算。|默认确定性实现|
| [aclnnBatchMatmulQuant](../../matmul/batch_matmul_quant/docs/aclnnBatchMatmulQuant.md)|实现输入Tensor的dtype是float16, 输出的dtype是int8的矩阵乘计算。|默认确定性实现|
| [aclnnBatchMatMulWeightNz](../../matmul/batch_mat_mul_v3/docs/aclnnBatchMatMulWeightNz.md) |完成张量self与张量mat2的矩阵乘计算, mat2仅支持昇腾亲和数据排布格式，只支持self为3维, mat2为5维。|默认确定性实现|
| [aclnnBatchNorm](../../norm/batch_norm_v3/docs/aclnnBatchNorm.md)|对一个批次的数据做批量归一化处理，正则化之后生成的数据的统计结果为0均值、1标准差。|默认非确定性实现，支持配置开启|
| [aclnnBatchNormElemt](../../norm/batch_norm_elemt/docs/aclnnBatchNormElemt.md)|将全局的均值和标准差倒数作为算子输入，对x做BatchNorm计算。|默认非确定性实现，支持配置开启|
| [aclnnBatchNormElemtBackward](../../norm/sync_batch_norm_backward_elemt/docs/aclnnBatchNormElemtBackward.md)|aclnnBatchNormElemt的反向计算。用于计算输入张量的元素级梯度，以便在反向传播过程中更新模型参数。|默认确定性实现|
| [aclnnBatchNormBackward](../../norm/batch_norm_grad_v3/docs/aclnnBatchNormBackward.md)|[aclnnBatchNorm](../../norm/batch_norm_v3/docs/aclnnBatchNorm.md)的反向传播。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。|默认非确定性实现，支持配置开启|
| [aclnnBatchNormGatherStatsWithCounts](../../norm/sync_batch_norm_gather_stats_with_counts/docs/aclnnBatchNormGatherStatsWithCounts.md)|收集所有device的均值和方差，更新全局的均值和标准差的倒数。|默认确定性实现|
| [aclnnBatchNormReduce](../../norm/bn_training_reduce/docs/aclnnBatchNormReduce.md)|对数据做正则化处理的第一步，对数据进行求和及平方和|默认非确定性实现，支持配置开启|
| [aclnnBatchNormReduceBackward](../../norm/sync_batch_norm_backward_reduce/docs/aclnnBatchNormReduceBackward.md)|主要用于反向传播过程中计算BatchNorm操作的梯度，并进行一些中间结果的规约操作以优化计算效率。|默认确定性实现|
| [aclnnBidirectionLSTM](../../rnn/bidirection_lstm/docs/aclnnBidirectionLSTM.md) | 进行LSTM网络计算，接收输入序列和初始状态，返回输出序列和最终状态。|默认确定性实现|
| [aclnnBidirectionLSTMV2](../../rnn/bidirection_lstm_v2/docs/aclnnBidirectionLSTMV2.md) | 进行LSTM网络计算，接收输入序列和初始状态，返回输出序列和最终状态。|默认确定性实现|
| [aclnnBinaryCrossEntropy](../../loss/binary_cross_entropy/docs/aclnnBinaryCrossEntropy.md) | 计算self和target的二元交叉熵。|默认确定性实现|
| [aclnnBinaryCrossEntropyBackward](../../loss/binary_cross_entropy_grad/docs/aclnnBinaryCrossEntropyBackward.md) | 求二元交叉熵反向传播的梯度值。|默认确定性实现|
| [aclnnBinaryCrossEntropyWithLogits](../../loss/sigmoid_cross_entropy_with_logits_v2/docs/aclnnBinaryCrossEntropyWithLogits.md) |计算输入logits与标签target之间的BCELoss损失。|默认确定性实现|
| [aclnnBinaryCrossEntropyWithLogitsTargetBackward](../../activation/logsigmoid/docs/aclnnBinaryCrossEntropyWithLogitsTargetBackward.md) |将输入self执行logits计算，将得到的值与标签值target一起进行BECLoss关于target的反向传播计算。|默认确定性实现|
| [aclnnCelu&aclnnInplaceCelu](../../activation/celu_v2/docs/aclnnCelu&aclnnInplaceCelu.md) |aclnnCelu对输入张量self中的每个元素x调用连续可微指数线性单元激活函数CELU，并将得到的结果存入输出张量out中。|默认确定性实现|
| [aclnnChamferDistanceBackward](../../loss/chamfer_distance_grad/docs/aclnnChamferDistanceBackward.md) | ChamferDistance（倒角距离）的反向算子，根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。|	默认非确定性实现，支持配置开启|
| [aclnnConvolution](../../conv/convolution_forward/docs/aclnnConvolution.md) |实现卷积功能，支持1D/2D/3D、转置卷积、空洞卷积、分组卷积。|默认确定性实现|
| [aclnnConvolutionBackward](../../conv/convolution_backward/docs/aclnnConvolutionBackward.md) |实现卷积的反向传播。|默认非确定性实现，支持配置开启。|
| [aclnnConvDepthwise2d](../../conv/convolution_forward/docs/aclnnConvDepthwise2d.md) |实现二维深度卷积（DepthwiseConv2D）计算。|默认确定性实现|
| [aclnnConvertWeightToINT4Pack](../../matmul/convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md)|对输入weight数据做预处理，实现低比特数据由稀疏存储到紧密存储的排布转换。输出weightInt4Pack的[数据格式](../../docs/zh/context/数据格式.md)声明为FRACTAL_NZ时，该算子将[数据格式](../../docs/zh/context/数据格式.md)从ND转为FRACTAL_NZ。|默认确定性实现|
| [aclnnConvTbc](../../conv/convolution_forward/docs/aclnnConvTbc.md) |实现时序（TBC）一维卷积。|默认确定性实现|
| [aclnnConvTbcBackward](../../conv/convolution_backward/docs/aclnnConvTbcBackward.md) |用于计算时序卷积的反向传播。|默认确定性实现|
| [aclnnCrossEntropyLoss](../../loss/cross_entropy_loss/docs/aclnnCrossEntropyLoss.md) | 计算输入的交叉熵损失。|默认确定性实现|
| [aclnnCtcLoss](../../loss/ctc_loss_v2/docs/aclnnCtcLoss.md) | 计算连接时序分类损失值。|默认非确定性实现，支持配置开启。|
| [aclnnCtcLossBackward](../../loss/ctc_loss_v2_grad/docs/aclnnCtcLossBackward.md) | 连接时序分类损失值反向传播。|默认非确定性实现，支持配置开启。|
| [aclnnDeepNorm](../../norm/deep_norm/docs/aclnnDeepNorm.md)|对输入张量x的元素进行深度归一化，通过计算其均值和标准差，将每个元素标准化为具有零均值和单位方差的输出张量。|默认确定性实现|
| [aclnnDeepNormGrad](../../norm/deep_norm_grad/docs/aclnnDeepNormGrad.md)|[aclnnDeepNorm](../../norm/deep_norm/docs/aclnnDeepNorm.md)的反向传播，完成张量x、张量gx、张量gamma的梯度计算，以及张量dy的求和计算。|默认非确定性实现，不支持配置开启|
| [aclnnDeformableConv2d](../../conv/deformable_conv2d/docs/aclnnDeformableConv2d.md)|实现卷积功能，支持2D卷积，同时支持可变形卷积、分组卷积。|默认确定性实现|
| [aclnnDequantBias](../../quant/dequant_bias/docs/aclnnDequantBias.md)|对输入x反量化操作，将输入的int32的数据转化为FLOAT16/BFLOAT16输出。|默认确定性实现|
| [aclnnDequantSwigluQuant](../../quant/dequant_swiglu_quant/docs/aclnnDequantSwigluQuant.md)|在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。|默认确定性实现|
| [aclnnDequantSwigluQuantV2](../../quant/dequant_swiglu_quant/docs/aclnnDequantSwigluQuantV2.md)|在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。|默认确定性实现|
| [aclnnGroupedDynamicMxQuant](../../quant/grouped_dynamic_mx_quant/docs/aclnnGroupedDynamicMxQuant.md)|根据传入的分组索引的起始值，对传入的数据进行分组的float8的动态量化。|-|
| [aclnnDynamicDualLevelMxQuant](../../quant/dynamic_dual_level_mx_quant/docs/aclnnDynamicDualLevelMxQuant.md)|通过给定的level0BlockSize和level1BlockSize将输入进行两次划分，以数据块为粒度，进行目的数据类型为FLOAT4类的MX二级量化。| - |
| [aclnnGroupedDynamicBlockQuant](../../quant/grouped_dynamic_block_quant/docs/aclnnGroupedDynamicBlockQuant.md)|根据传入的分组索引的起始值对各个group以基本块的粒度进行量化，并输出量化参数scale。| - |
| [aclnnDynamicMxQuant](../../quant/dynamic_mx_quant/docs/aclnnDynamicMxQuant.md)|目的数据类型为FLOAT4类、FLOAT8类的MX量化。在给定的轴axis上，根据每blocksize个数，计算出这组数对应的量化尺度mxscale作为输出mxscaleOut的对应部分，然后对这组数每一个除以mxscale，根据round_mode转换到对应的dstType，得到量化结果y作为输出yOut的对应部分。在dstType为FLOAT8_E4M3FN、FLOAT8_E5M2时，根据scaleAlg的取值来指定计算mxscale的不同算法。|-|
| [aclnnDynamicBlockQuant](../../quant/dynamic_block_quant/docs/aclnnDynamicBlockQuant.md)|对输入张量，通过给定的rowBlockSize和colBlockSize将输入划分成多个数据块，以数据块为基本粒度进行量化。在每个块中，先计算出当前块对应的量化参数scaleOut，并根据scaleOut对输入进行量化。输出最终的量化结果，以及每个块的量化参数scaleOut。|默认确定性实现|
| [aclnnDynamicQuant](../../quant/dynamic_quant/docs/aclnnDynamicQuant.md)|对输入张量进行per-token对称动态量化。|默认确定性实现|
| [aclnnDynamicQuantV2](../../quant/dynamic_quant_v2/docs/aclnnDynamicQuantV2.md)|为输入张量进行per-token对称/非对称动态量化。|默认确定性实现|
| [aclnnDynamicQuantV3](../../quant/dynamic_quant/docs/aclnnDynamicQuantV3.md)|为输入张量进行动态量化。|默认确定性实现|
| [aclnnDynamicQuantUpdateScatter](../../quant/dynamic_quant_update_scatter/docs/aclnnDynamicQuantUpdateScatter.md)|将DynamicQuantV2和ScatterUpdate单算子自动融合为DynamicQuantUpdateScatterV2融合算子，以实现INT4类型的非对称量化。|默认确定性实现|
| [aclnnEinsum](../../matmul/batch_mat_mul_v3/docs/aclnnEinsum.md)|使用爱因斯坦求和约定执行张量计算，形式为“term1, term2 -> output-term”，按照以下等式生成输出张量，其中reduce-sum对出现在输入项(term1, term2)中但未出现在输出项中的所有索引执行求和。|默认确定性实现|
| [aclnnElu&aclnnInplaceElu](../../activation/elu/docs/aclnnElu&aclnnInplaceElu.md) |对输入张量self中的每个元素x调用指数线性单元激活函数ELU，并将得到的结果存入输出张量out中。|默认确定性实现|
| [aclnnEluBackward](../../activation/elu_grad_v2/docs/aclnnEluBackward.md) |aclnnElu激活函数的反向计算，输出ELU激活函数正向输入的梯度。|默认确定性实现|
| [aclnnEmbedding](../../index/gather_v2/docs/aclnnEmbedding.md) |把数据集合映射到向量空间，进而将数据进行量化。embedding的二维权重张量为weight(m+1行，n列)，对于任意输入索引张量indices（如1行3列），输出out是一个3行n列的张量。|默认确定性实现|
| [aclnnEmbeddingBag](../../index/embedding_bag/docs/aclnnEmbeddingBag.md) |根据indices从weight中获得一组被聚合的数，然后根据offsets的偏移和mode指定的聚合模式对获取的数进行max、sum、mean聚合。其余参数则更细化了计算过程的控制。|默认确定性实现|
| [aclnnEmbeddingDenseBackward](../../index/embedding_dense_grad_v2/docs/aclnnEmbeddingDenseBackward.md) |实现aclnnEmbedding的反向计算, 将相同索引indices对应grad的一行累加到out上。|默认非确定性实现，支持配置开启|
| [aclnnEmbeddingRenorm](../../index/gather_v2/docs/aclnnEmbeddingRenorm.md) |根据给定的maxNorm和normType返回输入tensor在指定indices下的修正结果。|默认确定性实现|
| [aclnnErfinv&aclnnInplaceErfinv](../../activation/erfinv/docs/aclnnErfinv&aclnnInplaceErfinv.md) |erfinv是高斯误差函数erf的反函数。返回输入Tensor中每个元素对应在标准正态分布函数的分位数。|默认确定性实现|
| [aclnnFakeQuantPerChannelAffineCachemask](../../quant/fake_quant_affine_cachemask/docs/aclnnFakeQuantPerChannelAffineCachemask.md)|对于输入数据self，使用scale和zero_point对输入self在指定轴axis上进行伪量化处理，并根据quant_min和quant_max对伪量化输出进行值域更新。|默认确定性实现|
| [aclnnFakeQuantPerTensorAffineCachemask](../../quant/fake_quant_affine_cachemask/docs/aclnnFakeQuantPerTensorAffineCachemask.md)|对输入self进行伪量化处理，并根据quant_min和quant_max对伪量化输出进行值域更新。|默认确定性实现|
| [aclnnFastBatchNormBackward](../../norm/batch_norm_grad_v3/docs/aclnnFastBatchNormBackward.md) |[aclnnBatchNorm](../../norm/batch_norm_v3/docs/aclnnBatchNorm.md)的反向传播（高性能版本）。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。|默认确定性实现|
| [aclnnFastGelu](../../activation/fast_gelu/docs/aclnnFastGelu.md) |快速高斯误差线性单元激活函数。|默认确定性实现|
| [aclnnFastGeluBackward](../../activation/fast_gelu_grad/docs/aclnnFastGeluBackward.md) |FastGelu的反向计算。|默认确定性实现|
| [aclnnFastLayerNorm](../../norm/layer_norm_v4/docs/aclnnFastLayerNorm.md) |对指定层进行均值为0、标准差为1的归一化计算。|默认确定性实现|
| [aclnnFatreluMul](../../activation/fatrelu_mul/docs/aclnnFatreluMul.md) |将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行Threshold计算，将计算结果与x2相乘。|默认确定性实现|
| [aclnnFlatQuant](../../quant/flat_quant/docs/aclnnFlatQuant.md)|融合算子为输入矩阵x一次进行两次小矩阵乘法。|默认确定性实现|
| [aclnnFlip](../../index/reverse_v2/docs/aclnnFlip.md) | 对n维张量的指定维度进行反转（倒序），dims中指定的每个轴的计算公式。|默认确定性实现|
| [aclnnForeachAbs](../../foreach/foreach_abs/docs/aclnnForeachAbs.md) | 对输入张量列表中的每个张量执行逐元素绝对值运算。 |默认确定性实现|
| [aclnnForeachAcos](../../foreach/foreach_acos/docs/aclnnForeachAcos.md) | 对输入张量列表中的每个张量执行逐元素反余弦运算。 |默认确定性实现|
| [aclnnForeachAddcdivList](../../foreach/foreach_addcdiv_list/docs/aclnnForeachAddcdivList.md) | 对多个张量进行逐元素加、乘、除操作，$x2_{i}$和$x3_{i}$进行逐元素相除，并将结果乘以scalars，再与$x1_{i}$相加。  |默认确定性实现|
| [aclnnForeachAddcdivScalar](../../foreach/foreach_addcdiv_scalar/docs/aclnnForeachAddcdivScalar.md) | 对多个张量进行逐元素加、乘、除操作，$x2_{i}$和$x3_{i}$进行逐元素相除，并将结果乘以scalar，再与$x1_{i}$相加。  |默认确定性实现|
| [aclnnForeachAddcdivScalarList](../../foreach/foreach_addcdiv_scalar_list/docs/aclnnForeachAddcdivScalarList.md) | 对多个张量进行逐元素加、乘、除操作，$x2_{i}$和$x3_{i}$进行逐元素相除，并将结果乘以$scalars_{i}$，再与$x1_{i}$相加。  |默认确定性实现|
| [aclnnForeachAddcdivScalarV2](../../foreach/foreach_addcdiv_scalar/docs/aclnnForeachAddcdivScalarV2.md) | 对多个张量进行逐元素加、乘、除操作，$x2_{i}$和$x3_{i}$进行逐元素相除，并将结果乘以scalar，再与$x1_{i}$相加。 |默认确定性实现|
| [aclnnForeachAddcmulList](../../foreach/foreach_addcmul_list/docs/aclnnForeachAddcmulList.md) | 先对张量列表x2和张量列表x3执行逐元素乘法，并将结果乘以张量scalars，最后将之前计算的结果与张量列表x1执行逐元素相加。 |默认确定性实现|
| [aclnnForeachAddcmulScalar](../../foreach/foreach_addcmul_scalar/docs/aclnnForeachAddcmulScalar.md) | 先对张量列表x2和张量列表x3执行逐元素乘法，再乘以张量scalar，最后将之前计算的结果与张量列表x1执行逐元素相加。 |默认确定性实现|
| [aclnnForeachAddcmulScalarList](../../foreach/foreach_addcmul_scalar_list/docs/aclnnForeachAddcmulScalarList.md) | 先对张量列表x2和张量列表x3执行逐元素乘法，再与张量scalars进行逐元素乘法，最后将之前计算的结果与张量列表x1执行逐元素相加。 |默认确定性实现|
| [aclnnForeachAddcmulScalarV2](../../foreach/foreach_addcmul_scalar/docs/aclnnForeachAddcmulScalarV2.md) | 先对张量列表x2和张量列表x3执行逐元素乘法，再乘以标量scalar，最后将之前计算的结果与张量列表x1执行逐元素相加。 |默认确定性实现|
| [aclnnForeachAddList](../../foreach/foreach_add_list/docs/aclnnForeachAddList.md) | 两个Tensor列表中的元素逐个相加，并可以通过alpha参数调整相加系数。 |默认确定性实现|
| [aclnnForeachAddListV2](../../foreach/foreach_add_list/docs/aclnnForeachAddListV2.md) | 两个Tensor列表中的元素逐个相加，并可以通过alpha参数调整相加系数。 |默认确定性实现|
| [aclnnForeachAddScalar](../../foreach/foreach_add_scalar/docs/aclnnForeachAddScalar.md) | 将指定的张量值加到张量列表中的每个张量中。 |默认确定性实现|
| [aclnnForeachAddScalarList](../../foreach/foreach_add_scalar_list/docs/aclnnForeachAddScalarList.md) | 输入张量列表和输入标量列表执行逐元素相加运算。 |默认确定性实现|
| [aclnnForeachAddScalarV2](../../foreach/foreach_add_scalar/docs/aclnnForeachAddScalarV2.md) | 将指定的标量值加到张量列表中的每个张量中。 |默认确定性实现|
| [aclnnForeachAsin](../../foreach/foreach_asin/docs/aclnnForeachAsin.md) | 按元素进行反正弦函数运算。  |默认确定性实现|
| [aclnnForeachAtan](../../foreach/foreach_atan/docs/aclnnForeachAtan.md) | 按元素进行反正切函数运算。  |默认确定性实现|
| [aclnnForeachCopy](../../foreach/foreach_copy/docs/aclnnForeachCopy.md) | 用于实现两个张量列表内容的复制。 |默认确定性实现|
| [aclnnForeachCos](../../foreach/foreach_cos/docs/aclnnForeachCos.md) | 按元素进行余弦函数运算。 |默认确定性实现|
| [aclnnForeachCosh](../../foreach/foreach_cosh/docs/aclnnForeachCosh.md) | 按元素进行双曲余弦函数运算。 |默认确定性实现|
| [aclnnForeachDivList](../../foreach/foreach_div_list/docs/aclnnForeachDivList.md) | 对张量列表x1和张量列表x2执行逐元素除法。 |默认确定性实现|
| [aclnnForeachDivScalar](../../foreach/foreach_div_scalar/docs/aclnnForeachDivScalar.md) | 计算张量列表x除以张量scalar。 |默认确定性实现|
| [aclnnForeachDivScalarList](../../foreach/foreach_div_scalar_list/docs/aclnnForeachDivScalarList.md) | 对张量列表x和标量列表scalars执行逐元素除法。 |默认确定性实现|
| [aclnnForeachDivScalarV2](../../foreach/foreach_div_scalar/docs/aclnnForeachDivScalarV2.md) | 计算张量列表x除以标量scalar。  |默认确定性实现|
| [aclnnForeachErf](../../foreach/foreach_erf/docs/aclnnForeachErf.md) | 按元素进行误差函数运算（也称之为高斯误差函数，error function or Gaussian error function）。 |默认确定性实现|
| [aclnnForeachErfc](../../foreach/foreach_erfc/docs/aclnnForeachErfc.md) | 按元素执行从x到无穷大积分的互补误差函数运算。 |默认确定性实现|
| [aclnnForeachExp](../../foreach/foreach_exp/docs/aclnnForeachExp.md) | 对输入张量列表的每个张量进行指数运算。 |默认确定性实现|
| [aclnnForeachExpm1](../../foreach/foreach_expm1/docs/aclnnForeachExpm1.md) | 对输入张量列表的每个张量执行指数运算，然后将得到的结果减1。 |默认确定性实现|
| [aclnnForeachLerpList](../../foreach/foreach_lerp_list/docs/aclnnForeachLerpList.md) | 对两个张量列表对应位置元素执行插值计算，其中张量列表weight是插值系数。 |默认确定性实现|
| [aclnnForeachLerpScalar](../../foreach/foreach_lerp_scalar/docs/aclnnForeachLerpScalar.md) | 对两个张量列表对应位置元素执行插值计算，其中标量weight是插值系数。 |默认确定性实现|
| [aclnnForeachLog](../../foreach/foreach_log/docs/aclnnForeachLog.md) | 对张量列表执行逐元素自然对数运算（ln(x)）。 |默认确定性实现|
| [aclnnForeachLog1p](../../foreach/foreach_log1p/docs/aclnnForeachLog1p.md) | 对张量列表中的每一个元素执行先加一再以e为底的对数函数运算。 |默认确定性实现|
| [aclnnForeachLog2](../../foreach/foreach_log2/docs/aclnnForeachLog2.md) | 对张量列表中的每一个元素执行以2为底的对数函数运算。 |默认确定性实现|
| [aclnnForeachLog10](../../foreach/foreach_log10/docs/aclnnForeachLog10.md) | 对张量列表中的每一个元素执行以10为底的对数函数运算。 |默认确定性实现|
| [aclnnForeachMaximumList](../../foreach/foreach_maximum_list/docs/aclnnForeachMaximumList.md) | 对张量列表x1和张量列表x2执行逐元素比较，计算每个元素对应的最大值。 |默认确定性实现|
| [aclnnForeachMaximumScalar](../../foreach/foreach_maximum_scalar/docs/aclnnForeachMaximumScalar.md) | 对张量列表和张量scalar执行逐元素比较，计算每个元素对应的最大值。 |默认确定性实现|
| [aclnnForeachMaximumScalarList](../../foreach/foreach_maximum_scalar_list/docs/aclnnForeachMaximumScalarList.md) | 对张量列表x和标量列表scalars执行逐元素比较，计算每个元素对应的最大值。 |默认确定性实现|
| [aclnnForeachMaximumScalarV2](../../foreach/foreach_maximum_scalar/docs/aclnnForeachMaximumScalarV2.md) | 对张量列表和标量值scalar执行逐元素比较，计算每个元素对应的最大值。 |默认确定性实现|
| [aclnnForeachMinimumList](../../foreach/foreach_minimum_list/docs/aclnnForeachMinimumList.md) | 对张量列表x1和张量列表x2执行逐元素比较，计算每个元素对应的最小值。 |默认确定性实现|
| [aclnnForeachMinimumScalar](../../foreach/foreach_minimum_scalar/docs/aclnnForeachMinimumScalar.md) | 对张量列表x和张量scalar执行逐元素比较，计算每个元素对应的最小值。 |默认确定性实现|
| [aclnnForeachMinimumScalarList](../../foreach/foreach_minimum_scalar_list/docs/aclnnForeachMinimumScalarList.md) | 对张量列表x和标量列表scalars执行逐元素比较，计算每个元素对应的最小值。 |默认确定性实现|
| [aclnnForeachMinimumScalarV2](../../foreach/foreach_minimum_scalar/docs/aclnnForeachMinimumScalarV2.md) | 对张量列表x和标量值scalar执行逐元素比较，计算每个元素对应的最小值。 |默认确定性实现|
| [aclnnForeachMulList](../../foreach/foreach_mul_list/docs/aclnnForeachMulList.md) | 对两个输入张量列表执行逐元素相乘。 |默认确定性实现|
| [aclnnForeachMulScalar](../../foreach/foreach_mul_scalar/docs/aclnnForeachMulScalar.md) | 对输入张量列表的每个张量与张量scalar执行相乘运算。 |默认确定性实现|
| [aclnnForeachMulScalarList](../../foreach/foreach_mul_scalar_list/docs/aclnnForeachMulScalarList.md) | 对输入张量列表与标量列表执行逐元素相乘运算。 |默认确定性实现|
| [aclnnForeachMulScalarV2](../../foreach/foreach_mul_scalar/docs/aclnnForeachMulScalarV2.md) | 对输入张量列表的每个张量与张量scalar执行相乘运算。 |默认确定性实现|
| [aclnnForeachNeg](../../foreach/foreach_neg/docs/aclnnForeachNeg.md) | 计算输入张量列表中每个张量的相反数。 |默认确定性实现|
| [aclnnForeachNonFiniteCheckAndUnscale](../../foreach/foreach_non_finite_check_and_unscale/docs/aclnnForeachNonFiniteCheckAndUnscale.md) | 遍历scaledGrads中的所有Tensor，检查是否存在Inf或NaN，如果存在则将foundInf设置为1.0，否则foundInf保持不变，并对scaledGrads中的所有Tensor进行反缩放。 |默认确定性实现|
| [aclnnForeachNorm](../../foreach/foreach_norm/docs/aclnnForeachNorm.md) | 对输入张量列表的每个张量进行范数运算。 |默认确定性实现|
| [aclnnForeachPowList](../../foreach/foreach_pow_list/docs/aclnnForeachPowList.md) | 对输入张量列表的每个张量进行x2次方运算。 |默认确定性实现|
| [aclnnForeachPowScalar](../../foreach/foreach_pow_scalar/docs/aclnnForeachPowScalar.md) | 对输入张量列表的每个张量进行n次方运算。 |默认确定性实现|
| [aclnnForeachPowScalarV2](../../foreach/foreach_pow_scalar/docs/aclnnForeachPowScalarV2.md) | 对输入张量列表的每个张量进行n次方运算。 |默认确定性实现|
| [aclnnForeachPowScalarAndTensor](../../foreach/foreach_pow_scalar_and_tensor/docs/aclnnForeachPowScalarAndTensor.md) | 对输入张量列表的每个张量进行x次方运算。 |默认确定性实现|
| [aclnnForeachPowScalarList](../../foreach/foreach_pow_scalar_list/docs/aclnnForeachPowScalarList.md) | 对输入张量列表的每个张量进行n次方运算。 |默认确定性实现|
| [aclnnForeachReciprocal](../../foreach/foreach_reciprocal/docs/aclnnForeachReciprocal.md) | 对输入张量列表的每个张量进行倒数运算。 |默认确定性实现|
| [aclnnForeachRoundOffNumber](../../foreach/foreach_round_off_number/docs/aclnnForeachRoundOffNumber.md) | 对输入张量列表的每个张量执行指定精度的四舍五入运算，可通过roundMode指定舍入方式。 |默认确定性实现|
| [aclnnForeachRoundOffNumberV2](../../foreach/foreach_round_off_number/docs/aclnnForeachRoundOffNumberV2.md) | 对输入张量列表的每个张量执行指定精度的四舍五入运算，可通过roundMode指定舍入方式。 |默认确定性实现|
| [aclnnForeachSigmoid](../../foreach/foreach_sigmoid/docs/aclnnForeachSigmoid.md) | 对输入张量列表的每个张量进行Sigmoid函数运算。 |默认确定性实现|
| [aclnnForeachSign](../../foreach/foreach_sign/docs/aclnnForeachSign.md) | 计算输入张量列表中每个张量的符号值。 |默认确定性实现|
| [aclnnForeachSin](../../foreach/foreach_sin/docs/aclnnForeachSin.md) | 对输入张量列表的每个张量进行正弦函数运算。 |默认确定性实现|
| [aclnnForeachSinh](../../foreach/foreach_sinh/docs/aclnnForeachSinh.md) | 对输入张量列表的每个张量进行双曲正弦函数运算。 |默认确定性实现|
| [aclnnForeachSqrt](../../foreach/foreach_sqrt/docs/aclnnForeachSqrt.md) | 对输入张量列表的每个张量进行平方根运算。 |默认确定性实现|
| [aclnnForeachSubList](../../foreach/foreach_sub_list/docs/aclnnForeachSubList.md) | 对输入的两个张量列表执行逐元素相减运算，并可以通过alpha参数调整相减系数。  |默认确定性实现|
| [aclnnForeachSubListV2](../../foreach/foreach_sub_list/docs/aclnnForeachSubListV2.md) | 对两个张量列表中的元素执行逐个相减，并可以通过alpha参数调整相减系数。 |默认确定性实现|
| [aclnnForeachSubScalar](../../foreach/foreach_sub_scalar/docs/aclnnForeachSubScalar.md) | 对输入张量列表的每个张量与张量scalar执行相减运算。 |默认确定性实现|
| [aclnnForeachSubScalarList](../../foreach/foreach_sub_scalar_list/docs/aclnnForeachSubScalarList.md) | 对输入张量列表的每个张量与标量列表scalars的每个标量逐元素执行相减运算。  |默认确定性实现|
| [aclnnForeachSubScalarV2](../../foreach/foreach_sub_scalar/docs/aclnnForeachSubScalarV2.md) | 对输入张量列表的每个张量与标量scalar执行相减运算。  |默认确定性实现|
| [aclnnForeachTan](../../foreach/foreach_tan/docs/aclnnForeachTan.md) | 对输入张量列表的每个张量进行正切函数运算。 |默认确定性实现|
| [aclnnForeachTanh](../../foreach/foreach_tanh/docs/aclnnForeachTanh.md) | 对输入张量列表的每个张量进行双曲正切函数运算。 |默认确定性实现|
| [aclnnForeachZeroInplace](../../foreach/foreach_zero_inplace/docs/aclnnForeachZeroInplace.md) | 原地更新输入张量列表，输入张量列表的每个张量置为0。  |默认确定性实现|
| [aclnnFusedLinearOnlineMaxSum](../../matmul/fused_linear_online_max_sum/docs/aclnnFusedLinearOnlineMaxSum.md) |功能等价Megatron的matmul与fused\_vocab\_parallel\_cross\_entropy的实现，支持vocabulary\_size维度切卡融合matmul与celoss。|默认确定性实现|
| [aclnnFusedLinearCrossEntropyLossGrad](../../matmul/fused_linear_cross_entropy_loss_grad/docs/aclnnFusedLinearCrossEntropyLossGrad.md) |是词汇表并行场景下交叉熵损失计算模块中的一部分，解决超大规模词汇表下的显存和计算效率问题，当前部分为梯度计算实现，用于计算叶子节点`input`和`weight`的梯度。|默认确定性实现|
| [aclnnFusedMatmul](../../matmul/fused_mat_mul/docs/aclnnFusedMatmul.md) |矩阵乘与通用向量计算融合。|默认确定性实现|
| [aclnnGather](../../index/gather_elements_v2/docs/aclnnGather.md) | 对输入tensor中指定的维度dim进行数据聚集。|默认确定性实现|
| [aclnnGatherNd](../../index/gather_nd/docs/aclnnGatherNd.md) | 对于维度为r≥1的输入张量self，和维度q≥1的输入张量indices，将数据切片收集到维度为 (q-1) + (r - indices_shape[-1]) 的输出张量out中。|默认确定性实现|
| [aclnnGatherV2](../../index/gather_v2/docs/aclnnGatherV2.md) | 从输入Tensor的指定维度dim，按index中的下标序号提取元素，保存到out Tensor中。|默认确定性实现|
| [aclnnGatherV3](../../index/gather_v2/docs/aclnnGatherV3.md) | 从输入Tensor的指定维度dim，按index中的下标序号提取元素，batchDims代表运算批次。保存到out Tensor中。|默认确定性实现|
| [aclnnGeGlu](../../activation/ge_glu_v2/docs/aclnnGeGlu.md) |高斯误差线性单元激活函数。|默认确定性实现|
| [aclnnGeGluBackward](../../activation/ge_glu_grad_v2/docs/aclnnGeGluBackward.md) |完成aclnnGeGlu的反向。|默认确定性实现|
| [aclnnGeGluV3](../../activation/ge_glu_v2/docs/aclnnGeGluV3.md) |高斯误差线性单元激活门函数，针对aclnnGeGlu，扩充了设置激活函数操作数据块方向的功能。|默认确定性实现|
| [aclnnGeGluV3Backward](../../activation/ge_glu_grad_v2/docs/aclnnGeGluV3Backward.md) |完成aclnnGeGluV3的反向。|默认确定性实现|
| [aclnnGelu](../../activation/gelu/docs/aclnnGelu.md) |高斯误差线性单元激活函数。|默认确定性实现|
| [aclnnGeluBackward](../../activation/gelu_grad/docs/aclnnGeluBackward.md) |完成aclnnGelu的反向。|默认确定性实现|
| [aclnnGeluBackwardV2](../../activation/gelu_grad_v2/docs/aclnnGeluBackwardV2.md) |完成aclnnGeluV2的反向。|默认确定性实现|
| [aclnnGeluMul](../../activation/gelu_mul/docs/aclnnGeluMul.md) |将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行Gelu计算，将计算结果与x2相乘。|默认确定性实现|
| [aclnnGeluQuant](../../activation/gelu_quant/docs/aclnnGeluQuant.md) |将GeluV2与DynamicQuant/AscendQuantV2进行融合，对输入的数据self进行gelu激活后，对激活的结果进行量化，输出量化后的结果。|默认确定性实现|
| [aclnnGeluV2](../../activation/gelu_v2/docs/aclnnGeluV2.md) |高斯误差线性单元激活函数。|默认确定性实现|
| [aclnnGemm](../../matmul/gemm/docs/aclnnGemm.md) |计算α 乘以A与B的乘积，再与β 和input C的乘积求和。|默认确定性实现|
| [aclnnGemmaRmsNorm](../../norm/gemma_rms_norm/docs/aclnnGemmaRmsNorm.md)|GemmaRmsNorm算子是大模型常用的归一化操作，相比RmsNorm算子，在计算时对gamma执行了+1操作。|默认确定性实现|
| [aclnnGlu](../../activation/sigmoid/docs/aclnnGlu.md) |GLU是一个门控线性单元函数，它将输入张量沿着指定的维度dim平均分成两个张量，并将其前部分张量与后部分张量的Sigmoid函数输出的结果逐元素相乘。|默认确定性实现|
| [aclnnGluBackward](../../activation/sigmoid/docs/aclnnGluBackward.md) |完成aclnnGlu的反向。|默认确定性实现|
| [aclnnHardshrink](../../activation/hard_shrink/docs/aclnnHardshrink.md) |以元素为单位，强制收缩λ范围内的元素。|默认确定性实现|
| [aclnnGroupNorm](../../norm/group_norm/docs/aclnnGroupNorm.md)|计算输入self的组归一化结果out，均值meanOut，标准差的倒数rstdOut。|默认确定性实现|
| [aclnnGroupNormBackward](../../norm/group_norm_grad/docs/aclnnGroupNormBackward.md)|[aclnnGroupNorm](../../norm/group_norm/docs/aclnnGroupNorm.md)的反向计算。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。|默认非确定性实现，支持配置开启|
| [aclnnGroupNormSiluV2](../../norm/group_norm_silu/docs/aclnnGroupNormSiluV2.md)|计算输入self的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及silu的输出。|默认确定性实现|
| [aclnnGroupNormSwish](../../norm/group_norm_swish/docs/aclnnGroupNormSwish.md)|计算输入x的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及swish的输出。|默认确定性实现|
| [aclnnGroupNormSwishGrad](../../norm/group_norm_swish_grad/docs/aclnnGroupNormSwishGrad.md)|[aclnnGroupNormSwish](../../norm/group_norm_swish/docs/aclnnGroupNormSwish.md)的反向操作。|默认非确定性实现，支持配置开启|
| [aclnnGroupQuant](../../quant/group_quant/docs/aclnnGroupQuant.md)|对输入x进行分组量化操作。|默认确定性实现|
| [aclnnHardshrinkBackward](../../activation/hard_shrink_grad/docs/aclnnHardshrinkBackward.md) |aclnnHardshrink计算反向传播的梯度gradInput。|默认确定性实现|
| [aclnnHardsigmoid&aclnnInplaceHardsigmoid](../../activation/hard_sigmoid/docs/aclnnHardsigmoid&aclnnInplaceHardsigmoid.md) |激活函数变种，根据公式返回一个新的tensor。结果的形状与输入tensor相同。|默认确定性实现|
| [aclnnHardsigmoidBackward](../../activation/hard_sigmoid_grad/docs/aclnnHardsigmoidBackward.md) |aclnnHardsigmoid的反向传播。|默认确定性实现|
| [aclnnHardswishBackward](../../activation/hard_swish_grad/docs/aclnnHardswishBackward.md) |aclnnHardswish的反向传播，完成张量self的梯度计算。|默认确定性实现|
| [aclnnHardswishBackwardV2](../../activation/hard_swish_grad_v2/docs/aclnnHardswishBackwardV2.md) |aclnnHardswish的反向传播，完成张量self的梯度计算。|默认确定性实现|
| [aclnnHardswish&aclnnInplaceHardswish](../../activation/hard_swish/docs/aclnnHardswish&aclnnInplaceHardswish.md) |激活函数，返回与输入tensor shape相同的输出tensor，输入的value小于-3时取0，大于3时取该value，其余时刻取value加3的和乘上value再除以6。|默认确定性实现|
| [aclnnHardtanh](../../activation/hardtanh_tanh/docs/aclnnHardtanh&aclnnInplaceHardtanh.md) |将输入的所有元素限制在[clipValueMin,clipValueMax]范围内，若元素大于clipValueMax则限制为clipValueMax，若元素小于clipValueMin则限制为clipValueMin，否则等于元素本身。|默认确定性实现|
| [aclnnHardtanhBackward](../../activation/hardtanh_grad/docs/aclnnHardtanhBackward.md) |激活函数aclnnHardtanh的反向。|默认确定性实现|
| [aclnnHeaviside](../../activation/heaviside/docs/aclnnHeaviside.md) |计算输入input中每个元素的Heaviside阶跃函数，作为模型的激活函数。|默认确定性实现|
| [aclnnIndex](../../index/index/docs/aclnnIndex.md) | 根据索引indices将输入x对应坐标的数据取出。|默认确定性实现|
| [aclnnIndexAdd](../../index/inplace_scatter_add/docs/aclnnIndexAdd.md) | 在指定维度上，根据给定的索引，将源张量中的值加到输入张量中对应位置的值上。|默认非确定性实现，支持配置开启|
| [aclnnIndexCopy&aclnnInplaceIndexCopy](../../index/scatter_update/docs/aclnnIndexCopy&aclnnInplaceIndexCopy.md) | 将index张量中元素值作为索引，针对指定轴dim，把source中元素复制到selfRef的对应位置上。|默认确定性实现|
| [aclnnIndexFillTensor&aclnnInplaceIndexFillTensor](../../index/index_fill_d/docs/aclnnIndexFillTensor&aclnnInplaceIndexFillTensor.md) | 沿输入self的给定轴dim，将index指定位置的值使用value进行替换。|默认确定性实现|
| [aclnnIndexPutImpl](../../index/index_put_v2/docs/aclnnIndexPutImpl.md) | 根据索引 indices 将输入 x 对应坐标的数据与输入 value 进行替换或累加。|默认非确定性实现，支持配置开启|
| [aclnnInplacePut](../../index/scatter_nd_update/docs/aclnnInplacePut.md) | 将selfRef视为一维张量，把index张量中元素值作为索引，如果accumulate为true，把source中元素和selfRef对应的位置上元素做累加操作；如果accumulate为false，把source中元素替换掉selfRef对应位置上的元素。|默认非确定性实现，支持配置开启|
| [aclnnIndexSelect](../../index/gather_v2/docs/aclnnIndexSelect.md) | 从输入Tensor的指定维度dim，按index中的下标序号提取元素，保存到out Tensor中。|默认确定性实现|
| [aclnnInplaceAddRmsNorm](../../norm/inplace_add_rms_norm/docs/aclnnInplaceAddRmsNorm.md)|RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。|默认确定性实现|
| [aclnnInplaceMaskedScatter](../../index/masked_scatter/docs/aclnnInplaceMaskedScatter.md) |根据掩码(mask)张量中元素为True的位置，复制(source)中的元素到(selfRef)对应的位置上。|默认确定性实现|
| [aclnnInplaceUpdateScatter](../../index/quant_update_scatter/docs/aclnnInplaceQuantScatter.md) |先将updates在quantAxis轴上进行量化：quantScales对updates做缩放操作，quantZeroPoints做偏移。然后将量化后的updates中的值按指定的轴axis，根据索引张量indices逐个更新selfRef中对应位置的值。|默认确定性实现|
| [aclnnInstanceNorm](../../norm/instance_norm_v3/docs/aclnnInstanceNorm.md)|用于执行Instance Normalization（实例归一化）操作。|默认确定性实现|
| [aclnnKlDivBackward](../../loss/kl_div_loss_grad/docs/aclnnKlDivBackward.md) | 进行aclnnKlDiv api的结果的反向计算。|默认确定性实现|
| [aclnnKthvalue](../../index/gather_v2/docs/aclnnKthvalue.md) | 返回输入Tensor在指定维度上的第k个最小值及索引。|默认确定性实现|
| [aclnnKvRmsNormRopeCache](../../norm/kv_rms_norm_rope_cache/docs/aclnnKvRmsNormRopeCache.md) |对输入张量(kv)的尾轴，拆分出左半边用于rms_norm计算，右半边用于rope计算，再将计算结果分别scatter到两块cache中。|默认确定性实现|
| [aclnnL1Loss](../../loss/lp_loss/docs/aclnnL1Loss.md) | 计算输入self和目标target中每个元素之间的平均绝对误差（Mean Absolute Error，简称MAE）。|默认确定性实现|
| [aclnnL1LossBackward](../../loss/l1_loss_grad/docs/aclnnL1LossBackward.md) | 计算aclnnL1Loss的反向传播。reduction指定损失函数的计算方式。|默认确定性实现|
| [aclnnLayerNorm&aclnnLayerNormWithImplMode](../../norm/layer_norm_v4/docs/aclnnLayerNorm&aclnnLayerNormWithImplMode.md)|对指定层进行均值为0、标准差为1的归一化计算。|默认确定性实现|
| [aclnnLayerNormBackward](../../norm/layer_norm_grad_v3/docs/aclnnLayerNormBackward.md)|[aclnnNorm](../../norm/lp_norm_v2/docs/aclnnNorm.md)的反向传播。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。|默认确定性实现|
| [aclnnLeakyRelu&aclnnInplaceLeakyRelu](../../activation/leaky_relu/docs/aclnnLeakyRelu&aclnnInplaceLeakyRelu.md) |激活函数，用于解决Relu函数在输入小于0时输出为0的问题，避免神经元无法更新参数。|默认确定性实现|
| [aclnnLeakyReluBackward](../../activation/leaky_relu_grad/docs/aclnnLeakyReluBackward.md) |LeakyRelu激活函数反向。|默认确定性实现|
| [aclnnLinalgVectorNorm](../../norm/lp_norm_v2/docs/aclnnLinalgVectorNorm.md)|计算输入张量的向量范数。|默认非确定性实现，支持配置开启|
| [aclnnLogit](../../loss/logit/docs/aclnnLogit.md) | 该算子是概率到对数几率（log-odds）转换的一个数学运算，常用于概率值的反变换。|默认确定性实现|
| [aclnnLogitGrad](../../loss/logit_grad/docs/aclnnLogitGrad.md) | 完成aclnnLogit的反向传播。|默认确定性实现|
| [aclnnLogSigmoid](../../activation/logsigmoid/docs/aclnnLogSigmoid.md) |对输入张量逐元素实现LogSigmoid运算。|默认确定性实现|
| [aclnnLogSigmoidBackward](../../activation/logsigmoid_grad/docs/aclnnLogSigmoidBackward.md) |aclnnLogSigmoid的反向传播，根据上一层传播的梯度与LogSigmoid正向输入计算其梯度输入。|默认确定性实现|
| [aclnnLogSigmoidForward](../../activation/logsigmoid/docs/aclnnLogSigmoidForward.md) |对输入张量逐元素实现LogSigmoid运算。|默认确定性实现|
| [aclnnLogSoftmax](../../activation/logsoftmax_v2/docs/aclnnLogSoftmax.md) |对输入张量计算logsoftmax值。|默认确定性实现|
| [aclnnLogSoftmaxBackward](../../activation/logsoftmax_grad/docs/aclnnLogSoftmaxBackward.md) |完成aclnnLogSoftmax的反向传播。|默认确定性实现|
| [aclnnMaskedSoftmaxWithRelPosBias](../../norm/masked_softmax_with_rel_pos_bias/docs/aclnnMaskedSoftmaxWithRelPosBias.md) |替换在swinTransformer中使用window attention计算softmax的部分。|默认确定性实现|
| [aclnnMatmul](../../matmul/mat_mul_v3/docs/aclnnMatmul.md) |完成1到6维张量self与张量mat2的矩阵乘计算。|默认确定性实现|
| [aclnnMatmulWeightNz](../../matmul/mat_mul_v3/docs/aclnnMatmulWeightNz.md) |完成张量self与张量mat2的矩阵乘计算，mat2仅支持昇腾亲和数据排布格式。|默认确定性实现|
| [aclnnMaxPool](../../pooling/max_pool_v3/docs/aclnnMaxPool.md)|对于dim=3 或4维的输入张量，进行最大池化（max pooling）操作。|默认确定性实现|
| [aclnnMaxPool2dWithIndices](../../pooling/max_pool3d_with_argmax_v2/docs/aclnnMaxPool2dWithIndices.md)|对于输入信号的输入通道，提供2维（H，W维度）最大池化（max pooling）操作，输出池化后的值out和索引indices。|默认确定性实现|
| [aclnnMaxPool2dWithIndicesBackward](../../pooling/max_pool3d_grad_with_argmax/docs/aclnnMaxPool2dWithIndicesBackward.md)|正向最大池化aclnnMaxPool2dWithIndices的反向传播。|默认非确定性实现，支持配置开启。|
| [aclnnMaxPool2dWithMask](../../pooling/max_pool3d_with_argmax_v2/docs/aclnnMaxPool2dWithMask.md)|对于输入信号的输入通道，提供2维最大池化（max pooling）操作，输出池化后的值out和索引indices（采用mask语义计算得出）。|默认确定性实现|
| [aclnnMaxPool2dWithMaskBackward](../../pooling/max_pool3d_grad_with_argmax/docs/aclnnMaxPool2dWithMaskBackward.md)|正向最大池化aclnnMaxPool2dWithMask的反向传播。|默认非确定性实现，支持配置开启。|
| [aclnnMaxPool3dWithArgmax](../../pooling/max_pool3d_with_argmax_v2/docs/aclnnMaxPool3dWithArgmax.md)|对于输入信号的输入通道，提供3维最大池化（max pooling）操作，输出池化后的值out和索引indices。|默认确定性实现|
| [aclnnMaxPool3dWithArgmaxBackWard](../../pooling/max_pool3d_grad_with_argmax/docs/aclnnMaxPool3dWithArgmaxBackward.md)|正向最大池化aclnnMaxPool3dWithArgmax的反向传播，将梯度回填到每个窗口最大值的坐标处，相同坐标处累加。|默认非确定性实现，支持配置开启。|
| [aclnnMaxUnpool2dBackward](../../index/gather_elements/docs/aclnnMaxUnpool2dBackward.md) | MaxPool2d的逆运算aclnnMaxUnpool2d的反向传播，根据indices索引在out中填入gradOutput的元素值。|默认确定性实现|
| [aclnnMaxUnpool3dBackward](../../index/gather_elements/docs/aclnnMaxUnpool3dBackward.md) | axPool3d的逆运算aclnnMaxUnpool3d的反向传播，根据indices索引在out中填入gradOutput的元素值。|默认确定性实现|
| [aclnnMedian](../../index/gather_v2/docs/aclnnMedian.md) | 返回所有元素的中位数。|默认确定性实现|
| [aclnnMm](../../matmul/mat_mul_v3/docs/aclnnMm.md) |完成2维张量self与张量mat2的矩阵乘计算。|默认确定性实现|
| [aclnnMish&aclnnInplaceMish](../../activation/mish/docs/aclnnMish&aclnnInplaceMish.md) |一个自正则化的非单调神经网络激活函数。|默认确定性实现|
| [aclnnMishBackward](../../activation/mish_grad/docs/aclnnMishBackward.md) |计算aclnnMish的反向传播过程。|默认确定性实现|
| [aclnnMseLoss](../../loss/mse_loss/docs/aclnnMseLoss.md) | 计算输入x和目标y中每个元素之间的均方误差。|默认确定性实现|
| [aclnnMseLossBackward](../../loss/mse_loss_grad_v2/docs/aclnnMseLossBackward.md) | 均方误差函数aclnnMseLoss的反向传播。|默认确定性实现|
| [aclnnMultilabelMarginLoss](../../loss/multilabel_margin_loss/docs/aclnnMultilabelMarginLoss.md) | 计算负对数似然损失值。|默认非确定性实现，支持配置开启。|
| [aclnnMultiScaleDeformableAttnFunction](../../vfusion/multi_scale_deformable_attn_function/docs/aclnnMultiScaleDeformableAttnFunction.md)|通过指定参数来遍历不同尺寸特征图的不同采样点。|默认确定性实现|
| [aclnnMultiScaleDeformableAttentionGrad](../../vfusion/multi_scale_deformable_attention_grad/docs/aclnnMultiScaleDeformableAttentionGrad.md)|正向算子功能主要通过指定参数来遍历不同尺寸特征图的不同采样点。而反向算子的功能为根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。|默认非确定性实现，支持配置开启。|
| [aclnnMv](../../matmul/mv/docs/aclnnMv.md) |计算矩阵input与向量vec的乘积。|默认确定性实现|
| [aclnnNorm](../../norm/lp_norm_v2/docs/aclnnNorm.md)|返回给定张量的矩阵范数或者向量范数。|默认非确定性实现，支持配置开启。|
| [aclnnNLLLoss](../../loss/nll_loss/docs/aclnnNLLLoss.md) | 计算负对数似然损失值。|默认非确定性实现，支持配置开启。|
| [aclnnNLLLoss2d](../../loss/nll_loss/docs/aclnnNLLLoss2d.md) | 计算负对数似然损失值。|默认非确定性实现，支持配置开启。|
| [aclnnNLLLoss2dBackward](../../loss/nll_loss_grad/docs/aclnnNLLLoss2dBackward.md) | 负对数似然损失反向。|默认确定性实现|
| [aclnnNLLLossBackward](../../loss/nll_loss_grad/docs/aclnnNLLLossBackward.md) | 负对数似然损失函数的反向传播。|默认确定性实现|
| [aclnnNonzero](../../index/non_zero/docs/aclnnNonzero.md) | 找出self中非零元素的位置，设self的维度为D，self中非零元素的个数为N，则返回out的shape为D * N，每一列表示一个非零元素的位置坐标。|默认确定性实现|
| [aclnnNonzeroV2](../../index/non_zero/docs/aclnnNonzeroV2.md) | 找出self中非零元素的位置，设self的维度为D，self中非零元素的个数为N，则返回out的shape为D * N，每一列表示一个非零元素的位置坐标。|默认确定性实现|
| [aclnnPrelu](../../activation/prelu/docs/aclnnPrelu.md) |激活函数，Tensor中value大于0，取该value，小于0时取权重与value的乘积。|默认确定性实现|
| [aclnnPreluBackward](../../activation/prelu_grad_update/docs/aclnnPreluBackward.md) |完成aclnnPreluBackward的反向函数。|默认确定性实现|
| [aclnnQuantConvolution](../../conv/convolution_forward/docs/aclnnQuantConvolution.md) |完成per-channel量化的2D/3D卷积计算。|默认确定性实现|
| [aclnnQuantize](../../quant/quantize/docs/aclnnQuantize.md)|对输入张量进行量化处理。|默认确定性实现|
| [aclnnQuantizeBatchNorm](../../norm/quantized_batch_norm/docs/aclnnQuantizedBatchNorm.md)|将输入Tensor执行一个反量化的计算，再根据输入的weight、bias、epsilon执行归一化，最后根据输出的outputScale以及outputZeroPoint执行量化。|默认确定性实现|
| [aclnnQuantMatmul](../../matmul/quant_matmul/docs/aclnnQuantMatmul.md)|完成张量self与张量mat2的矩阵乘计算（支持1维到6维作为输入的矩阵乘）。|默认确定性实现|
| [aclnnQuantMatmulV2](../../matmul/quant_matmul/docs/aclnnQuantMatmulV2.md)|完成量化的矩阵乘计算，最大支持输入维度为3维。相似接口有aclnnMm（仅支持2维Tensor作为输入的矩阵乘）和aclnnBatchMatMul（仅支持三维的矩阵乘，其中第一维是Batch维度）。|默认确定性实现|
| [aclnnQuantMatmulV3](../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV3.md) |完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。|默认确定性实现|
| [aclnnQuantMatmulV4](../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md) |完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。|默认确定性实现|
| [aclnnQuantMatmulV5](../../matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md) |完成量化的矩阵乘计算。|默认确定性实现|
| [aclnnQuantMatmulReduceSumWeightNz](../../matmul/quant_matmul_reduce_sum/docs/aclnnQuantMatmulReduceSumWeightNz.md) |完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出。|默认非确定性实现，支持配置开启。|
| [aclnnQuantMatmulWeightNz](../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulWeightNz.md) |完成量化的矩阵乘计算。|默认确定性实现|
| [aclnnRelu&aclnnInplaceRelu](../../activation/relu/docs/aclnnRelu&aclnnInplaceRelu.md) |激活函数，返回与输入tensor shape相同的tensor, tensor中value大于等于0时，取值该value，小于0，取0。|默认确定性实现|
| [aclnnRenorm&aclnnInplaceRenorm](../../norm/renorm/docs/aclnnRenorm&aclnnInplaceRenorm.md)|返回一个张量，其中输入张量self沿维度dim的每个子张量都经过归一化，使得子张量的p范数低于maxNorm值。|默认确定性实现|
| [aclnnRepeatInterleave](../../index/repeat_interleave/docs/aclnnRepeatInterleave.md) | 将tensor self进行flatten后，重复Tensor repeats中的相应次数。|默认确定性实现|
| [aclnnRepeatInterleaveGrad](../../index/repeat_interleave_grad/docs/aclnnRepeatInterleaveGrad.md) | 算子repeatInterleave的反向, 将yGrad tensor的axis维度按repeats进行ReduceSum。|默认确定性实现|
| [aclnnRmsNorm](../../norm/rms_norm/docs/aclnnRmsNorm.md)|RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。|默认确定性实现|
| [aclnnRmsNormGrad](../../norm/rms_norm_grad/docs/aclnnRmsNormGrad.md)|[aclnnRmsNorm](../../norm/rms_norm/docs/aclnnRmsNorm.md)的反向计算。用于计算RMSNorm的梯度，即在反向传播过程中计算输入张量的梯度。|默认非确定性实现，支持配置开启。|
| [aclnnRmsNormQuant]()|aclnnRmsNormQuant算子将aclnnRmsNorm前的Add算子以及aclnnRmsNorm后的Quantize算子融合起来，减少搬入搬出操作。|默认确定性实现|
| [aclnnRReluWithNoise&aclnnInplaceRReluWithNoise](../../activation/leaky_relu/docs/aclnnRReluWithNoise&aclnnInplaceRReluWithNoise.md) |实现了带噪声的随机修正线性单元激活函数，它在输入小于等于0时，斜率为a；输入大于0时斜率为1。|默认确定性实现|
| [aclnnScatter&aclnnInplaceScatter](../../index/scatter_elements_v2/docs/aclnnScatter&aclnnInplaceScatter.md) |将tensor src中的值按指定的轴和方向和对应的位置关系逐个替换/累加/累乘至tensor self中。|默认确定性实现|
| [aclnnScatterAdd](../../index/scatter_add/docs/aclnnScatterAdd.md) |将src tensor中的值按指定的轴方向和index tensor中的位置关系逐个填入self tensor中，若有多于一个src值被填入到self的同一位置，那么这些值将会在这一位置上进行累加。 |默认确定性实现|
| [aclnnScatterNd](../../index/scatter_nd_update/docs/aclnnScatterNd.md) | 拷贝data的数据至out，同时在指定indices处根据updates更新out中的数据。|默认确定性实现|
| [aclnnScatterNdUpdate](../../index/scatter_nd_update/docs/aclnnScatterNdUpdate.md) | 将tensor updates中的值按指定的索引indices逐个更新tensor varRef中的值。|默认确定性实现|
| [aclnnScaledMaskedSoftmax](../../vfusion/scaled_masked_softmax_v2/docs/aclnnScaledMaskedSoftmax.md)|将输入的数据x先进行scale缩放和mask，然后执行softmax的输出。|默认确定性实现|
| [aclnnScaledMaskedSoftmaxBackward](../../vfusion/scaled_masked_softmax_grad_v2/docs/aclnnScaledMaskedSoftmaxBackward.md)|softmax的反向传播，并对结果进行缩放以及掩码。|默认非确定性实现，支持配置开启。|
| [aclnnSelu&aclnnInplaceSelu](../../activation/selu/docs/aclnnSelu&aclnnInplaceSelu.md) |对输入tensor逐元素进行Selu符号函数的运算并输出结果tensor。|默认确定性实现|
| [aclnnSeluBackward](../../activation/selu_grad/docs/aclnnSeluBackward.md) |完成aclnnSelu的反向。|默认确定性实现|
| [aclnnShrink](../../activation/shrink/docs/aclnnShrink.md) |对输入张量进行非线性变换，根据输入值self与阈值lambd的关系，对输入通过偏移量bias进行缩放和偏移处理。|默认确定性实现|
| [aclnnSigmoid&aclnnInplaceSigmoid](../../activation/sigmoid/docs/aclnnSigmoid&aclnnInplaceSigmoid.md) |对输入Tensor完成sigmoid运算。|默认确定性实现|
| [aclnnSigmoidBackward](../../activation/sigmoid_grad/docs/aclnnSigmoidBackward.md) |完成sigmoid的反向传播，根据sigmoid反向传播梯度与正向输出计算sigmoid的梯度输入。|默认确定性实现|
| [aclnnSilu](../../activation/swish/docs/aclnnSilu.md) |该算子也被称为Swish函数。|默认确定性实现|
| [aclnnSiluBackward](../../activation/silu_grad/docs/aclnnSiluBackward.md) |aclnnSilu的反向传播，根据silu反向传播梯度与正向输出计算silu的梯度输入。|默认确定性实现|
| [aclnnSmoothL1Loss](../../loss/smooth_l1_loss_v2/docs/aclnnSmoothL1Loss.md) | 计算SmoothL1损失函数。|默认确定性实现|
| [aclnnSmoothL1LossBackward](../../loss/smooth_l1_loss_grad_v2/docs/aclnnSmoothL1LossBackward.md) | 计算aclnnSmoothL1Loss api的反向传播。|默认确定性实现|
| [aclnnSoftMarginLoss](../../loss/soft_margin_loss/docs/aclnnSoftMarginLoss.md) | 计算输入self和目标target的二分类逻辑损失函数。|默认确定性实现|
| [aclnnSoftMarginLossBackward](../../loss/soft_margin_loss_grad/docs/aclnnSoftMarginLossBackward.md) | 计算aclnnSoftMarginLoss二分类逻辑损失函数的反向传播。|默认确定性实现|
| [aclnnSoftmax](../../activation/softmax_v2/docs/aclnnSoftmax.md) |对输入张量计算softmax值。|默认确定性实现|
| [aclnnSoftmaxBackward](../../activation/softmax_grad/docs/aclnnSoftmaxBackward.md) |完成softmax的反向传播。|默认确定性实现|
| [aclnnSoftmaxCrossEntropyWithLogits](../../activation/softmax_cross_entropy_with_logits/docs/aclnnSoftmaxCrossEntropyWithLogits.md) |计算softmax和cross entropy的交叉熵损失，并给出对输入logits的反向梯度。|默认确定性实现|
| [aclnnSoftplus](../../activation/softplus_v2/docs/aclnnSoftplus.md) |激活函数softplus。|默认确定性实现|
| [aclnnSoftplusBackward](../../activation/softplus_v2_grad/docs/aclnnSoftplusBackward.md) |aclnnSoftplus的反向传播。|默认确定性实现|
| [aclnnSoftshrink](../../activation/softshrink/docs/aclnnSoftshrink.md) |以元素为单位，强制收缩λ范围内的元素。|默认确定性实现|
| [aclnnSoftshrinkBackward](../../activation/softshrink_grad/docs/aclnnSoftshrinkBackward.md) |完成Softshrink函数的反向接口。|默认确定性实现|
| [aclnnSquaredRelu](../../activation/squared_relu/docs/aclnnSquaredRelu.md) |SquaredReLU 函数是一个基于标准ReLU函数的变体，其主要特点是对ReLU函数的输出进行平方，常作为模型的激活函数。|默认确定性实现|
| [aclnnSwiGlu](../../activation/swi_glu/docs/aclnnSwiGlu.md) |Swish门控线性单元激活函数，实现x的SwiGlu计算。|默认确定性实现|
| [aclnnSwiGluGrad](../../activation/swi_glu_grad/docs/aclnnSwiGluGrad.md) |完成aclnnSwiGlu的反向计算，完成x的SwiGlu反向梯度计算。|默认确定性实现|
| [aclnnSwiGluQuant](../../quant/swi_glu_quant/docs/aclnnSwiGluQuant.md)|在SwiGlu激活函数后添加quant操作，实现输入x的SwiGluQuant计算。|默认确定性实现|
| [aclnnSwiGluQuantV2](../../quant/swi_glu_quant/docs/aclnnSwiGluQuantV2.md)|在SwiGlu激活函数后添加quant操作，实现输入x的SwiGluQuant计算，支持int8或int4量化输出。|默认确定性实现|
| [aclnnSwish](../../activation/swish/docs/aclnnSwish.md) |Swish激活函数，对输入Tensor逐元素进行Swish函数运算并输出结果Tensor。|默认确定性实现|
| [aclnnSwishBackward](../../activation/swish_grad/docs/aclnnSwishBackward.md) |aclnnSwishBackward是aclnnSwish激活函数的反向传播，用于计算Swish激活函数的梯度。|默认确定性实现|
| [aclnnSyncBatchNormGatherStats](../../norm/sync_batch_norm_gather_stats/docs/aclnnSyncBatchNormGatherStats.md) |收集所有device的均值和方差，更新全局的均值和方差。|默认确定性实现|
| [aclnnTake](../../index/gather_v2/docs/aclnnTake.md) | 将输入的self张量视为一维数组，把index的值当作索引，从self中取值，输出shape与index一致的Tensor。|默认确定性实现|
| [aclnnThreshold&aclnnInplaceThreshold](../../activation/threshold/docs/aclnnThreshold&aclnnInplaceThreshold.md) |对输入x进行阈值操作。当x中的elements大于threshold时，返回elements；否则，返回value。|默认确定性实现|
| [aclnnThresholdBackward](../../activation/threshold_grad_v2_d/docs/aclnnThresholdBackward.md) |完成aclnnThreshold的反向。|默认确定性实现|
| [aclnnTransposeBatchMatMul](../../matmul/transpose_batch_mat_mul/docs/aclnnTransposeBatchMatMul.md) |完成张量x1与张量x2的矩阵乘计算。|默认确定性实现|
| [aclnnTransposeQuantBatchMatMul](../../matmul/transpose_quant_batch_mat_mul/docs/aclnnTransposeQuantBatchMatMul.md) |完成张量x1与张量x2量化的矩阵乘计算。|默认确定性实现|
| [aclnnTransQuantParam](../../quant/trans_quant_param/docs/aclnnTransQuantParam.md)|将输入scale数据从FLOAT32类型转换为硬件需要的UINT64类型，并存储到quantParam中。|默认确定性实现|
| [aclnnTransQuantParamV2](../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV2.md)|完成量化计算参数scale数据类型的转换，将FLOAT32的数据类型转换为硬件需要的UINT64，INT64类型。|默认确定性实现|
| [aclnnTransQuantParamV3](../../quant/trans_quant_param_v2/docs/aclnnTransQuantParamV3.md)|完成量化计算参数scale数据类型的转换，将Float32的数据类型转换为硬件需要的UINT64，INT64类型。|默认确定性实现|
| [aclnnTopKTopPSampleV2](../../index/top_k_top_p_sample_v2/docs/aclnnTopKTopPSampleV2.md) |根据输入词频logits、topK/topP/minP采样参数、随机采样权重分布q，进行topK-topP-minP-sample采样计算。当输入isNeedSampleResult为false时，输出每个batch的最大词频logitsSelectIdx，以及topK-topP-minP采样后的词频分布logitsTopKPSelect；当输入isNeedSampleResult为true时，输出topK-topP-minP采样后的中间计算结果logitsIdx和logitsSortMasked，其中logitsSortMasked为词频logits经过topK-topP-minP采样计算后的中间结果，logitsIdx为logitsSortMasked在logits中对应的索引。|默认确定性实现|
| [aclnnUnique](../../index/scatter_elements/docs/aclnnUnique.md) |返回输入张量中的唯一元素。|默认确定性实现|
| [aclnnUnique2](../../index/scatter_elements/docs/aclnnUnique2.md) |对输入张量self进行去重，返回self中的唯一元素。unique功能的增强，新增返回值countsOut，表示valueOut中各元素在输入self中出现的次数，用returnCounts参数控制。|默认确定性实现|
| [aclnnUniqueConsecutive](../../index/unique_consecutive/docs/aclnnUniqueConsecutive.md) |去除每一个元素后的重复元素。当dim不为空时，去除对应维度上的每一个张量后的重复张量。|默认确定性实现|
| [aclnnUniqueDim](../../index/unique_with_counts_ext2/docs/aclnnUniqueDim.md) |在某一dim轴上，对输入张量self做去重操作。|默认确定性实现|
| [aclnnWeightQuantBatchMatmul](../../matmul/weight_quant_batch_matmul/docs/aclnnWeightQuantBatchMatmul.md)|伪量化用于对self * mat2（matmul/batchmatmul）中的mat2进行量化。|-|
| [aclnnWeightQuantBatchMatmulNz](../../matmul/weight_quant_batch_matmul_v2/docs/aclnnWeightQuantBatchMatmulNz.md) |完成一个输入为伪量化场景的矩阵乘计算，仅支持NZ场景。|默认确定性实现|
| [aclnnWeightQuantBatchMatmulV2](../../matmul/weight_quant_batch_matmul_v2/docs/aclnnWeightQuantBatchMatmulV2.md) |完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。|默认非确定性实现，支持配置开启。|
| [aclnnWeightQuantBatchMatmulV3](../../matmul/weight_quant_batch_matmul_v2/docs/aclnnWeightQuantBatchMatmulV3.md) |完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。|默认非确定性实现，支持配置开启。|

## 废弃接口
|废弃接口|说明|
|-------|----|
| [aclnnWeightQuantBatchMatmul](../../matmul/weight_quant_batch_matmul/docs/aclnnWeightQuantBatchMatmul.md)|此接口后续版本会废弃，请使用最新接口[aclnnWeightQuantBatchMatmulV2](../../matmul/weight_quant_batch_matmul_v2/docs/aclnnWeightQuantBatchMatmulV2.md)、[aclnnWeightQuantBatchMatmulV3](../../matmul/weight_quant_batch_matmul_v2/docs/aclnnWeightQuantBatchMatmulV3.md)。 |
| [aclnnQuantMatmul](../../matmul/quant_matmul/docs/aclnnQuantMatmul.md)|此接口后续版本会废弃，请使用最新接口[aclnnQuantMatmulV4](../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md)。 |
| [aclnnQuantMatmulV2](../../matmul/quant_matmul/docs/aclnnQuantMatmulV2.md)|此接口后续版本会废弃，请使用最新接口[aclnnQuantMatmulV4](../../matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md)。 |