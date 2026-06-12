# ConvolutionForward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |

## 功能说明

- 算子功能：实现卷积功能，支持1D卷积、2D卷积、3D卷积，同时支持转置卷积、空洞卷积、分组卷积。
  对于入参`transposed = True` 时，表示使用转置卷积或者分数步长卷积。它可以看作是普通卷积的梯度或者逆向操作，即从卷积的输出形状恢复到输入形状，同时保持与卷积相容的连接模式。它的参数和普通卷积类似，包括输入通道数、输出通道数、卷积核大小、步长、填充、输出填充、分组、偏置、扩张等。

- 计算公式：

  我们假定输入（input）的shape是 $(N, C_{\text{in}}, H, W)$，（weight）的shape是 $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$，输出（output）的shape是 $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$，那输出将被表示为：

  $$
    \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
  $$

  其中，$\star$ 表示卷积计算，根据卷积输入的维度，卷积的类型（空洞卷积、分组卷积）而定。$N$ 代表批次大小（batch size），$C$ 代表通道数，$W$ 和 $H$ 分别代表宽和高，相应输出维度的计算公式如下：

  $$
    H_{\text{out}}=[(H + 2 * padding[0] - dilation[0] * (K_h - 1) - 1 ) / stride[0]] + 1 \\
    W_{\text{out}}=[(W + 2 * padding[1] - dilation[1] * (K_w - 1) - 1 ) / stride[1]] + 1
  $$

## 参数说明

| <div style="width:120px">参数名</div>  | <div style="width:120px">输入/输出/属性</div>  | <div style="width:350px">描述</div> | <div style="width:350px">数据类型</div>  | <div style="width:220px">数据格式</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------------- | ----------------- | --------------------- |
| input | 输入 | <ul><li>公式中的input，表示卷积输入。</li><li>数据类型需要与weight满足数据类型推导规则（[互推导关系](../../docs/zh/context/互推导关系.md)和<a href="#约束说明">约束说明</a>）。</li><li>N≥0，C≥1，其他维度≥0。</li></ul> | FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN| NCL、NCHW、NCDHW |
| weight | 输入 | <ul><li>公式中的weight，表示卷积权重。</li><li>数据类型需要与input满足数据类型推导规则（[互推导关系](../../docs/zh/context/互推导关系.md)和<a href="#约束说明">约束说明</a>）。</li><li>所有维度≥1。</li></ul> | FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN | NCL、NCHW、NCDHW |
| bias | 输入 | <ul><li>公式中的bias，表示卷积偏置。</li><li>当transposed=false时为一维且数值与weight第一维相等；当transposed=true时为一维且数值与weight.shape[1] * groups相等。</li></ul> | FLOAT、FLOAT16、BFLOAT16 | ND |
| stride | 输入 | <ul><li>卷积扫描步长。</li><li>数组长度需等于input的维度减2，值应该大于0。</li></ul> | INT32 | - |
| padding | 输入 | <ul><li>对input的填充。</li><li>数组长度：conv1d非转置为1或2；conv2d为2或4；conv3d为3。值应该大于等于0。</li></ul> | INT32 | - |
| dilation | 输入 | <ul><li>卷积核中元素的间隔。</li><li>数组长度需等于input的维度减2，值应该大于0。</li></ul> | INT32 | - |
| transposed | 输入 | <ul><li>是否为转置卷积。</li></ul> | BOOL | - |
| outputPadding | 输入 | <ul><li>转置卷积情况下，对输出所有边的填充。</li><li>非转置卷积情况下忽略该配置。值应大于等于0，且小于stride或dilation对应维度的值。</li></ul> | INT32 | - |
| groups | 输入 | <ul><li>表示从输入通道到输出通道的块链接个数。</li><li>数值必须大于0，且满足groups*weight的C维度=input的C维度。</li></ul> | INT64 | - |
| output | 输出 | <ul><li>公式中的out，表示卷积输出。</li><li>数据类型需要与input与weight推导之后的数据类型保持一致。</li><li>通道数等于weight第一维，其他维度≥0。</li></ul> | FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN | NCL、NCHW、NCDHW |
| cubeMathType | 输入 | <ul><li>用于判断Cube单元应该使用哪种计算逻辑进行运算。</li><li>0 (KEEP_DTYPE): 保持输入数据类型进行计算。</li><li> 1 (ALLOW_FP32_DOWN_PRECISION): 允许FLOAT32降低精度计算，提升性能。</li><li> 2 (USE_FP16): 使用FLOAT16精度进行计算。</li><li> 3 (USE_HF32): 使用HF32（混合精度）进行计算。</li></ul> | INT8 | - |

* <term>Atlas A2训练系列产品/Atlas A2推理系列产品</term>、<term>Atlas A3训练系列产品/Atlas A3推理系列产品</term>：
    - input、weight数据类型不支持HIFLOAT8、FLOAT8_E4M3FN。
    - bias数据类型不支持HIFLOAT8、FLOAT8_E4M3FN。数据类型与input、weight一致。
    - conv1d、conv2d、conv3d正向场景下bias会转成FLOAT参与计算。
    - conv2d和conv3d transposed=true场景，weight H、W的大小应该在[1,255]范围内，其他维度应该大于等于1。
    - conv1d transposed=true场景，weight L的大小应该在[1,255]范围内，其他维度应该大于等于1。
    - conv3d正向场景，weight H、W的大小应该在[1,511]范围内。
    - cubeMathType为0(KEEP_DTYPE) 时，当输入是FLOAT暂不支持。
    - cubeMathType为1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是FLOAT允许转换为HFLOAT32计算。
    - cubeMathType为2(USE_FP16) 时，当输入是BFLOAT16不支持该选项。
    - cubeMathType为3(USE_HF32) 时，当输入是FLOAT转换为HFLOAT32计算。
* <term>Ascend 950PR/Ascend 950DT</term>：
    - input、weight数据类型支持FLOAT、FLOAT16、BFLOAT16、HIFLOAT8。
    - transposed=true时：input数据类型额外支持FLOAT8_E4M3FN，支持N维度大于等于0，其他各个维度的大小应该大于等于1。
    - transposed=false时：当input数据类型为HIFLOAT8时，weight的数据类型必须与input一致。支持N维度大于等于0，支持D、H、W维度大于等于0（等于0的场景仅在output推导的D、H、W维度也等于0时支持），支持C维度大于等于0（等于0的场景仅在output推导的N、C、D、H、W其中某一维度等于0时支持）。
    - bias数据类型不支持HIFLOAT8、FLOAT8_E4M3FN。
    - transposed=true时：当input和weight数据类型是HIFLOAT8和FLOAT8_E4M3FN时，不支持带bias。
    - transposed=false时：当input和weight数据类型是HIFLOAT8时，bias数据类型会转成FLOAT参与计算。
    - cubeMathType为1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是FLOAT允许转换为HFLOAT32计算。
    - cubeMathType为2(USE_FP16) 时，当输入是BFLOAT16不支持该选项。
    - cubeMathType为3(USE_HF32) 时，当输入是FLOAT转换为HFLOAT32计算。
    
## 约束说明

* <term>Atlas A2训练系列产品/Atlas A2推理系列产品</term>、<term>Atlas A3训练系列产品/Atlas A3推理系列产品</term>：input, weight, bias中每一组tensor的每一维大小都应不大于1000000。
* <term>Ascend 950PR/Ascend 950DT</term>：transposed为true的场景，支持1D、2D和3D卷积，支持空Tensor。
* 由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击[Link](https://www.hiascend.com/support)获取技术支持。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| aclnn接口 | [test_aclnn_convolution_1d_transpose](examples/test_aclnn_convolution_1d_transpose.cpp) | 通过[aclnnConvolution](docs/aclnnConvolution.md)接口演示1D转置卷积场景，调用Conv2DTranspose算子 |
| aclnn接口 | [test_aclnn_convolution_2d_transpose](examples/test_aclnn_convolution_2d_transpose.cpp) | 通过[aclnnConvolution](docs/aclnnConvolution.md)接口演示2D转置卷积场景，调用Conv2DTranspose算子 |
| aclnn接口 | [test_aclnn_convolution_3d_transpose](examples/test_aclnn_convolution_3d_transpose.cpp) | 通过[aclnnConvolution](docs/aclnnConvolution.md)接口演示3D转置卷积场景，调用Conv3DTransposeV2算子 |
| aclnn接口 | [test_aclnn_conv_depthwise_2d](examples/test_aclnn_conv_depthwise_2d.cpp) | 使用[aclnnConvDepthwise2d](docs/aclnnConvDepthwise2d.md)接口演示2D深度可分离卷积，调用Conv2D算子 |
| aclnn接口 | [test_aclnn_conv_tbc](examples/test_aclnn_conv_tbc.cpp) | 使用[aclnnConvTbc](docs/aclnnConvTbc.md)接口演示time-batch-channel卷积，调用Conv2D算子 |
| aclnn接口 | [test_aclnn_quant_convolution](examples/test_aclnn_quant_convolution.cpp) | 使用[aclnnQuantConvolution](docs/aclnnQuantConvolution.md)接口演示3D量化卷积，调用Conv3DV2算子 |
