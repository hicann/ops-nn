# ConvolutionForward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：实现卷积功能，支持 1D 卷积、2D 卷积、3D 卷积，同时支持转置卷积、空洞卷积、分组卷积。
  对于入参 `transposed = True` 时，表示使用转置卷积或者分数步长卷积。它可以看作是普通卷积的梯度或者逆向操作，即从卷积的输出形状恢复到输入形状，同时保持与卷积相容的连接模式。它的参数和普通卷积类似，包括输入通道数、输出通道数、卷积核大小、步长、填充、输出填充、分组、偏置、扩张等。

- 计算公式：

  我们假定输入（input）的 shape 是 $(N, C_{\text{in}}, H, W)$，（weight）的 shape 是 $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$，输出（output）的 shape 是 $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$，那输出将被表示为：

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
| input | 输入 | <ul><li>公式中的 input，表示卷积输入。</li><li>数据类型需要与 weight 满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li><li>N≥0，C≥1，其他维度≥0。</li></ul> | FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN| NCL、NCHW、NCDHW |
| weight | 输入 | <ul><li>公式中的 weight，表示卷积权重。</li><li>数据类型需要与 input 满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li><li>所有维度≥1。</li></ul> | FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN | NCL、NCHW、NCDHW |
| bias | 输入 | <ul><li>公式中的 bias，表示卷积偏置。</li><li>当 transposed=false 时为一维且数值与 weight 第一维相等；当 transposed=true 时为一维且数值与 weight.shape[1] * groups 相等。</li></ul> | FLOAT、FLOAT16、BFLOAT16 | ND |
| stride | 输入 | <ul><li>卷积扫描步长。</li><li>数组长度需等于 input 的维度减 2，值应该大于 0。</li></ul> | INT32 | - |
| padding | 输入 | <ul><li>对 input 的填充。</li><li>数组长度：conv1d 非转置为 1 或 2；conv2d 为 2 或 4；conv3d 为 3。值应该大于等于 0。</li></ul> | INT32 | - |
| dilation | 输入 | <ul><li>卷积核中元素的间隔。</li><li>数组长度需等于 input 的维度减 2，值应该大于 0。</li></ul> | INT32 | - |
| transposed | 输入 | <ul><li>是否为转置卷积。</li></ul> | BOOL | - |
| outputPadding | 输入 | <ul><li>转置卷积情况下，对输出所有边的填充。</li><li>非转置卷积情况下忽略该配置。值应大于等于0，且小于 stride 或 dilation 对应维度的值。</li></ul> | INT32 | - |
| groups | 输入 | <ul><li>表示从输入通道到输出通道的块链接个数。</li><li>数值必须大于 0，且满足 groups*weight 的 C 维度=input 的 C 维度。</li></ul> | INT64 | - |
| output | 输出 | <ul><li>公式中的 out，表示卷积输出。</li><li>数据类型需要与 input 与 weight 推导之后的数据类型保持一致。</li><li>通道数等于 weight 第一维，其他维度≥0。</li></ul> | FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN | NCL、NCHW、NCDHW |
| cubeMathType | 输入 | <ul><li>用于判断 Cube 单元应该使用哪种计算逻辑进行运算。</li><li>0 (KEEP_DTYPE): 保持输入数据类型进行计算。</li><li> 1 (ALLOW_FP32_DOWN_PRECISION): 允许 FLOAT32 降低精度计算，提升性能。</li><li> 2 (USE_FP16): 使用 FLOAT16 精度进行计算。</li><li> 3 (USE_HF32): 使用 HF32（混合精度）进行计算。</li></ul> | INT8 | - |

* <term>Atlas A2 训练系列产品 / Atlas 800I A2 推理产品 / A200I A2 Box 异构组件、Atlas A3 训练系列产品 / Atlas A3 推理系列产品</term>：
    - input、weight 数据类型不支持 HIFLOAT8、FLOAT8_E4M3FN。
    - bias 数据类型不支持 HIFLOAT8、FLOAT8_E4M3FN。数据类型与 self、weight 一致。
    - conv1d、conv2d、conv3d 正向场景下 bias 会转成 FLOAT 参与计算。
    - conv2d 和 conv3d transposed=true 场景，weight H、W 的大小应该在[1,255]范围内，其他维度应该大于等于 1。
    - conv1d transposed=true 场景，weight L 的大小应该在[1,255]范围内，其他维度应该大于等于 1。
    - conv3d 正向场景，weight H、W 的大小应该在[1,511]范围内。
    - cubeMathType 为 0(KEEP_DTYPE) 时，当输入是 FLOAT 暂不支持。
    - cubeMathType 为 1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是 FLOAT 允许转换为 HFLOAT32 计算。
    - cubeMathType 为 2(USE_FP16) 时，当输入是 BFLOAT16 不支持该选项。
    - cubeMathType 为 3(USE_HF32) 时，当输入是 FLOAT 转换为 HFLOAT32 计算。
* <term>Atlas 推理系列产品、Atlas 训练系列产品</term>：
    - input、weight 数据类型支持 FLOAT、FLOAT16。
    - bias 数据类型不支持 BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。
    - cubeMathType 为 0(KEEP_DTYPE) 时，当输入是 FLOAT 暂不支持。
    - cubeMathType 为 1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是 FLOAT 允许转换为 FLOAT16 计算。
    - cubeMathType 为 2(USE_FP16) 时，当输入是 BFLOAT16 不支持该选项。
    - cubeMathType 为 3(USE_HF32) 时暂不支持。
* <term>昇腾 910_95 AI 处理器</term>：
    - input、weight 数据类型支持 FLOAT、FLOAT16、BFLOAT16、HIFLOAT8。
    - transposed=true 时：input 数据类型额外支持 FLOAT8_E4M3FN，支持 N 维度大于等于0，其他各个维度的大小应该大于等于1。
    - transposed=false 时：当 input 数据类型为 HIFLOAT8 时，weight 的数据类型必须与 input 一致。支持 N 维度大于等于0，支持 D、H、W 维度大于等于0（等于0的场景仅在 output 推导的 D、H、W 维度也等于0时支持），支持 C 维度大于等于0（等于0的场景仅在 output 推导的 N、C、D、H、W 其中某一维度等于0时支持）。
    - bias 数据类型不支持 HIFLOAT8、FLOAT8_E4M3FN。
    - transposed=true 时：当 input 和 weight 数据类型是 HIFLOAT8 和 FLOAT8_E4M3FN 时，不支持带 bias。
    - transposed=false 时：当 input 和 weight 数据类型是 HIFLOAT8 时，bias 数据类型会转成 FLOAT 参与计算。
    - cubeMathType 为 1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是 FLOAT 允许转换为 HFLOAT32 计算。
    - cubeMathType 为 2(USE_FP16) 时，当输入是 BFLOAT16 不支持该选项。
    - cubeMathType 为 3(USE_HF32) 时，当输入是 FLOAT 转换为 HFLOAT32 计算。
* <term>Atlas 200I/500 A2 推理产品</term>：

## 约束说明

* <term>Atlas 推理系列产品</term>：当前仅支持1D和2D卷积，暂不支持3D卷积。input, weight, bias中每一组tensor的每一维大小都应不大于1000000。
* <term>Atlas 训练系列产品</term>：支持1D和2D卷积，3D卷积仅支持transposed为false且输入数据类型为FLOAT16场景。input, weight, bias 中每一组 tensor 的每一维大小都应不大于 1000000。
* <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：input, weight, bias 中每一组 tensor 的每一维大小都应不大于 1000000。
* <term>昇腾 910_95 AI处理器</term>：transposed 为 true 的场景，支持 1D、2D 和 3D 卷积，支持空 Tensor。
* <term>Atlas 200I/500 A2 推理产品</term>：当前仅支持 2D 卷积，暂不支持 1D、3D 卷积。input, weight, bias 中每一组 tensor 的每一维大小都应不大于 1000000。
* 由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击[Link](https://www.hiascend.com/support)获取技术支持。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| aclnn接口 | [test_aclnn_convolution_1d_transpose](examples/test_aclnn_convolution_1d_transpose.cpp) | 通过[aclnnConvolution](docs/aclnnConvolution.md)接口演示 1D 转置卷积场景，调用 Conv2DTranspose 算子 |
| aclnn接口 | [test_aclnn_convolution_2d_transpose](examples/test_aclnn_convolution_2d_transpose.cpp) | 通过[aclnnConvolution](docs/aclnnConvolution.md)接口演示 2D 转置卷积场景，调用 Conv2DTranspose 算子 |
| aclnn接口 | [test_aclnn_convolution_3d_transpose](examples/test_aclnn_convolution_3d_transpose.cpp) | 通过[aclnnConvolution](docs/aclnnConvolution.md)接口演示 3D 转置卷积场景，调用 Conv3DTransposeV2 算子 |
| aclnn接口 | [test_aclnn_conv_depthwise_2d](examples/test_aclnn_conv_depthwise_2d.cpp) | 使用[aclnnConvDepthwise2d](docs/aclnnConvDepthwise2d.md)接口演示 2D 深度可分离卷积，调用 Conv2D 算子 |
| aclnn接口 | [test_aclnn_conv_tbc](examples/test_aclnn_conv_tbc.cpp) | 使用[aclnnConvTbc](docs/aclnnConvTbc.md)接口演示 time-batch-channel 卷积，调用 Conv2D 算子 |
| aclnn接口 | [test_aclnn_quant_convolution](examples/test_aclnn_quant_convolution.cpp) | 使用[aclnnQuantConvolution](docs/aclnnQuantConvolution.md)接口演示 3D 量化卷积，调用 Conv3DV2 算子 |
