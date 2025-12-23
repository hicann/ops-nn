# ConvolutionBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |


## 功能说明

- 算子功能：卷积的反向传播。根据输出掩码设置计算输入、权重和偏差的梯度。此函数支持1D、2D和3D卷积。  

- 计算公式

 假定输入的shape为($N,C_{in},D_{in},H_{in},W_{in}$)、输出的shape为($N,C_{out},D_{out},H_{out},W_{out}$)，那么它们与卷积步长($stride$)、卷积核大小($kernelSize，kD,kH,kW$)、膨胀参数($dilation$)的关系是：

  $$
    D_{out}=\lfloor \frac{D_{in}+2*padding[0]-dilation[0] * (kernelSize[0] - 1) - 1}{stride[0]}+1 \rfloor
  $$

  $$
    H_{out}=\lfloor \frac{H_{in}+2*padding[1]-dilation[1] * (kernelSize[1] - 1) - 1}{stride[1]}+1 \rfloor
  $$

  $$
    W_{out}=\lfloor \frac{W_{in}+2*padding[2]-dilation[2] * (kernelSize[2] -1) -1}{stride[2]}+1 \rfloor
  $$
  
  卷积反向传播需要计算对卷积正向的输入张量 $x$、卷积核权重张量 $w$ 和偏置 $b$ 的梯度。  
  - 对于 $x$ 的梯度 $\frac{\partial L}{\partial x}$：
  
    $$
    \frac{\partial L}{\partial x_{n, c_{in}, i, j}} = \sum_{c_{out}=1}^{C_{out}} \sum_{p=1}^{k_H} \sum_{q=1}^{k_W} \frac{\partial L}{\partial y_{n, c_{out}, i-p, j-q}}\cdot w_{c_{out}, c_{in}, p, q}
    $$
  
    其中，$L$ 为损失函数，$\frac{\partial L}{\partial y}$ 为输出张量 $y$ 对 $L$ 的梯度。  
  
  - 对于 $w$ 的梯度 $\frac{\partial L}{\partial w}$：
  
    $$
    \frac{\partial L}{\partial w_{c_{out}, c_{in}, p, q}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, i \cdot s_H + p, j \cdot s_W + q} \cdot \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$
  
  - 对于 $b$ 的梯度 $\frac{\partial L}{\partial b}$：
  
    $$
    \frac{\partial L}{\partial b_{c_{out}}} = \sum_{n=1}^{N}       \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$

## 参数说明

| <div style="width:120px">参数名</div>  | <div style="width:120px">输入/输出/属性</div>  | <div style="width:350px">描述</div> | <div style="width:350px">数据类型</div>  | <div style="width:220px">数据格式</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------------- | ----------------- | --------------------- |
| gradOutput | 输入 | <ul><li>输出张量$y$对$L$的梯度，相当于公式中的$\frac{\partial L}{\partial y}$。</li><li>数据格式需要与'input'、'gradInput'一致。</li></ul> | FLOAT、FLOAT16、BFLOAT16 | NCL、NCHW、NCDHW |
| input | 输入 | <ul><li>公式中卷积正向的输入张量$x$。</li><li>数据格式需要与'gradOutput'、'gradInput'一致。</li></ul> | FLOAT、FLOAT16、BFLOAT16 | NCL、NCHW、NCDHW |
| weight | 输入 | <ul><li>公式中卷积核权重张量$w$。</li><li>数据格式需要与'gradWeight'一致。</li></ul> | FLOAT、FLOAT16、BFLOAT16 | NCL、NCHW、NCDHW |
| biasSizes | 输入 | <ul><li>卷积正向过程中偏差(bias)的shape，数组长度是1。</li><li>其在普通卷积中等于[weight.shape[0]],在转置卷积中等于[weight.shape[1] * groups]。</li><li>空Tensor场景下，当outputMask指定偏差的梯度需要计算时，biasSizes不能为nullptr。</li></ul> | INT64 | - |
| stride | 输入 | <ul><li>反向传播过程中卷积核在输入上移动的步长，相当于公式中的stride[0]、stride[1]、stride[2]。</li><li>数组长度为weight维度减2，数值必须大于0。</li></ul> | INT64 | - |
| padding | 输入 | <ul><li>反向传播过程中对于输入填充，相当于公式中的padding[0]、padding[1]、padding[2]。</li><li>数组长度可以为weight维度减2，在2d场景下数组长度可以为4。数值必须大于等于0。</li></ul> | INT64 | - |
| dilation | 输入 | <ul><li>反向传播过程中的膨胀参数，相当于公式中的dilation[0]、dilation[1]、dilation[2]。</li><li>数组长度可以为weight维度减2。数值必须大于0。</li></ul> | INT64 | - |
| transposed | 输入 | <ul><li>转置卷积使能标志位, 当其值为True时使能转置卷积。</li></ul> | - | - |
| outputPadding | 输入 | <ul><li>反向传播过程中对于输出填充，数组长度可以为weight维度减2，各维度的数值范围满足[0, stride对应维度数值)。transposed为False场景下，要求每个元素值为0。</li></ul> | INT64 | - |
| groups | 输入 | <ul><li>反向传播过程中输入通道的分组数。</li><li>需满足groups*weight的C维度=input的C维度，groups取值范围为[1,65535]。</li></ul> | INT32 | - |
| gradInput | 输出 | <ul><li>输入张量$x$对$L$的梯度，相当于公式中的$\frac{\partial L}{\partial x}$。</li><li>数据类型与'input'保持一致。</li><li>数据格式需要与'input'、'gradOutput'一致。</li></ul> | FLOAT、FLOAT16、BFLOAT16 | NCL、NCHW、NCDHW |
| gradWeight | 输出 | <ul><li>卷积核权重张量$w$对$L$的梯度，相当于公式中的$\frac{\partial L}{\partial w}$。</li><li>数据格式需要与'weight'一致。</li></ul> | FLOAT、FLOAT16、BFLOAT16 | NCL、NCHW、NCDHW |
| gradBias | 输出 | <ul><li>偏置$b$对$L$的梯度，相当于公式中的$\frac{\partial L}{\partial b}$。</li><li>数据类型与'gradOutput'一致。</li></ul> | FLOAT、FLOAT16、BFLOAT16 | ND |

* <term>昇腾910_95 AI处理器</term>：
    - 只有在transposed=true且output_mask[0]=true时，数据类型才支持HIFLOAT8、FLOAT8_E4M3FN。
* <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>： 
    - 不支持BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。
    - gradOutput、weight参数下，不支持空Tensor。
* <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 不支持HIFLOAT8、FLOAT8_E4M3FN。

## 约束说明  

* <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：当前仅支持1D和2D卷积的反向传播，暂不支持3D卷积的反向传播。
* 更详细的约束说明可查看[aclnnConvolutionBackward](docs/aclnnConvolutionBackward.md)接口资料。

## 调用说明

| 调用方式  | 样例代码  | 说明                 |
| -----------  | ------------------- | ---------- |
| aclnn接口   | [test_aclnn_conv_backward_1d](examples/test_aclnn_conv_backward_1d.cpp)  | 通过[aclnnConvolutionBackward](docs/aclnnConvolutionBackward.md)接口方式调用Conv3DBackpropFilterV2、Conv3DBackpropInputV2算子 |
| aclnn接口   | [test_aclnn_conv_backward_2d](examples/test_aclnn_conv_backward_2d.cpp)  | 通过[aclnnConvolutionBackward](docs/aclnnConvolutionBackward.md)接口方式调用Conv3DBackpropFilterV2、Conv3DBackpropInputV2算子 |
| aclnn接口   | [test_aclnn_conv_backward_3d](examples/test_aclnn_conv_backward_3d.cpp)  | 通过[aclnnConvolutionBackward](docs/aclnnConvolutionBackward.md)接口方式调用Conv3DBackpropFilterV2、Conv3DBackpropInputV2算子 |
