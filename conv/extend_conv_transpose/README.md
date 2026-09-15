# ExtendConvTranspose

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                   |    ×     |
| <term>Atlas 推理系列产品</term>                           |    ×     |
| <term>Atlas 训练系列产品</term>                           |    ×     |

## 功能说明

- 算子功能：计算三维转置卷积（反卷积），用于量化推理场景。

- 计算公式：

  假定输入的shape为($N,C_{in},D_{in},H_{in},W_{in}$)、输出的shape为($N,C_{out},D_{out},H_{out},W_{out}$)，那么它们与卷积步长($stride$)、填充($pads$)、卷积核大小($kernel\_size$, kD, kH, kW)、膨胀参数($dilation$)的关系是：

  $$
    D_{out}=(D_{in} - 1) * stride[0] - (pads[0] + pads[1]) + dilation[0] * (kernel\_size[0] - 1) + output\_padding[0] + 1
  $$

  $$
    H_{out}=(H_{in} - 1) * stride[1] - (pads[2] + pads[3]) + dilation[1] * (kernel\_size[1] - 1) + output\_padding[1] + 1
  $$

  $$
    W_{out}=(W_{in} - 1) * stride[2] - (pads[4] + pads[5]) + dilation[2] * (kernel\_size[2] - 1) + output\_padding[2] + 1
  $$

## 参数说明

| <div style="width:120px">参数名</div>  | <div style="width:120px">输入/输出/属性</div>  | <div style="width:380px">描述</div> | <div style="width:350px">数据类型</div>             | <div style="width:220px">数据格式</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------------------ |-------------------------------------------------|-------------------------------------|
| input_size | 输入  | <ul><li>1D张量，用于表示输入张量的形状，即卷积正向的输入（本算子输出'y'）的形状。</li></ul> | INT32、INT64                                     | -                                   |
| x  | 输入 | <ul><li>特征图，相当于公式中的($N,C_{in},D_{in},H_{in},W_{in}$)。</li><li>一个张量，数据格式与'y'一致。</li></ul> | INT8                   | NCDHW                         |
| filter | 输入 | <ul><li>一个5D张量，表示卷积核权重，形状为($C_{out}$, $C_{in}$/groups, kD, kH, kW)，其中kD、kH、kW相当于公式中的kernelSize[0]、kernelSize[1]、kernelSize[2]。</li></ul> | INT8                   | NCDHW                         |
| bias | 可选输入 | <ul><li>一个1D张量，卷积偏置张量，按$out\_channels$顺序存储。</li></ul> | INT32                           | ND                                  |
| scale | 可选输入 | <ul><li>一个1D张量，与输出对应的通道维量化/反量化参数（channelwise）。</li></ul> | UINT64                                          | ND                                  |
| strides | 必填属性 | <ul><li>一个包含5个整数的元组或列表，用于指定滑动窗口在输入张量'x'每个维度上的步长。</li><li>轴顺序与特征图格式一致。</li></ul> | -                                               | -                                   |
| pads | 必填属性 | <ul><li>一个包含6个整数的元组或列表，用于指定各方向的填充量，相当于公式中的pads[0]～pads[5]。</li></ul> | -                                               | -                                   |
| dilations | 可选属性 | <ul><li>一个包含5个整数的元组或列表，表示输入各维度的膨胀（空洞）因子，相当于公式中的dilation[0]、dilation[1]、dilation[2]。</li><li>默认值为[1,1,1,1,1]。</li><li>轴顺序与特征图格式一致。</li></ul>| -                                               | -                                   |
| groups | 可选属性 | <ul><li>整数，范围为[1,65535]，默认1。</li><li>表示从$c_{in}$到$c_{out}$的分组连接数。</li><li>$c_{in}$与$c_{out}$必须能被'groups'整除。</li></ul> | INT                                             | -                                   |
| data_format | 可选属性 | <ul><li>字符串，当前仅支持取值"NCDHW"。对应关系为：batch(N)、channels(C)、depth(D)、height(H)、width(W)。</li><li>指定'x'与'y'的数据排布格式。</li></ul> | STRING                                          | -                                   |
| output_padding | 可选属性 | <ul><li>将在输出形状末尾额外增加的尺寸，默认值为[0,0,0,0,0]。</li><li>相当于公式中的output_padding[0]、output_padding[1]、output_padding[2]。</li></ul> | -                                   | -                                   |
| offset_x  | 可选属性 | <ul><li>默认值为0，保留字段。</li></ul> | INT                                             | -                                   |
| fusion_mode | 可选属性 | <ul><li>整数，取值为0或1，默认0。表示输出是否使能ReLU：0表示不使能，1表示使能。</li></ul> | INT                                             | -                                   |
| y_quant_mode | 可选属性 | <ul><li>默认值为0，保留字段。</li></ul> | INT                                             | -                                   |
| y | 输出 | <ul><li>相当于公式中的($N,C_{out},D_{out},H_{out},W_{out}$)。</li><li>数据格式与'x'一致。</li></ul> | FLOAT16、INT8 | NCDHW                         |

## 约束说明
* input_size
    - 可输入的轴序列如下：
        - [batch, in_channels, in_depth, in_height, in_width]
* x
    - N、C、D、H和W维度的取值范围必须在 [1,2147483647] 之间。
* filter
    - N(out_channels)、C(in_channels/groups)和D维度的取值范围必须在 [1,2147483647] 之间。
    - kernel_height(H)与kernel_width(W)维长度须在 [1,511] 范围内。
* strides
    - N和C的维度必须为1。
    - D、H、W维度取值范围必须在 [1,2147483647] 之间。
* pads
    - 填充顺序为：[front, back, top, bottom, left, right]。
    - D、H和W维度的取值范围必须在 [0,2147483647] 之间。
* dilations
    - N与C的维度必须为1。
    - D、H和W维度的取值范围必须在 [1,2147483647] 之间。
* output_padding
    - N和C维度必须为0，仅允许在深度、高度、宽度方向上添加。
    - D、H和W维度的取值范围必须在 [0,2147483647] 之间。
* 由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击 [Link](https://www.hiascend.com/support)获取技术支持。

## 调用说明
该算子无对应的aclnn接口，仅在图模式调用Conv2DTranspose算子且输入类型为INT8时，会转为ExtendConvTranspose进行量化相关实现。
