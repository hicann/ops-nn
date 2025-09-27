# Conv3DTransposeV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：计算三维卷积的转置（反卷积）相对于输入的梯度。

- 计算公式：

  假定输入的shape为($N,C_{in},D_{in},H_{in},W_{in}$)、输出的shape为($N,C_{out},D_{out},H_{out},W_{out}$)，那么他们与卷积步长($stride$)、卷积核大小($kernelSize，kD,kH,kW$)、膨胀参数($dilation$)的关系是：

  $$
    D_{out}=\lfloor \frac{D_{in}+2*padding[0]-dilation[0] * (kernelSize[0] - 1) - 1}{stride[0]}+1 \rfloor
  $$

  $$
    H_{out}=\lfloor \frac{H_{in}+2*padding[1]-dilation[1] * (kernelSize[1] - 1) - 1}{stride[1]}+1 \rfloor
  $$

  $$
    W_{out}=\lfloor \frac{W_{in}+2*padding[2]-dilation[2] * (kernelSize[2] -1) -1}{stride[2]}+1 \rfloor
  $$

## 参数说明

| <div style="width:120px">参数名</div>  | <div style="width:120px">输入/输出/属性</div>  | <div style="width:380px">描述</div> | <div style="width:350px">数据类型</div>  | <div style="width:220px">数据格式</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------------------- | ----------------- | --------------------- |
| input_size | 输入  | <ul><li>1D张量，用于表示输入张量的形状。</li></ul> | INT32、INT64 | - |
| x  | 输入 | <ul><li>输入特征图，相当于公式中的($N,C_{in},D_{in},H_{in},W_{in}$)。</li><li>一个张量，数据格式与'y'一致。</li></ul> | FLOAT16、FLOAT32、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN| NDHWC、NCDHW |
| filter | 输入 | <ul><li>一个5D张量，代表卷积核尺寸，相当于公式中的kernelSize[0]、kernelSize[1]、kernelSize[2]。</li></ul> | FLOAT16、BFLOAT16、FLOAT32、HIFLOAT8、FLOAT8_E4M3FN | NCDHW、NDHWC、DHWCN |
| bias | 可选属性 | <ul><li>一个1D张量，当前未启用。</li></ul> | FLOAT16、FLOAT32 | - |
| offset_w | 可选属性 | <ul><li>一个1D张量，用于量化反卷积，当前未启用。</li></ul> | INT8 | - |
| strides | 必填属性 | <ul><li>一个包含5个整数的元组或列表，用于指定滑动窗口在输入张量'x'每个维度上的步长。</li><li>轴顺序与特征图格式一致。</li></ul>  | - | - |
| pads | 必填属性 | <ul><li>一个包含6个整数的元组或列表，用于指定各方向的填充量，相当于公式中的padding[0]、padding[1]、padding[2]。</li></ul> | - | - |
| dilations | 可选属性 | <ul><li>一个包含5个整数的元组或列表，表示输入各维度的膨胀（空洞）因子，相当于公式中的dilation[0]、dilation[1]、dilation[2]</li><li>默认值为[1,1,1,1,1]。</li><li>轴顺序与特征图格式一致。</li></ul>| - | - |
| groups | 可选属性 | <ul><li>整数，范围为[1,65535]，默认1。</li><li>表示从$c_{in}$到$c_{out}$的分组连接数。</li><li>$c_{in}$与$c_{out}$必须能被'groups'整除。</li><li>不同'groups'与dtype的组合支持见下方表格说明。</li></ul> | - | - |
| data_format | 可选属性 | <ul><li>字符串，取值必须为["NDHWC","NCDHW"]之一，默认"NDHWC"。对应关系为：batch(N)、depth(D)、height(H)、width(W)、channels(C)。</li><li>指定'x'与'y'的数据排布格式。</li></ul> | - | - |
| output_padding | 可选属性 | <ul><li>将在输出形状末尾额外增加的尺寸，默认值为[0,0,0,0,0]。</li></ul> | - | - |
| offset_x  | 可选属性 | <ul><li>默认值为 0，保留字段。</li></ul> | - | - |
| y | 输出 | <ul><li>相当于公式中的($N,C_{out},D_{out},H_{out},W_{out}$)。</li><li>数据格式与'x'一致。</li></ul> | FLOAT16、BFLOAT16、FLOAT32、HIFLOAT8、FLOAT8_E4M3FN | NDHWC、NCDHW |

- 不同groups取值与dtype的组合说明

    | groups |        dtype           |   x format  | filter format |    y format    |
    |--------|------------------------|-------------|---------------|----------------|
    |  =1    |hifloat8/float8_e4m3fn  |    NCDHW    |      NCDHW    |     NCDHW      |
    |  =1    |hifloat8/float8_e4m3fn  |    NDHWC    |      NCDHW    |     NDHWC      |
    |  =1    |float16/bfloat16/float32|    NCDHW    |      NCDHW    |     NCDHW      |
    |  =1    |float16/bfloat16/float32|    NCDHW    |      NDHWC    |     NCDHW      |
    |  =1    |float16/bfloat16/float32|    NCDHW    |      DHWCN    |     NCDHW      |
    |  =1    |float16/bfloat16/float32|    NDHWC    |      NDHWC    |     NDHWC      |
    |  =1    |float16/bfloat16/float32|    NDHWC    |      NCDHW    |     NDHWC      |
    |  =1    |float16/bfloat16/float32|    NDHWC    |      DHWCN    |     NDHWC      |
    |  >1    |hifloat8/float8_e4m3fn  |    NCDHW    |      NCDHW    |     NCDHW      |
    |  >1    |float16/bfloat16/float32|    NCDHW    |      NCDHW    |     NCDHW      |
    |  >1    |float16/bfloat16/float32|    NCDHW    |      NDHWC    |     NCDHW      |
    |  >1    |float16/bfloat16/float32|    NCDHW    |      DHWCN    |     NCDHW      |


## 约束说明

* filter
    - 当数据格式为DHWCN或NDHWC时，不支持HIFLOAT8、FLOAT8_E4M3FN。
    - 高度(H)与宽度(W)维长度必须在[1,511]范围内。
* strides
    - N和C的维度必须为1。
    - H和W的维度的取值范围必须在[1,63]之间。
    - D维度的取值范围必须在[1,255]之间。
* pads
    - 所有维度的取值范围必须在[0,255]之间。
* dilations
    - N与C的维度必须为1。
    - W、H和D维度的取值范围必须在 [1,255] 之间。
* output_padding 
    -  N和C维度必须为0，仅允许在深度、高度、宽度方向添加。


## 调用说明

| 调用方式  | 样例代码  | 说明                 |
| -----------  | ------------------- | ---------- |
| aclnn接口   | [test_aclnn_conv3d_transpose_v2](examples/test_aclnn_conv3d_transpose_v2.cpp)  | 通过[aclnnConvolution](../convolution_forward/docs/aclnnConvolution.md)接口方式调用Conv3DTransposeV2算子 |