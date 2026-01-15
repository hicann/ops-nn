# Conv3DBackpropInputV2

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：计算三维卷积正向的输入张量$x$对损失函数$L$的梯度 $\frac{\partial L}{\partial x}$。

- 计算公式：

  $$
    \frac{\partial L}{\partial x_{n, c_{in}, d, i, j}} = \sum_{c_{out}=1}^{C_{out}} \sum_{r=1}^{k_D} \sum_{p=1}^{k_H} \sum_{q=1}^{k_W} \frac{\partial L}{\partial y_{n, c_{out}, d-r, i-p, j-q}}\cdot w_{c_{out}, c_{in}, r, p, q}
  $$

## 参数说明

| <div style="width:120px">参数名</div>  | <div style="width:120px">输入/输出/属性</div>  | <div style="width:380px">描述</div> | <div style="width:350px">数据类型</div>  | <div style="width:220px">数据格式</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------ | ----------------- | --------------------- |
| input_size | 输入 | <ul><li>一维张量，给出待求梯度特征图的5D形状。</li><li>张量中整数的顺序由特征图格式决定，整数表示特征图各维度的长度。</li></ul> | INT32、INT64 | - |
| filter | 输入   | <ul><li>5D卷积核张量$w_{c_{out}, c_{in}, r, p, q}$。</li><li>$c_{out}$相当于'filter'的N维度。</li><li>$k_H$、$k_W$、$k_D$相当于'filter'的H维度、W维度和D维度。</li></ul> | FLOAT16、BFLOAT16、FLOAT32 | NCDHW、NDHWC、DHWCN |
| out_backprop | 输入 | <ul><li>5D输出梯度张量，等同于公式中的$\frac{\partial L}{\partial y_{n, c_{out}, d-r, i-p, j-q}}$。</li><li>数据格式与'y'一致。</li></ul> | FLOAT16、FLOAT32、BFLOAT16 | NDHWC、NCDHW |
| strides | 必填属性 | <ul><li>一个包含5个整数的元组或列表，用于指定滑动窗口在特征图每个维度上的步长。</li><li>轴顺序与特征图格式一致。</li></ul> | - | - |
| pads | 必填属性 | <ul><li>一个包含 6 个整数的元组或列表，用于指定特征图在各个方向上的填充量。</li><li>仅沿深度（D）、高度（H）和宽度（W）维度进行填充。</li><li>通过为各方向设置合适的填充值，可实现“SAME”和“VALID”两种填充模式。</li></ul> | - | - |
| dilations  | 可选属性 | <ul><li>一个包含5个整数的元组或列表，表示特征图各维度的膨胀（空洞）因子。</li><li>默认值为[1,1,1,1,1]。</li><li>轴顺序与特征图格式一致。</li></ul> | - | - |
| groups  | 可选属性 | <ul><li>整数，范围为[1,65535]，默认1。</li><li>表示从$c_{in}$到$c_{out}$的分组连接数。</li><li>$c_{in}$与$c_{out}$必须能被'groups'整除。</li><li>不同'groups'与dtype的组合支持见下方表格说明。</li></ul> | INT | - |
| data_format  | 可选属性 | <ul><li>字符串，取值必须为["NDHWC","NCDHW"]之一，默认"NDHWC"。对应关系为：batch(N)、depth(D)、height(H)、width(W)、channels(C)。</li><li>指定'out_backprop'与'y'的数据排布格式。</li></ul> | STRING | - |
| y  | 输出 | <ul><li>与'out_backprop'同格式的5D张量，即公式中的$\frac{\partial L}{\partial x_{n, c_{in}, d, i, j}}$。</li></ul> | FLOAT16、BFLOAT16、FLOAT32 | NDHWC、NCDHW |

- 特征图：正向卷积中的[输入特征图'x'](../conv3d_v2/README.md)。
- <term>Ascend 950PR/Ascend 950DT处理器</term>：当'out_backprop'和'y'数据格式为NDHWC，暂不支持groups>1。
- 不同groups取值与dtype的组合说明

    | groups |        dtype           | out_backprop format | filter format |     y format   |
    |--------|------------------------|---------------------|---------------|----------------|
    |  =1    |HIFLOAT8/FLOAT8_E4M3FN  |       NCDHW         |      NCDHW    |     NCDHW      |
    |  =1    |HIFLOAT8/FLOAT8_E4M3FN  |       NDHWC         |      NCDHW    |     NDHWC      |
    |  =1    |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      NCDHW    |     NCDHW      |
    |  =1    |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      NDHWC    |     NCDHW      |
    |  =1    |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      DHWCN    |     NCDHW      |
    |  =1    |FLOAT16/BFLOAT16/FLOAT32|       NDHWC         |      NDHWC    |     NDHWC      |
    |  =1    |FLOAT16/BFLOAT16/FLOAT32|       NDHWC         |      NCDHW    |     NDHWC      |
    |  =1    |FLOAT16/BFLOAT16/FLOAT32|       NDHWC         |      DHWCN    |     NDHWC      |
    |  >1    |HIFLOAT8/FLOAT8_E4M3FN  |       NCDHW         |      NCDHW    |     NCDHW      |
    |  >1    |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      NCDHW    |     NCDHW      |
    |  >1    |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      NDHWC    |     NCDHW      |
    |  >1    |FLOAT16/BFLOAT16/FLOAT32|       NCDHW         |      DHWCN    |     NCDHW      |

## 约束说明

* input_size 
    - 可输入的轴序列如下：
        - [batch, in_depth, in_height, in_width, in_channels]
        - [batch, in_channels, in_depth, in_height, in_width]
* filter
    - kernel_height(H)与kernel_width(W)维长度须在[1,511]范围内。
* strides
    - N和C的维度必须为1。
    - H和W的维度的取值范围必须在 [1,63] 之间。
    - D维度的取值范围必须在 [1,255] 之间。
* pads
    - 填充顺序为：[front, back, top, bottom, left, right]。
    - H、W和D维度的取值范围必须在 [0,255] 之间。
* dilations
    - N与C的维度必须为1。
    - W、H和D维度的取值范围必须在 [1,255] 之间。


## 调用说明

| 调用方式  | 样例代码  | 说明                 |
| -----------  | ------------------- | ---------- |
| aclnn接口   | [test_aclnn_conv3d_backprop_input_v2](examples/test_aclnn_conv3d_backprop_input_v2.cpp)  | 通过[aclnnConvolutionBackward](../convolution_backward/docs/aclnnConvolutionBackward.md)接口方式调用Conv3DBackpropInputV2算子 |