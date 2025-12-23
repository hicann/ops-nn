# Conv3DBackpropFilterV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾 950 AI 处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：计算三维卷积核权重张量$w$的梯度 $\frac{\partial L}{\partial w}$。

- 计算公式：

  $$
  \frac{\partial L}{\partial w_{c_{out}, c_{in}, r, p, q}} = \sum_{n=1}^{N} \sum_{d=1}^{D_{out}} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, d \cdot s_D + r, i \cdot s_H + p, j \cdot s_W + q} \cdot \frac{\partial L}{\partial y_{n, c_{out}, d, i, j}}
  $$

## 参数说明

| <div style="width:120px">参数名</div>  | <div style="width:120px">输入/输出/属性</div>  | <div style="width:350px">描述</div> | <div style="width:350px">数据类型</div>  | <div style="width:220px">数据格式</div> |
| ------------------| ------------------ | ------------------------------------------------------------------------------------------- | ----------------- | --------------------- |
| x | 输入 | <ul><li>一个5D张量，为特征图，相当于公式中的$x_{n, c_{in}, d \cdot s_D + r, i \cdot s_H + p, j \cdot s_W + q}$。</li><li>$N$相当于'x'的N维度 。</li><li>$c_{in}$相当于'x'的C维度。</li><li>数据格式需要与'out_backprop'一致。</li></ul> | FLOAT16、FLOAT32、BFLOAT16 | NDHWC、NCDHW |
| filter_size | 输入 | <ul><li>一个1D张量，用于表示滤波器的形状，该滤波器是一个5D张量。</li><li>张量中整数的顺序由滤波器格式决定，这些整数表示滤波器各维度的长度。</li></ul> | INT32、INT64 | - | 
| out_backprop | 输入 | <ul><li>一个5D张量，相当于公式中的$\frac{\partial L}{\partial y_{n, c_{out}, d, i, j}}$。</li><li>$c_{out}$相当于'out_backprop'中的C维度。</li><li>$H_{out}$、$W_{out}$、$D_{out}$相当于'out_backprop'中的H维度、W维度和D维度。</li><li>数据格式需要与'x'一致。</li></ul> | FLOAT16、FLOAT32、BFLOAT16 | NDHWC、NCDHW |
| strides | 必填属性 | <ul><li>一个包含5个整数的元组或列表，指定滑动窗口在特征图每个维度上的步长。</li><li>轴顺序与特征图格式一致。</li><li>$s_H$与$s_W$、$s_D$相当于'strides'属性的H维度、W维度和D维度。</li></ul> | - | - |
| pads | 必填属性 | <ul><li>一个包含6个整数的元组/列表，指定特征图在每个方向上的填充量。</li><li>仅沿D、H和W维度进行填充。</li><li>通过为各方向设置合适的填充值，即可实现 “SAME” 和 “VALID” 两种填充模式。</li></ul> | - | - |
| dilations | 可选属性 | <ul><li>一个包含5个整数的元组/列表，表示输入各维度的膨胀（空洞）因子。</li><li>轴顺序与特征图格式一致。</li><li>默认值为 [1,1,1,1,1]。</li></ul> | -  | -  |
| groups | 可选属性  |  <ul><li>整数，有效范围是[1,65535]，默认值为1。</li><li>表示从$c_{in}$到$c_{out}$的分组连接数。</li><li>目前要求$c_{in}$与$c_{out}$均能被'groups'整除。</li><li>当'groups'不为1时，不支持HIFLOAT8。</li></ul>  | INT | - |
| data_format | 可选属性 | <ul><li>字符串，取值必须为["NDHWC","NCDHW"]之一，默认"NDHWC"。各字母含义：batch(N)、depth(D)、height(H)、width(W)、channels(C)。</li><li>用于指定'x'与'out_backprop'的数据排布格式。</li></ul> | STRING | - |
| y | 输出 | <ul><li>相当于公式中的$\frac{\partial L}{\partial w_{c_{out}, c_{in}, r, p, q}}$。</li><li>其形状由'filter_size'给出。</li></ul>  |  FLOAT32  | NCDHW、NDHWC、DHWCN |

## 约束说明

* filter_size
    - W、H维度的取值范围必须在 [1,511] 之间。
    - 可输入的轴序列如下：
        - [out_channels, in_channels, filter_depth, filter_height, filter_width]
        - [out_channels, filter_depth, filter_height, filter_width, in_channels]
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
| aclnn接口   | [test_aclnn_conv3d_backprop_filter_v2](examples/test_aclnn_conv3d_backprop_filter_v2.cpp)  | 通过[aclnnConvolutionBackward](../convolution_backward/docs/aclnnConvolutionBackward.md)接口方式调用Conv3DBackpropFilterV2算子 |

