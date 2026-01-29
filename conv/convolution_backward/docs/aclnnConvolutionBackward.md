# aclnnConvolutionBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/conv/convolution_backward)

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

- 接口功能：卷积的反向传播。根据输出掩码设置计算输入、权重和偏差的梯度。此函数支持1D、2D和3D卷积。   

- 计算公式：

  输入input($N,C_{in},D_{in},H_{in},W_{in}$)、输出out($N,C_{out},D_{out},H_{out},W_{out}$)和卷积步长($stride$)、卷积核大小($kernelSize，kD,kH,kW$)、膨胀参数($dilation$)的关系是：

  $$
    D_{out}=\lfloor \frac{D_{in}+2*padding[0]-dilation[0] * (kernelSize[0] - 1) - 1}{stride[0]}+1 \rfloor
  $$

  $$
    H_{out}=\lfloor \frac{H_{in}+2*padding[1]-dilation[1] * (kernelSize[1] - 1) - 1}{stride[1]}+1 \rfloor
  $$

  $$
    W_{out}=\lfloor \frac{W_{in}+2*padding[2]-dilation[2] * (kernelSize[2] -1) -1}{stride[2]}+1 \rfloor
  $$
  
  卷积反向传播需要计算对卷积正向的输入张量 $x$（对应函数原型中的input）、卷积核权重张量 $w$ （对应函数原型中的weight）和偏置 $b$ 的梯度。  
  - 对于 $x$ 的梯度 $\frac{\partial L}{\partial x}$（对应函数原型中的gradInput参数）：
  
    $$
    \frac{\partial L}{\partial x_{n, c_{in}, i, j}} = \sum_{c_{out}=1}^{C_{out}} \sum_{p=1}^{k_H} \sum_{q=1}^{k_W} \frac{\partial L}{\partial y_{n, c_{out}, i-p, j-q}}\cdot w_{c_{out}, c_{in}, p, q}
    $$
  
    其中，$L$ 为损失函数，$\frac{\partial L}{\partial y}$ 为输出张量 $y$ 对 $L$ 的梯度（对应函数原型中的gradOutput参数）。  
  
  - 对于 $w$ 的梯度 $\frac{\partial L}{\partial w}$（对应函数原型中的gradWeight参数）：
  
    $$
    \frac{\partial L}{\partial w_{c_{out}, c_{in}, p, q}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, i \cdot s_H + p, j \cdot s_W + q} \cdot \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$
  
  - 对于 $b$ 的梯度 $\frac{\partial L}{\partial b}$（对应函数原型中的gradBias参数）：
  
    $$
    \frac{\partial L}{\partial b_{c_{out}}} = \sum_{n=1}^{N}       \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnConvolutionBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConvolutionBackward”接口执行计算。

```cpp
aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(  
    const aclTensor     *gradOutput,
    const aclTensor     *input,
    const aclTensor     *weight,
    const aclIntArray   *biasSizes,
    const aclIntArray   *stride,
    const aclIntArray   *padding,
    const aclIntArray   *dilation,
    bool                 transposed,
    const aclIntArray   *outputPadding,
    int                  groups,
    const aclBoolArray  *outputMask,
    int8_t               cubeMathType,
    aclTensor           *gradInput,
    aclTensor           *gradWeight,
    aclTensor           *gradBias,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```cpp
aclnnStatus aclnnConvolutionBackward(   
    void                *workspace,   
    uint64_t             workspaceSize,  
    aclOpExecutor       *executor,  
    const aclrtStream    stream)
```

## aclnnConvolutionBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width:170px">
    <col style="width:120px">
    <col style="width:300px">
    <col style="width:330px">
    <col style="width:212px">
    <col style="width:100px">
    <col style="width:190px">
    <col style="width:145px">
    </colgroup>
    <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
      </tr></thead>
  <tbody>
    <tr>
      <td>gradOutput</td>
      <td>输入</td>
      <td>输出张量y对L的梯度。</a></td>
      <td>  
       <ul><li>支持空Tensor。</li>
       <li>数据类型与input、weight满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推关系</a>和<a href="#约束说明" target="_blank">约束说明</a>）。</li>
       <li>shape不支持broadcast，要求和input、weight满足卷积输入输出shape的推导关系。</li>
       <li>数据格式需要与input、gradInput一致。</li>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCL、NCHW、NCDHW</td>
      <td>参见<a href="#约束说明" target="_blank">约束说明。</a></td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>公式中的x。</td>
      <td>
       <ul><li>支持空Tensor。</li>
       <li>数据类型与gradOutput、weight满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推关系</a>和<a href="#约束说明" target="_blank">约束说明</a>）。</li>
       <li>shape不支持broadcast，要求和gradOutput、weight满足卷积输入输出shape的推导关系。</li>
       <li>数据格式需要与gradOutput、gradInput一致。</li>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCL、NCHW、NCDHW</td>
      <td>参见<a href="#约束说明" target="_blank">约束说明。</a></td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>公式中的w。</td>
      <td>
       <ul><li>支持空Tensor。</li>
       <li>数据类型与gradOutput、input满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推关系</a>和<a href="#约束说明" target="_blank">约束说明</a>）。</li>
       <li>shape不支持broadcast，要求和gradOutput、input满足卷积输入输出shape的推导关系。</li>
       <li>数据格式需要与gradWeight一致。</li>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCL、NCHW、NCDHW</td>
      <td>参见<a href="#约束说明" target="_blank">约束说明。</a></td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasSizes</td>
      <td>输入</td>
      <td>卷积正向过程中偏差(bias)的shape。</td>
      <td>
       <ul><li>数组长度是1。</li>
       <li>在普通卷积中等于[weight.shape[0]]，在转置卷积中等于[weight.shape[1] * groups]。</li>
       <li>空Tensor场景下，当outputMask指定偏差的梯度需要计算时，biasSizes不能为nullptr。</li>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>反向传播过程中卷积核在输入上移动的步长。</td>
      <td>
       <ul><li>对于一维卷积反向，数组长度必须为1。</li>
       <li>数组长度为weight维度减2，数值必须大于0。</li>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>参见<a href="#约束说明" target="_blank">约束说明。</a></td>
      <td>×</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>输入</td>
      <td>反向传播过程中对于输入填充。</td>
      <td>
       <ul><li>对于一维卷积反向，数组长度必须为1。</li>
       <li>数组长度可以为weight维度减2，在2d场景下数组长度可以为4。</li>
       <li>数值必须大于等于0。</li>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>参见<a href="#约束说明" target="_blank">约束说明。</a></td>
      <td>×</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>输入</td>
      <td>反向传播过程中的膨胀参数。</td>
      <td>
       <ul><li>对于一维卷积反向，数组长度必须为1。</li>
       <li>数组长度可以为weight维度减2。</li>
       <li>数值必须大于0。</li>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>参见<a href="#约束说明" target="_blank">约束说明。</a></td>
      <td>×</td>
    </tr>
    <tr>
      <td>transposed</td>
      <td>输入</td>
      <td>转置卷积使能标志位, 当其值为True时使能转置卷积。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>outputPadding</td>
      <td>输入</td>
      <td>反向传播过程中对于输出填充。</td>
      <td>
       <ul><li>数组长度可以为weight维度减2，各维度的数值范围满足[0,stride对应维度数值)。</li>
       <li>transposed为False场景下，要求每个元素值为0。</li>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>输入</td>
      <td>反向传播过程中输入通道的分组数。</td>
      <td>
       需满足groups*weight的C维度=input的C维度，groups取值范围为[1,65535]。
      </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>outputMask</td>
      <td>输入</td>
      <td>
       <ul><li>输出掩码参数, 指定输出中是否包含输入、权重、偏差的梯度。</li>
       <li>反向传播过程输出掩码参数为True对应位置的梯度。</li>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>输入</td>
      <td>用于判断Cube单元应该使用哪种计算逻辑进行运算。</td>
      <td>
       <ul><li>如果输入的数据类型存在<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>，该参数默认对互推导后的数据类型进行处理。</li>
       <li>支持的枚举值如下：</li>
       <ul><li>0：KEEP_DTYPE，保持输入的数据类型进行计算。</li>
       <li>1：ALLOW_FP32_DOWN_PRECISION，允许将输入数据降精度计算。</li>
       <li>2：USE_FP16，允许转换为数据类型FLOAT16进行计算。当输入数据类型是FLOAT，转换为FLOAT16计算。</li>
       <li>3：USE_HF32，允许转换为数据类型HFLOAT32计算。当输入是FLOAT16，仍使用FLOAT16计算。</li>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>输入张量x对L的梯度。</td>
      <td>
       <ul><li>支持空Tensor。</li>
       <li>数据类型与input保持一致。</li>
       <li>数据格式需要与input、gradOutput一致。</li>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCL、NCHW、NCDHW</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradWeight</td>
      <td>输出</td>
      <td>卷积核权重张量w对L的梯度。</td>
      <td>
       <ul><li>支持空Tensor。</li>
       <li>数据格式需要与weight一致。</li>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16、</td>
      <td>NCL、NCHW、NCDHW</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradBias</td>
      <td>输出</td>
      <td>偏置b对L的梯度。</td>
      <td>
       <ul><li>支持空Tensor。</li>
       <li>数据类型与gradOutput一致。</li>
      </td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
   </tbody>
  </table>  

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。  

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:1050px">
    </colgroup>
   <thead>
    <tr>
     <th>返回值</th>
     <th>错误码</th>
     <th>描述</th>
   </tr>
   </thead>
   <tbody>

   <tr>
     <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
     <td rowspan="2">161001</td>
     <td>传入的gradOutput、input、weight、biasSizes、stride、padding、dilation、outputPadding、outputMask、gradInput、gradWeight是空指针。</td>
   </tr>
   <tr>
     <td>输出中包含偏差的梯度时，传入的gradBias是空指针。</td>
   </tr>
   <tr>
     <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
     <td rowspan="7">161002</td>
   </tr>
   <tr>
     <td>gradOutput、input、weight的数据类型不在支持的范围之内。</td>
   </tr>
   <tr>
     <td>gradOutput、input、weight的数据格式不在支持的范围之内。</td>
   </tr>
   <tr>
     <td>gradOutput、input、weight的shape不符合约束。</td>
   </tr>
   <tr>
     <td>biasSizes、stride、padding、dilation、outputPadding的shape不符合约束。</td>
   </tr>
   <tr>
     <td>不符合groups*weight的C维度=input的C维度。</td>
   </tr>
   <tr>
     <td>当前处理器不支持卷积反向传播。</td>
   </tr>
   <tr>
     <td>ACLNN_ERR_INNER_NULLPTR</td>
     <td>561103</td>
     <td>API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。</td>
   </tr>
   </tbody>

   </table>


## aclnnConvolutionBackward

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
       <col style="width:100px">
       <col style="width:100px">
       <col style="width:950px">
       </colgroup>
    <thead>
       <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
       <tr>
       <td>workspace</td>
       <td>输入</td>
       <td>在Device侧申请的workspace内存地址。</td>
       </tr>
       <tr>
       <td>workspaceSize</td>
       <td>输入</td>
       <td>在Device侧申请的workspace大小，由第一段接口aclnnConvolutionBackwardGetWorkspaceSize获取。</td>
       </tr>
       <tr>
       <td>executor</td>
       <td>输入</td>
       <td>op执行器，包含了算子计算流程。</td>
       </tr>       
       <tr>
       <td>stream</td>
       <td>输入</td>
       <td>指定执行任务的Stream。</td>
       </tr> 
    </tbody>
  </table>


- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
  
- 确定性计算
  - aclnnConvolutionBackward默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

  <table style="undefined;table-layout: fixed; width: 1396px"><colgroup>
  <col style="width: 168px">
  <col style="width: 404px">
  <col style="width: 429px">
  <col style="width: 395px">
  </colgroup>
   <thead>
    <tr>
     <th><term>约束类型</term></th>
     <th><term>昇腾910_95 AI处理器</term></th>
     <th><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></th>
     <th><term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term></th>
   </tr>
   </thead>
   <tbody>

   <tr>
     <th scope="row">空Tensor约束</th>
     <td colspan="2">
        <ul><li>input的维度必须是2维以上。stride、padding、dilation、outputPadding的维度必须等于input维度-2，其中padding在2D的时候可以是4维。</li>
        <li>transposed=false时，D、H、W维度约束满足公式一（见表格下方说明）。</li>
        <li>transposed=true时：</li>
          <ul><li>weight的N轴必须大于0。</li>
          <li>weight的N轴必须和input的C轴相等。</li></ul>
        </ul>
     </td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">gradOutput约束</th>
     <td>-</td>
     <td>
        <ul>1d、2d和3d transposed=false场景，各个维度的大小应该大于等于1, 当input为空Tensor时，支持N、C、D、H、W维度为0。</ul>
     </td>
     <td>
        <ul>不支持空Tensor。</ul>
     </td>
   </tr>
   <tr>
     <th scope="row">input约束</th>
     <td>
        <ul><li>transposed=true：支持N维度为0的空Tensor。当N维度为0且满足空Tensor约束时，还支持DHW维度为0。</li>
        <li>transposed=false：支持N、C维度为0的空Tensor（C维度为0时要求weight的C维度也为0）。当N或C维度为0且满足空Tensor约束时，还支持DHW维度为0。</li></ul>
     </td>
     <td>
        <ul>1d、2d和3d transposed=false场景，各个维度的大小应该大于等于1。
          <ul><li>transposed=true：支持N维度为0的空Tensor。当N维度为0且满足空Tensor约束时，还支持DHW维度为0。</li>
          <li>transposed=false：支持N、C维度为0的空Tensor（C维度为0时要求weight的C维度也为0）。当N或C维度为0且满足空Tensor约束时，还支持DHW维度为0。</li></ul>
         </ul>
     </td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">weight约束</th>
     <td>
        <ul><li>transposed=true：H、W的大小应该在[1,255]的范围内，其他维度的大小应该大于等于1。当input为空Tensor且满足空Tensor约束时，此时支持C、D、H、W轴为0。</li>
        <li>transposed=false：支持C维度等于0（C为0时要求input C维度也为0），所有维度的大小应该大于等于1。当input为空Tensor且满足空Tensor约束时，此时支持D、H、W轴为0。</li></ul>
     </td>
     <td>
        <ul>2d和3d transposed=false场景，H、W的大小应该在[1,255]的范围内，其他维度的大小应该大于等于1。1d transposed=false场景，L的大小应该在[1,255]的范围内，其他维度的大小应该大于等于1。
           <ul><li>transposed=true: 当input为空Tensor且满足空Tensor约束时，此时支持C、D、H、W轴为0。</li>
           <li>transposed=false：支持C维度等于0（C为0时要求input C维度也为0），当input为空Tensor且满足空Tensor约束时，此时支持D、H、W轴为0。</li></ul>
        </ul>
     </td>
     <td>
        <ul>不支持空Tensor。</ul>
     </td>
   </tr>
   <tr>
     <th scope="row">stride约束</th>
     <td>
        <ul>1d、2d和3d transposed=false场景，各个值都应该大于等于1。1d、2d和3d transposed=true场景，strideD应该在[1,1000000]的范围内，strideH、strideW应该在[1,63]的范围内。</ul>
     </td>
     <td>
        <ul>3d transposed=false场景，strideD应该大于等于1，strideH、strideW应该在[1,63]的范围内。1d和2d transposed=false场景，各个值都应该大于等于1。</ul>
     </td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">padding约束</th>
     <td>
        <ul>1d、2d和3d transposed=false场景，各个值都应该大于等于0。1d、2d和3d transposed=true场景，paddingD应该在[0,1000000]的范围内，paddingH、paddingW应该在[0,255]的范围内。</ul>
     </td>
     <td>
        <ul>3d transposed=false场景，paddingD应该大于等于0，paddingH、paddingW应该在[0,255]的范围内。1d和2d transposed=false场景，各个值都应该在[0,255]的范围内。</ul>
     </td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">dilation约束</th>
     <td>
        <ul>1d、2d和3d transposed=false场景，各个值都应该大于等于1。1d、2d和3d transposed=true场景，dilationD应该在[1,1000000]的范围内，dilationH、dilationW应该在[1,255]的范围内。</ul>
     </td>
     <td>
        <ul>1d、2d和3d transposed=false场景，各个值都应该在[1,255]的范围内。</ul>
     </td>
     <td>-</td>
   </tr>
   <tr>
     <th scope="row">dtype约束</th>
     <td>
        <ul>只有在transposed=true且output_mask[0]=true时，数据类型才支持HIFLOAT8、FLOAT8_E4M3FN。
        </ul>
     </td>
     <td>
        <ul>不支持HIFLOAT8、FLOAT8_E4M3FN。</ul>
     </td>
     <td>
        <ul>不支持BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。</ul>
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType说明</th>
     <td colspan="2">
        <ul><li>枚举值为0：暂无说明。</li>
        <li>枚举值为1：当输入是FLOAT，转换为HFLOAT32计算。当输入为其他数据类型时不做处理。</li>
        <li>枚举值为2：当输入是BFLOAT16时不支持该选项。当输入为其他数据类型时不做处理。</li>
        <li>枚举值为3：当输入是FLOAT，转换为HFLOAT32计算。当输入为其他数据类型时不做处理。</li>
        </ul>
     </td>
     <td>
        <ul><li>枚举值为0：当输入是FLOAT，Cube计算单元暂不支持，取0时会报错。</li>
        <li>枚举值为1：当输入是FLOAT，转换为FLOAT16计算。当输入为其他数据类型时不做处理。</li>
        <li>枚举值为2：暂无说明。</li>
        <li>枚举值为3：当输入是FLOAT，Cube计算单元暂不支持。</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">其他约束</th>
     <td>-</td>
     <td>-</td>
     <td>
        <ul>当前仅支持1D和2D卷积的反向传播，暂不支持3D卷积的反向传播。</ul>
     </td>
   </tr>
   </tbody>
  </table>
    
  - 公式一：
    $$
        (input_{dim} + pad_{dim} \times 2) \ge ((weight_{dim}  - 1) \times dilation_{dim} + 1)
    $$

由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击[Link](https://www.hiascend.com/support)获取技术支持。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_convolution_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)        \
    do {                                         \
        if (!(cond)) {                           \
            Finalize(deviceId, stream); \
            return_expr;                         \
        }                                        \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **DeviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(DeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*DeviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    if (shape.size() == 4) {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                  shape.data(), shape.size(), *DeviceAddr);
    } else {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *DeviceAddr);
    }

    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnConvolutionBackwardTest(int32_t deviceId, aclrtStream &stream)
{
    // 1. 初始化
    auto ret = Init(deviceId, &stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradOutputShape = {2, 2, 7, 7};
    std::vector<int64_t> inputShape = {2, 2, 7, 7};
    std::vector<int64_t> weightShape = {2, 2, 1, 1};
    std::vector<int64_t> biasSize = {2};
    std::vector<int64_t> stride = {1, 1};
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    bool transposed = false;
    std::vector<int64_t> outputPadding = {0, 0};
    int groups = 1;
    bool outputMask[3] = {true, true, true};
    int8_t cubeMathType = 1;

    std::vector<int64_t> gradInputShape = {2, 2, 7, 7};
    std::vector<int64_t> gradWeightShape = {2, 2, 1, 1};
    std::vector<int64_t> gradBiasShape = {2};

    // 创建gradOutput aclTensor
    std::vector<float> gradOutputData(GetShapeSize(gradOutputShape), 1);
    aclTensor *gradOutput = nullptr;
    void *gradOutputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradOutputData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradOutputTensorPtr(gradOutput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradOutputDeviceAddrPtr(gradOutputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建input aclTensor
    std::vector<float> inputData(GetShapeSize(inputShape), 1);
    aclTensor *input = nullptr;
    void *inputDeviceAddr = nullptr;
    ret = CreateAclTensor(inputData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> inputDeviceAddrPtr(inputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建weight aclTensor
    std::vector<float> weightData(GetShapeSize(weightShape), 1);
    aclTensor *weight = nullptr;
    void *weightDeviceAddr = nullptr;
    ret = CreateAclTensor(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradInput aclTensor
    std::vector<float> gradInputData(GetShapeSize(inputShape), 1);
    aclTensor *gradInput = nullptr;
    void *gradInputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradInputData, inputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradInputTensorPtr(gradInput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradInputDeviceAddrPtr(gradInputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradWeight aclTensor
    std::vector<float> gradWeightData(GetShapeSize(weightShape), 1);
    aclTensor *gradWeight = nullptr;
    void *gradWeightDeviceAddr = nullptr;
    ret = CreateAclTensor(gradWeightData, weightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradWeightTensorPtr(gradWeight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradWeightDeviceAddrPtr(gradWeightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradBias aclTensor
    std::vector<float> gradBiasData(GetShapeSize(biasSize), 1);
    aclTensor *gradBias = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    ret = CreateAclTensor(gradBiasData, biasSize, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradBiasTensorPtr(gradBias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradBiasDeviceAddrPtr(gradBiasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建biasSizes aclIntArray
    aclIntArray *biasSizes = aclCreateIntArray(biasSize.data(), 1);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> biasSizesPtr(biasSizes, aclDestroyIntArray);
    CHECK_FREE_RET(biasSizes != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 创建strides aclIntArray
    aclIntArray *strides = aclCreateIntArray(stride.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
    CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 创建pads aclIntArray
    aclIntArray *pads = aclCreateIntArray(padding.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
    CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 创建dilations aclIntArray
    aclIntArray *dilations = aclCreateIntArray(dilation.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
    CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 创建outputPads aclIntArray
    aclIntArray *outputPads = aclCreateIntArray(outputPadding.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outputPadsPtr(outputPads, aclDestroyIntArray);
    CHECK_FREE_RET(outputPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 创建outMask aclBoolArray
    aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);
    std::unique_ptr<aclBoolArray, aclnnStatus (*)(const aclBoolArray *)> outMaskPtr(outMask, aclDestroyBoolArray);
    CHECK_FREE_RET(outMask != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnConvolutionBackwardGetWorkspaceSize第一段接口
    ret = aclnnConvolutionBackwardGetWorkspaceSize(gradOutput, input, weight, biasSizes, strides, pads, dilations,
                                                   transposed, outputPads, groups, outMask, cubeMathType, gradInput,
                                                   gradWeight, gradBias, &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
                   return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnConvolutionBackward第二段接口
    ret = aclnnConvolutionBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackward failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> gradInputResult(size, 0);
    ret = aclrtMemcpy(gradInputResult.data(), gradInputResult.size() * sizeof(gradInputResult[0]), gradInputDeviceAddr,
                      size * sizeof(gradInputResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradInputResult[%ld] is: %f\n", i, gradInputResult[i]);
    }

    size = GetShapeSize(gradWeightShape);
    std::vector<float> gradWeightResult(size, 0);
    ret = aclrtMemcpy(gradWeightResult.data(), gradWeightResult.size() * sizeof(gradWeightResult[0]), gradWeightDeviceAddr,
                      size * sizeof(gradWeightResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradWeightResult[%ld] is: %f\n", i, gradWeightResult[i]);
    }

    size = GetShapeSize(gradBiasShape);
    std::vector<float> gradBiasResult(size, 0);
    ret = aclrtMemcpy(gradBiasResult.data(), gradBiasResult.size() * sizeof(gradBiasResult[0]), gradBiasDeviceAddr,
                      size * sizeof(gradBiasResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradBiasResult[%ld] is: %f\n", i, gradBiasResult[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnConvolutionBackwardTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionBackwardTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
