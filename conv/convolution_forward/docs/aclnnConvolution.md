# aclnnConvolution

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/conv/convolution_forward)

## 产品支持情况

<table>
<tr>
<th style="text-align:left">产品</th>
<th style="text-align:center; width:100px">是否支持</th>
</tr>
<tr>
<td><term>昇腾910_95 AI处理器</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas 200I/500 A2 推理产品</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas 推理系列产品 </term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas 训练系列产品</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas 200/300/500 推理产品</term></td>
<td style="text-align:center">×</td>
</tr>
</table>

## 功能说明

- 接口功能：实现卷积功能，支持 1D 卷积、2D 卷积、3D 卷积，同时支持转置卷积、空洞卷积、分组卷积。
  对于入参 `transposed = True` 时，表示使用转置卷积或者分数步长卷积。它可以看作是普通卷积的梯度或者逆向操作，即从卷积的输出形状恢复到输入形状，同时保持与卷积相容的连接模式。它的参数和普通卷积类似，包括输入通道数、输出通道数、卷积核大小、步长、填充、输出填充、分组、偏置、扩张等。

- 计算公式：

  我们假定输入（input）的 shape 是 $(N, C_{\text{in}}, D, H, W)$，(weight) 的 shape 是 $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$，输出（output）的 shape 是 $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$，其中 $N$ 表示批次大小（batch size），$C$ 是通道数，$D$、$H$ 和 $W$ 分别是样本的深度、高度和宽度，$K_d$、$K_h$ 和 $K_w$ 分别是卷积核的深度、高度和宽度，那输出将被表示为：

  $$
  \text{output}(N_i, C_{\text{out}_j}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}) = \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k) + \text{bias}(C_{\text{out}_j})
  $$

  其中，$\star$ 表示卷积计算，根据卷积输入的维度，卷积的类型（空洞卷积、分组卷积）而定。$N$ 代表批次大小（batch size），$C$ 代表通道数，$D$、$H$ 和 $W$ 分别代表深度、高度和宽度，相应输出维度的计算公式如下：

  - 对于入参 `transposed = False` 时：
  $$
  D_{\text{out}}=[(D + 2 \times padding[0] - dilation[0] \times (K_d - 1) - 1 ) / stride[0]] + 1 \\
  H_{\text{out}}=[(H + 2 \times padding[1] - dilation[1] \times (K_h - 1) - 1 ) / stride[1]] + 1 \\
  W_{\text{out}}=[(W + 2 \times padding[2] - dilation[2] \times (K_w - 1) - 1 ) / stride[2]] + 1
  $$

  - 对于入参 `transposed = True` 时：
  $$
  D_{\text{out}}=(D - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
            \times (K_d - 1) + \text {outputPadding}[0] + 1 \\
  H_{\text{out}}=(H - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
            \times (K_h - 1) + \text {outputPadding}[1] + 1 \\
  W_{\text{out}}=(W - 1) \times \text{stride}[2] - 2 \times \text{padding}[2] + \text{dilation}[2]
            \times (K_w - 1) + \text {outputPadding}[2] + 1
  $$

## 函数原型

每个算子分为<a href="../../../docs/zh/context/两段式接口.md">两段式接口</a>，必须先调用 aclnnConvolutionGetWorkspaceSize 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 aclnnConvolution 接口执行计算。

```cpp
aclnnStatus aclnnConvolutionGetWorkspaceSize(
    const aclTensor       *input,
    const aclTensor       *weight,
    const aclTensor       *bias,
    const aclIntArray     *stride,
    const aclIntArray     *padding,
    const aclIntArray     *dilation,
    bool                   transposed,
    const aclIntArray     *outputPadding,
    const int64_t          groups,
    aclTensor             *output,
    int8_t                 cubeMathType,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnConvolution(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnConvolutionGetWorkspaceSize

- **参数说明：**

  <table>
  <tr>
  <th style="width:170px">参数名</th>
  <th style="width:120px">输入/输出</th>
  <th style="width:300px">描述</th>
  <th style="width:400px">使用说明</th>
  <th style="width:212px">数据类型</th>
  <th style="width:100px">数据格式</th>
  <th style="width:100px">维度（shape）</th>
  <th style="width:145px">非连续 Tensor</th>
  </tr>
  <tr>
  <td>input</td>
  <td>输入</td>
  <td>公式中的 input，表示卷积输入。</td>
  <td><ul><li>支持空 Tensor。</li><li>数据类型需要与 weight 满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li><li>N≥0，C≥1，其他维度≥0。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN</td>
  <td>NCL、NCHW、NCDHW</td>
  <td>3-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight</td>
  <td>输入</td>
  <td>公式中的 weight，表示卷积权重。</td>
  <td><ul><li>支持空 Tensor。</li><li>数据类型需要与 input 满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN</td>
  <td>NCL、NCHW、NCDHW</td>
  <td>3-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>bias</td>
  <td>输入</td>
  <td>公式中的 bias，表示卷积偏置。</td>
  <td><ul><li>无 bias 场景，可传入 nullptr。</li><li>当 transposed=false 时为一维且数值与 weight 第一维相等；当 transposed=true 时为一维且数值与 weight.shape[1] * groups 相等。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>1-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>stride</td>
  <td>输入</td>
  <td>卷积扫描步长。</td>
  <td>数组长度需等于 input 的维度减 2，值应该大于 0。</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>padding</td>
  <td>输入</td>
  <td>对 input 的填充。</td>
  <td>数组长度：conv1d 非转置为 1 或 2；conv1d转置为1；conv2d 为 2 或 4；conv3d 为 3。值应该大于等于 0。</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>dilation</td>
  <td>输入</td>
  <td>卷积核中元素的间隔。</td>
  <td>数组长度需等于 input 的维度减 2，值应该大于 0。</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>transposed</td>
  <td>输入</td>
  <td>是否为转置卷积。</td>
  <td>-</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>outputPadding</td>
  <td>输入</td>
  <td>转置卷积情况下，对输出所有边的填充。</td>
  <td>非转置卷积情况下忽略该配置。数组长度需等于input的维度减2。值应大于等于0，且小于 stride 或 dilation 对应维度的值。</td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>groups</td>
  <td>输入</td>
  <td>表示从输入通道到输出通道的块链接个数。</td>
  <td>数值需要在[1,65535]的范围内，且满足 groups*weight 的 C 维度=input 的 C 维度。</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>output</td>
  <td>输出</td>
  <td>公式中的 out，表示卷积输出。</td>
  <td><ul><li>数据类型需要与 input 与 weight 推导之后的数据类型保持一致。</li><li>支持空 Tensor。</li><li>通道数等于 weight 第一维，其他维度≥0。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN</td>
  <td>NCL、NCHW、NCDHW</td>
  <td>3-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>cubeMathType</td>
  <td>输入</td>
  <td>用于判断 Cube 单元应该使用哪种计算逻辑进行运算。</td>
  <td><ul><li>如果输入的数据类型存在<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>，该参数默认对互推导后的数据类型进行处理。</li><li>支持的枚举值如下：</li><ul><li> 0 (KEEP_DTYPE): 保持输入数据类型进行计算。</li></ul><ul><li> 1 (ALLOW_FP32_DOWN_PRECISION): 允许 FLOAT 降低精度计算，提升性能。</li></ul><ul><li> 2 (USE_FP16): 使用 FLOAT16 精度进行计算。</li></ul><ul><li> 3 (USE_HF32): 使用 HIFLOAT32（混合精度）进行计算。</li></ul></td>
  <td>INT8</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>输出</td>
  <td>返回需要在 Device 侧申请的 workspace 大小。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>输出</td>
  <td>返回 op 执行器，包含算子计算流程。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>


- **返回值：**

  aclnnStatus：返回状态码，具体参见<a href="../../../docs/zh/context/aclnn返回码.md">aclnn返回码</a>。

  第一段接口完成入参校验，出现以下场景时报错：

  <table>
  <tr>
  <td align="center">返回值</td>
  <td align="center">错误码</td>
  <td align="center">描述</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">传入的指针类型入参是空指针。</td>
  </tr>
  <tr>
  <td rowspan="11" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="11" align="left">161002</td>
  <td align="left">input、weight、bias、output数据类型和数据格式不在支持的范围之内。</td>
  </tr>
  <tr><td align="left">stride、padding、dilation、outputPadding输入shape不对。</td></tr>
  <tr><td align="left">input和output数据类型不一致；transposed=false时，支持input和output数据类型不一致，不会触发该类型报错。</td></tr>
  <tr><td align="left">groups 输入不对的情况。</td></tr>
  <tr><td align="left">output的shape不满足infershape结果。</td></tr>
  <tr><td align="left">outputPadding值不满足要求。</td></tr>
  <tr><td align="left">input、weight、bias、output传入的空 Tensor中部分维度为零的不满足要求。</td></tr>
  <tr><td align="left">input空间尺度在padding操作后小于weight(经过dilation扩张（如存在dilation>1的情况）)的空间尺度（非transpose模式下）。</td></tr>
  <tr><td align="left">transpose模式下bias的shape不为1。</td></tr>
  <tr><td align="left">stride、dilation小于0情况下不满足要求。</td></tr>
  <tr><td align="left">当前处理器不支持卷积。</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">API调用npu runtime的接口异常，如SocVersion不支持。</td>
  </tr>
  </table>

## aclnnConvolution

- **参数说明：**

  <table>
  <tr>
  <th style="width:120px">参数名</th>
  <th style="width:80px">输入/输出</th>
  <th>描述</th>
  </tr>
  <tr>
  <td>workspace</td>
  <td>输入</td>
  <td>在 Device 侧申请的 workspace 内存地址。</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>输入</td>
  <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnConvolutionGetWorkspaceSize 获取。</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>输入</td>
  <td>op 执行器，包含了算子计算流程。</td>
  </tr>
  <tr>
  <td>stream</td>
  <td>输入</td>
  <td>指定执行任务的 Stream。</td>
  </tr>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见<a href="../../../docs/zh/context/aclnn返回码.md">aclnn返回码</a>。

## 约束说明

- 确定性计算
  - aclnnConvolution默认确定性实现。

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width:70px">
    <col style="width:200px">
    <col style="width:200px">
    <col style="width:200px">
    <col style="width:200px">
    </colgroup>
   <thead>
    <tr>
     <th><term>约束类型</term></th>
     <th><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></th>
     <th><term>Atlas 推理系列产品</term></th>
     <th><term>Atlas 训练系列产品</term></th>
     <th><term>Atlas 200I/500 A2 推理产品</term></th>
     <th><term>昇腾 910_95 AI 处理器</term></th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">input、weight</th>
     <td>
        <ul><li>input、weight 数据类型不支持 HIFLOAT8、FLOAT8_E4M3FN。</li></ul>
        <ul><li>conv2d 和 conv3d transposed=true 场景，weight H、W 的大小应该在[1,255]范围内，其他维度应该大于等于 1。</li></ul>
        <ul><li>conv1d transposed=true 场景，weight L 的大小应该在[1,255]范围内，其他维度应该大于等于 1。</li></ul>
        <ul><li>conv3d 正向场景，weight H、W 的大小应该在[1,511]范围内。</li></ul>
     </td>
     <td>
        <ul><li>input、weight 数据类型不支持 BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。</li></ul>
        <ul><li>Conv3D正向场景仅支持FLOAT16，且input的H、W维度大小应小于等于4096，weight的D、H、W维度大小应小于等于255。</li></ul>
     </td>
     <td>
        <ul>input、weight 数据类型不支持 BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。</ul>
     </td>
     <td>
        <ul>input、weight 数据类型不支持 HIFLOAT8、FLOAT8_E4M3FN</ul>
     </td>
     <td>
        <ul>
           input、weight 数据类型支持 FLOAT、FLOAT16、BFLOAT16、HIFLOAT8。
           <ul>
              <li>transposed=true 时：input 数据类型额外支持 FLOAT8_E4M3FN，当input数据类型为HIFLOAT8或FLOAT8_E4M3FN时，output和weight的数据类型必须与input一致。支持 N 维度大于等于0，其他各个维度的大小应该大于等于1。weight数据类型额外支持 FLOAT8_E4M3FN，所有维度的大小应该大于等于0（当D、H 或 W 维度为0时，要求推导出的output对应维度也为0）。</li>
              <li>transposed=false 时：当 input 数据类型为 HIFLOAT8 时，weight 的数据类型必须与 input 一致。支持 N 维度大于等于0，支持 D、H、W 维度大于等于0（等于0的场景仅在 output 推导的 D、H、W 维度也等于0时支持），支持 C 维度大于等于0（等于0的场景仅在 output 推导的 N、C、D、H、W 其中某一维度等于0时支持）。weight的H、W的大小应该在[1,255]的范围内。N维度大小应该大于等于0（等于0的场景仅在bias、output的N维度也等于0时支持）。C维度大小的支持情况与input的C维度一致。</li>
            </ul>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">bias</th>
     <td>
        <ul>
          <li>bias 数据类型不支持 HIFLOAT8、FLOAT8_E4M3FN。数据类型与 self、weight 一致。</li>
          <li>conv1d、conv2d、conv3d 正向场景下 bias 会转成 FLOAT 参与计算。</li>
        </ul>
     </td>
     <td>
        <ul>
          bias 数据类型不支持 BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。Conv3D正向场景仅支持 FLOAT16。
        </ul>
     </td>
     <td>
        <ul>
          bias 数据类型不支持 BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。
        </ul>
     </td>
     <td>
        <ul>
          <li>bias 数据类型不支持 HIFLOAT8、FLOAT8_E4M3FN。</li>
          <li>bias 在 conv1d、conv2d、conv3d 正向场景下会转成 FLOAT 参与计算。</li>
        </ul>
     </td>
     <td>
        <ul>
          bias 数据类型不支持 HIFLOAT8、FLOAT8_E4M3FN。
          <ul>
            <li>transposed=true 时：当 input 和 weight 数据类型是 HIFLOAT8 和 FLOAT8_E4M3FN 时，不支持带 bias。</li>
            <li>transposed=false 时：当 input 和 weight 数据类型是 HIFLOAT8 时，bias 数据类型会转成 FLOAT 参与计算。</li>
          </ul>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">stride</th>
     <td colspan="4">
        <ul>
          conv3d transposed=true场景，strideD应该大于等于1，strideH、strideW应该在[1,63]的范围内。conv1d和conv2d transposed=true场景，各个值都应该大于等于1。conv3d正向场景，strideH和strideW应该在[1,63]的范围内。
        </ul>
     </td>
     <td>
        <ul>
          conv3d transposed=true场景，strideD、strideH、strideW应该大于等于1。conv3d transposed=false场景，strideH、strideW应该在[1,63]的范围内，strideD应该在[1, 1000000]的范围内。
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">padding</th>
     <td colspan="4">
        <ul>
          conv3d transposed=true场景，paddingD应该大于等于0，paddingH、paddingW应该在[0,255]的范围内。conv1d和conv2d transposed=false场景，各个值都应该在[0,255]的范围内。conv3d正向场景，paddingH和paddingW应该在[0,255]的范围内。
        </ul>
     </td>
     <td>
        <ul>
          conv3d transposed=true场景，paddingD、paddingH、paddingW应该大于等于0。conv3d transposed=false场景，paddingH、paddingW应该在[0,255]范围内，paddingD应该在[0, 1000000]的范围内。
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">dilation</th>
     <td colspan="4">
        <ul>
          conv1d、conv2d和conv3d transposed=true场景，各个值都应该在[1,255]的范围内。conv3d正向场景，dilationH和dilationW应该在[1,255]的范围内。
        </ul>
     </td>
     <td>
        <ul>
          conv3d transposed=true场景，dilationD、dilationH、dilationW应该大于等于1。conv3d transposed=false场景，dilationH、dilationW应该在[1,255]范围内，dilationD应该在[1, 1000000]的范围内。
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType</th>
     <td>
        <ul>
          <li>为 0(KEEP_DTYPE) 时，当输入是 FLOAT 暂不支持。</li>
          <li>为 1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是 FLOAT 允许转换为 HFLOAT32 计算。</li>
          <li>为 2(USE_FP16) 时，当输入是 BFLOAT16 不支持该选项。</li>
          <li>为 3(USE_HF32) 时，当输入是 FLOAT 转换为 HFLOAT32 计算。</li>
        <ul>
     </td>
     <td colspan="2">
        <ul>
          <li>为 0(KEEP_DTYPE) 时，当输入是 FLOAT 暂不支持。</li>
          <li>为 1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是 FLOAT 允许转换为 FLOAT16 计算。</li>
          <li>为 2(USE_FP16) 时，当输入是 BFLOAT16 不支持该选项。</li>
          <li>为 3(USE_HF32) 时暂不支持。</li>
        <ul>
     </td>
     <td>
        <ul>
          <li>为 0(KEEP_DTYPE) 时，当输入是 FLOAT 暂不支持。</li>
          <li>为 1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是 FLOAT 允许转换为 HFLOAT32 计算。</li>
          <li>为 2(USE_FP16) 时，当输入是 BFLOAT16 不支持该选项。</li>
          <li>为 3(USE_HF32) 时，当输入是 FLOAT 转换为 HFLOAT32 计算。</li>
        </ul>
     </td>
     <td>
        <ul>
          <li>为 1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是 FLOAT 允许转换为 HFLOAT32 计算。</li>
          <li>为 2(USE_FP16) 时，当输入是 BFLOAT16 不支持该选项。</li>
          <li>为 3(USE_HF32) 时，当输入是 FLOAT 转换为 HFLOAT32 计算。</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">其他约束</th>
     <td>
        <ul>
          input, weight, bias中每一组tensor的每一维大小都应不大于1000000。
        </ul>
     </td>
     <td>
        <ul>
          支持1D和2D卷积，3D卷积正向场景仅支持FLOAT16。且3D卷积正向场景input的H、W维度应小于等于4096，weight的D、H、W维度应小于等于255。input, weight, bias中每一组tensor的每一维大小都应不大于1000000。
        </ul>
     </td>
     <td>
        <ul>
          支持1D和2D卷积，3D卷积仅支持transposed为false且输入数据类型为FLOAT16场景。input, weight, bias中每一组tensor的每一维大小都应不大于1000000。
        </ul>
     </td>
     <td>
        <ul>
          当前仅支持2D卷积，暂不支持1D、3D卷积。input, weight, bias中每一组tensor的每一维大小都应不大于1000000。
        </ul>
     </td>
     <td>
        <ul>
          transposed为true的场景，支持1D、2D和3D卷积，支持空 Tensor。
        </ul>
     </td>
   </tr>
   </tbody>
  </table>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考<a href="../../../docs/zh/context/编译与运行样例.md">编译与运行样例</a>。
```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                    \
    if (!(cond)) {                        \
      Finalize(deviceId, stream);         \
      return_expr;                        \
    }                                     \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i: shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnConvolutionTest(int32_t deviceId, aclrtStream& stream)
{
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> shapeInput = {2, 2, 2, 2};
  std::vector<int64_t> shapeWeight = {1, 2, 1, 1};
  std::vector<int64_t> shapeResult = {2, 1, 2, 2};
  std::vector<int64_t> convStrides;
  std::vector<int64_t> convPads;
  std::vector<int64_t> convOutPads;
  std::vector<int64_t> convDilations;

  void* deviceDataA = nullptr;
  void* deviceDataB = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* result = nullptr;
  std::vector<float> inputData(GetShapeSize(shapeInput), 1);
  std::vector<float> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> outputData(GetShapeSize(shapeResult), 1);
  convStrides = {1, 1, 1, 1};
  convPads = {0, 0, 0, 0};
  convOutPads = {0, 0, 0, 0};
  convDilations = {1, 1, 1, 1};

  // 创建input aclTensor
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, aclDataType::ACL_FLOAT, &input);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建weight aclTensor
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, aclDataType::ACL_FLOAT, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult, aclDataType::ACL_FLOAT, &result);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
  CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outPadsPtr(outPads, aclDestroyIntArray);
  CHECK_FREE_RET(outPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
  CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnConvolution第一段接口
  ret = aclnnConvolutionGetWorkspaceSize(input, weight, nullptr, strides, pads, dilations, false, outPads, 1, result, 1,
                                         &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // 调用aclnnConvolution第二段接口
  ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolution failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(shapeResult);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnConvolutionTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvolutionTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
