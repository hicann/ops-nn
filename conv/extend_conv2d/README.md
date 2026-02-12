# ExtendConv2D

## 产品支持情况

<table>
<tr>
<th style="text-align:left">产品</th>
<th style="text-align:center; width:100px">是否支持</th>
</tr>
<tr>
<td><term>Ascend 950PR/Ascend 950DT </term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></td>
<td style="text-align:center">×</td>
</tr>
<tr>
<td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></td>
<td style="text-align:center">×</td>
</tr>
<tr>
<td><term>Atlas 200I/500 A2 推理产品</term></td>
<td style="text-align:center">×</td>
</tr>
<tr>
<td><term>Atlas 推理系列产品 </term></td>
<td style="text-align:center">×</td>
</tr>
<tr>
<td><term>Atlas 训练系列产品</term></td>
<td style="text-align:center">×</td>
</tr>
</table>

## 功能说明

- 算子功能：实现 2D 卷积功能。

- 计算公式：

  - 假定输入（`x`）的 shape 是 $(N, C_{\text{in}}, H, W)$ ，（`filter`）的 shape 是 $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$，输出（`y`）的 shape 是 $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$

  - 输出表示为：

  $$
    \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)\\
    \text{y0}(N_i, C_{\text{out}_j}) = \text{out}(N_i, C_{\text{out}_j}) \times\text{scale0}(C_{\text{out}_j})\\
    \text{y1}(N_i, C_{\text{out}_j}) = \text{out}(N_i, C_{\text{out}_j}) \times\text{scale1}(C_{\text{out}_j})   
  $$
  如果启用relu，那么：
  $$
    \text{y0} = \text{relu}(\text{y0})\\
    \text{y1} = \text{relu}(\text{y1})
  $$

  其中，$\star$ 表示卷积计算，支持空洞卷积、分组卷积。$N$ 代表 `batch size`，$C$ 代表通道数，$H$ 和 $W$ 分别代表高和宽，相应输出维度的计算公式如下：

  $$
    H_{\text{out}} = (H + \text{pad\_top} + \text{pad\_bottom} - (\text{dilation\_h} \times (K_h - 1) + 1)) / \text{stride\_h} + 1 \\
    W_{\text{out}} = (W + \text{pad\_left} + \text{pad\_right} - (\text{dilation\_w} \times (K_w - 1) + 1)) / \text{stride\_w} + 1
  $$

## 参数说明

<table>
<tr>
<th style="width:100px">参数名</th>
<th style="width:180px">输入 / 输出 / 属性</th>
<th style="width:420px">描述</th>
<th style="width:420px">数据类型</th>
<th style="width:200px">数据格式</th>
</tr>
<tr>
<td>x</td>
<td>输入</td>
<td>公式中的输入张量 x。</td>
<td>HIFLOAT8、INT8、FLOAT8_E4M3FN</td>
<td>NCHW、NHWC</td>
</tr>
<tr>
<td>filter</td>
<td>输入</td>
<td>公式中的卷积权重张量 filter。</td>
<td>HIFLOAT8、INT8、FLOAT8_E4M3FN</td>
<td>NCHW、HWCN</td>
</tr>
<tr>
<td>bias</td>
<td>可选输入</td>
<td>卷积偏置张量 bias。</td>
<td>FLOAT、INT32</td>
<td>ND</td>
</tr>
<tr>
<td>offset_w</td>
<td>可选输入</td>
<td>量化偏移张量 offset_w（未使用）。</td>
<td>INT8</td>
<td>-</td>
</tr>
<tr>
<td>scale0</td>
<td>可选输入</td>
<td>输出0的量化参数 scale0。</td>
<td>INT64，UINT64</td>
<td>ND</td>
</tr>
<tr>
<td>relu_weight0</td>
<td>可选输入</td>
<td>输出0对应的relu斜率参数relu_weight0。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>clip_value0</td>
<td>可选输入</td>
<td>输出0对应的截断值 clip_value0（未使用）。</td>
<td>FLOAT16、HIFLOAT8、INT8、FLOAT8_E4M3FN、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>scale1</td>
<td>可选输入</td>
<td>输出1的量化参数 scale1。</td>
<td>INT64，UINT64</td>
<td>ND</td>
</tr>
<tr>
<td>relu_weight1</td>
<td>可选输入</td>
<td>输出1对应的relu斜率参数relu_weight1。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>clip_value1</td>
<td>可选输入</td>
<td>输出1对应的截断值 clip_value1（未使用）。</td>
<td>FLOAT16、HIFLOAT8、INT8、FLOAT8_E4M3FN、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>y0</td>
<td>输出</td>
<td>公式中的输出张量 y0。</td>
<td>FLOAT16、FLOAT、BFLOAT16、HIFLOAT8、INT8、FLOAT8_E4M3FN</td>
<td>NCHW、NHWC</td>
</tr>
<tr>
<td>y1</td>
<td>输出</td>
<td>公式中的输出张量 y1。</td>
<td>FLOAT16、FLOAT、BFLOAT16、HIFLOAT8、INT8、FLOAT8_E4M3FN</td>
<td>NCHW、NHWC</td>
</tr>
<tr>
<td>strides</td>
<td>属性</td>
<td>卷积扫描步长，stride_h, stride_w ∈ [1,63]。</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>pads</td>
<td>可选属性</td>
<td>对输入的填充，pad_h, pad_w ∈ [0,255]。</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>dilations</td>
<td>可选属性</td>
<td>卷积核中元素的间隔，dilation_h, dilation_w ∈ [1,255]。</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>groups</td>
<td>可选属性</td>
<td>从输入通道到输出通道的块链接个数，必须满足 groups × filter 的 in_channels 维度 = x 的 in_channels 维度，以及filter的out_channels必须是groups的倍数。支持范围 [1, 65535]。</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>data_format</td>
<td>可选属性</td>
<td>输入数据格式，支持 "NCHW"、"NHWC"。</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>offset_x</td>
<td>可选属性</td>
<td>量化算法中的偏移 offset_x。</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>round_mode</td>
<td>可选属性</td>
<td>舍入模式，如果输出的数据类型是 hifloat8，round_mode 可以设置为 'round'。否则可以设置为 'rint'。</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>pad_mode</td>
<td>可选属性</td>
<td>填充模式，支持 "SPECIFIC"、"SAME"、"VALID"、"SAME_UPPER", "SAME_LOWER"。</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>enable_hf32</td>
<td>可选属性</td>
<td>是否启用 HF32 计算，支持 true、false（未使用）。</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>enable_relu0</td>
<td>可选属性</td>
<td>是否为第0个输出启用 relu，支持 true、false。</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>enable_relu1</td>
<td>可选属性</td>
<td>是否为第1个输出启用 relu，支持 true、false。</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>dual_output</td>
<td>可选属性</td>
<td>是否使用双输出，支持 true、false。</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>dtype0</td>
<td>可选属性</td>
<td>表示输出 y0 的数据类型。支持的列表包括 [-1(默认)，0(FLOAT)，1(FLOAT16)，2(INT8)，27(BFOAT16)， 34(HIFLOAT8), 36(FLOAT8_E4M3FN)]。</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>dtype1</td>
<td>可选属性</td>
<td>表示输出 y1 的数据类型。支持的列表包括 [-1(默认)，0(FLOAT)，1(FLOAT16)，2(INT8)，27(BFOAT16)， 34(HIFLOAT8), 36(FLOAT8_E4M3FN)]</td>
<td>BOOL</td>
<td>-</td>
</tr>
</table>

## 约束说明

- Ascend 950PR/Ascend 950DT ：
  - `x` 的数据类型必须与 `filter` 一致。`N` 维度大小应该大于等于 0。`H`、`W` 维度大小应该大于等于 0（等于 0 的场景仅在输出 `y` 的 `H`、`W` 维度也等于 0 时支持）。`C` 维度大小应该大于等于 0（等于 0 的场景仅在输出 `y` 的任意维度也等于 0 时支持）。
  - 对于 `filter` 输入，`H`、`W` 的大小应该在 [1, 511] 的范围内。`N` 维度大小应该大于等于 0（等于 0 的场景仅在 `bias`、`output` 的 `N` 维度也等于 0 时支持），`C` 维度大小的支持情况与输入 `x` 的 `C` 维度一致。
  - `bias`和`scale`维度大小应该与`filter`的`N`维度大小一致。


  <table>
  <tr>
  <th style="text-align:center; width:80px">张量</th>
  <th style="text-align:center; width:150px">x</th>
  <th style="text-align:center; width:150px">filter</th>
  <th style="text-align:center; width:100px">bias</th>
   <th style="text-align:center; width:150px">scale0/1</th>
   <th style="text-align:center; width:150px">clip_value0/1</th>
  <th style="text-align:center; width:150px">y0/1</th>
  </tr>
  <tr>
  <td rowspan="3" style="text-align:center">数据类型</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">INT32</td>
  <td style="text-align:center">INT64/UINT64</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">FLOAT16/INT8</td>
  </tr>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">INT64/UINT64</td>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">FLOAT/FLOAT16/BFLOAT16/HIFLOAT8</td>
  </tr>
  <tr>
  <td style="text-align:center">FLOAT8_E4M3FN</td>
  <td style="text-align:center">FLOAT8_E4M3FN</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">INT64/UINT64</td>
  <td style="text-align:center">FLOAT8_E4M3FN</td>
  <td style="text-align:center">FLOAT/FLOAT16/BFLOAT16/FLOAT8_E4M3FN</td>
  </tr>
  <tr>
  <td rowspan="2" style="text-align:center">数据格式</td>
  <td style="text-align:center">NCHW</td>
  <td style="text-align:center">NCHW</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NCHW</td>
  </tr>
  <tr>
  <td style="text-align:center">NHWC</td>
  <td style="text-align:center">HWCN</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NHWC</td>
  </tr>
  </table>

- `x`、`filter`、`bias`、`scale0/1`、`relu_weight0/1`、`clip_value0/1`、`y` 中每一组 `tensor` 的每一维大小都应不大于 1000000。

- `groups` ∈ [1, 65535]。

- 如果任何参数超出上述范围，算子的正确性无法保证。

- 由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击 [Link](https://www.hiascend.com/support) 获取技术支持。


## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_extend_conv2d](./examples/arch35/test_aclnn_extend_conv2d.cpp) | 通过 [aclnnQuantConvolution](../convolution_forward/docs/aclnnQuantConvolution.md) 接口方式调用 extend_conv2d 算子。
