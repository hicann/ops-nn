# Conv3DV2

## 产品支持情况

<table>
<tr>
<th style="text-align:left">产品</th>
<th style="text-align:center; width:100px">是否支持</th>
</tr>
<tr>
<td><term>Ascend 950PR/Ascend 950DT</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></td>
<td style="text-align:center">√</td>
</tr>
</table>

## 功能说明

- 算子功能：实现 3D 卷积功能。

- 计算公式：

  - 假定输入（`x`）的 shape 是 $(N, C_{\text{in}}, D, H, W)$ ，（`filter`）的 shape 是 $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$，输出（`y`）的 shape 是 $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$

  - 对于 INT8 类型的输入，输出将被表示为：

  $$
    \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \text{scale} \times \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  - 对于其他数据类型的输入，输出将被表示为：

  $$
    \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  其中，$\star$ 表示卷积计算，支持空洞卷积、分组卷积。$N$ 代表 `batch size`，$C$ 代表通道数，$D$、$H$ 和 $W$ 分别代表深度、高和宽，相应输出维度的计算公式如下：

  $$
    D_{\text{out}} = (D + \text{pad\_head} + \text{pad\_tail} - (\text{dilation\_d} \times (K_d - 1) + 1)) / \text{stride\_d} + 1 \\
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
<td>FLOAT16、FLOAT、BFLOAT16、INT8、HIFLOAT8</td>
<td>NCDHW、NDHWC、NDC1HWC0</td>
</tr>
<tr>
<td>filter</td>
<td>输入</td>
<td>公式中的卷积权重张量 filter。</td>
<td>FLOAT16、FLOAT、BFLOAT16、INT8、HIFLOAT8</td>
<td>NCDHW、DHWCN、FRACTAL_Z_3D</td>
</tr>
<tr>
<td>bias</td>
<td>可选输入</td>
<td>卷积偏置张量 bias。</td>
<td>FLOAT16、FLOAT、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>scale</td>
<td>可选输入</td>
<td>缩放因子张量 scale。</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>offset</td>
<td>可选输入</td>
<td>偏移张量 offset（未使用）。</td>
<td>FLOAT</td>
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
<td>y</td>
<td>输出</td>
<td>公式中的输出张量 y。</td>
<td>FLOAT16、FLOAT、BFLOAT16、HIFLOAT8</td>
<td>NCDHW、NDHWC、NDC1HWC0</td>
</tr>
<tr>
<td>strides</td>
<td>属性</td>
<td>卷积扫描步长，stride_d ∈ [1,1000000]，stride_h, stride_w ∈ [1,63]。</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>pads</td>
<td>可选属性</td>
<td>对输入的填充，pad_h, pad_w ∈ [0,255]，paddingD ∈ [0,1000000]。</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>dilations</td>
<td>可选属性</td>
<td>卷积核中元素的间隔，dilation_h, dilation_w ∈ [1,255]，dilation_d ∈ [1,1000000]。</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>groups</td>
<td>可选属性</td>
<td>从输入通道到输出通道的块链接个数，必须满足 groups × filter 的 in_channels 维度 = x 的 in_channels 维度。支持范围 [1, 65535]。</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>data_format</td>
<td>可选属性</td>
<td>输入数据格式，支持 "NCDHW"、"NDHWC"、"NDC1HWC0"。</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>offset_x</td>
<td>可选属性</td>
<td>量化算法中的偏移 offset_x（未使用）。</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>pad_mode</td>
<td>可选属性</td>
<td>填充模式，支持 "SPECIFIC"、"SAME"、"VALID"。</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>enable_hf32</td>
<td>可选属性</td>
<td>是否启用 HF32 计算，支持 true、false。</td>
<td>BOOL</td>
<td>-</td>
</tr>
</table>

- Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
  - 不支持 `HIFLOAT8` 数据类型。
  - `scale` 和 `offset` 参数仅支持 `FLOAT` 类型。
  - 输入为 `INT8` 数据类型时，`bias` 为必选输入，`groups` 仅支持 1。
  - `filter`、`y` 不支持 `DHWCN` 数据格式。
  - 不支持 `pad_mode` 属性。
  - 当不满足 `Pointwise` 分支情况时，`x` 支持 `NDC1HWC0`，`filter` 支持 `FRACTAL_Z_3D`。

- Ascend 950PR/Ascend 950DT：
  - 不支持 `INT8` 数据类型。

## 约束说明

- Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
  - `filter` 的 `H`、`W` 维度范围：[1,511]。
  - 不支持空 `tensor`。
  - 当 `groups` 为 1, `dilation` 全为 1，`padding` 全为 0，`filter` 没有为 1 的维度， `x` 的 `D` * `H` * `W` 小于 65536，`bias` 为 `FLOAT` 时，会进入 `Pointwise` 分支，可以使用 `NCDHW` 格式。

  <table>
  <tr>
  <th style="text-align:center; width:80px">张量</th>
  <th style="text-align:center; width:120px">x</th>
  <th style="text-align:center; width:120px">filter</th>
  <th style="text-align:center; width:100px">bias</th>
  <th style="text-align:center; width:80px">scale</th>
  <th style="text-align:center; width:80px">offset</th>
  <th style="text-align:center; width:120px">y</th>
  </tr>
  <tr>
  <td rowspan="4" style="text-align:center">数据类型</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">BFLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  </tr>
  <tr>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">BFLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">数据格式</td>
  <td style="text-align:center">NCDHW、NDC1HWC0</td>
  <td style="text-align:center">NCDHW、FRACTAL_Z_3D</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NCDHW、NDC1HWC0</td>
  </tr>
  </table>

- Ascend 950PR/Ascend 950DT：
  - 当 `x` 数据类型为 `HIFLOAT8` 时，`filter` 的数据类型必须与 `x` 一致。`N` 维度大小应该大于等于 0。`D`、`H`、`W` 维度大小应该大于等于 0（等于 0 的场景仅在输出 `y` 的 `D`、`H`、`W` 维度也等于 0 时支持）。`C` 维度大小应该大于等于 0（等于 0 的场景仅在输出 `y` 的任意维度也等于 0 时支持）。
  - 对于 `filter` 输入，`H`、`W` 的大小应该在 [1, 511] 的范围内。`N` 维度大小应该大于等于 0（等于 0 的场景仅在输入 `bias`、输出 `y` 的 `N` 维度也等于 0 时支持），`C` 维度大小的支持情况与输入 `x` 的 `C` 维度一致。
  - 当 `x` 和 `filter` 数据类型是 `HIFLOAT8` 时，`bias` 数据类型会转成 `FLOAT` 参与计算。
  - 不支持 `scale` 参数。


  <table>
  <tr>
  <th style="text-align:center; width:80px">张量</th>
  <th style="text-align:center; width:150px">x</th>
  <th style="text-align:center; width:150px">filter</th>
  <th style="text-align:center; width:100px">bias</th>
  <th style="text-align:center; width:150px">y</th>
  </tr>
  <tr>
  <td rowspan="4" style="text-align:center">数据类型</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">BFLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  </tr>
  <tr>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">HIFLOAT8</td>
  </tr>
  <tr>
  <td rowspan="2" style="text-align:center">数据格式</td>
  <td style="text-align:center">NCDHW</td>
  <td style="text-align:center">NCDHW</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NCDHW</td>
  </tr>
  <tr>
  <td style="text-align:center">NDHWC</td>
  <td style="text-align:center">DHWCN</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NDHWC</td>
  </tr>
  </table>

- `x`、`filter`、`bias`、`scale`、`y` 中每一组 `tensor` 的每一维大小都应不大于 1000000。

- `groups` ∈ [1, 65535]。

- 如果任何参数超出上述范围，算子的正确性无法保证。

- 由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击 [Link](https://www.hiascend.com/support) 获取技术支持。


## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_conv3d_v2](./examples/test_aclnn_conv3d_v2.cpp) | 通过 [aclnnConvolution](../convolution_forward/docs/aclnnConvolution.md) 接口方式调用 Conv3DV2 算子。    |
