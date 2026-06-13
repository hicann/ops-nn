# SwigluBackwardMxQuantWithDualAxis

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |
| <term>Kirin X90 处理器系列产品</term> | × |
| <term>Kirin 9030 处理器系列产品</term> | × |

## 功能说明

- 接口功能：融合算子，实现SwiGLU激活函数反向梯度计算与双轴动态块量化的组合计算。先对输入x和梯度yGrad计算SwiGLU反向梯度，然后对梯度结果分别在-1轴和-2轴进行基于块的动态量化，输出低精度的FP8张量和对应的缩放因子。

- 计算公式：

  **阶段1：SwiGLU反向梯度计算**

  - 当`activateLeft=true`时：

  $$
  A_i = x_i \text{的前半部分}, B_i = x_i \text{的后半部分}
  $$

  - 当`activateLeft=false`时：

  $$
  A_i = x_i \text{的后半部分}, B_i = x_i \text{的前半部分}
  $$

  - SwiGLU正向计算：

  $$
  swigluOut_i = Swish(A_i) * B_i = \frac{A_i}{1 + e^{-A_i}} * B_i
  $$

  - SwiGLU反向计算：

  $$
  grad\_A = yGrad * B * sigmoid(A) * (1 + A - (A * sigmoid(A))) \\
  grad\_B = yGrad * A * sigmoid(A) \\
  grad\_x = concat(grad\_A, grad\_B)
  $$

  **阶段2：双轴动态块量化**

- **-1轴量化（列方向）**：将SwiGLU结果在-1轴上按照32个数进行分组，一组32个数$\{\{V_i\}_{i=1}^{32}\}$量化为$\{mxscale1, \{P_i\}_{i=1}^{32}\}$

  $$
  shared\_exp = floor(log_2(max_i(|V_i|))) - emax
  $$

  $$
  mxscale1 = 2^{shared\_exp}
  $$

  $$
  P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
  $$
- **-2轴量化（行方向）**：将SwiGLU结果在-2轴上按照32个数进行分组，一组32个数$\{\{V_j\}_{j=1}^{32}\}$量化为$\{mxscale2, \{P_j\}_{j=1}^{32}\}$

  $$
  shared\_exp = floor(log_2(max_j(|V_j|))) - emax
  $$

  $$
  mxscale2 = 2^{shared\_exp}
  $$

  $$
  P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
  $$
- -1轴量化后的$P_{i}$按对应的$V_{i}$的位置组成输出y1Out，mxscale1按对应的-1轴维度上的分组组成输出mxscale1Out。-2轴量化后的$P_{j}$按对应的$V_{j}$的位置组成输出y2Out，mxscale2按对应的-2轴维度上的分组组成输出mxscale2Out。
- emax: 对应数据类型的最大正则数的指数位。

  |   DataType   | emax |
  | :-----------: | :--: |
  |  FLOAT4_E2M1  |  2  |
  |  FLOAT4_E1M2  |  0  |
  | FLOAT8_E4M3FN |  8  |
  |  FLOAT8_E5M2  |  15  |

## 参数说明

<table><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入张量，公式中的x。</td>
      <td><ul><li>shape为[M, 2N]，最后一维必须为偶数。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_grad</td>
      <td>输入</td>
      <td>反向梯度输入，shape为[M, N]，即正向SwiGLU输出的梯度。</td>
      <td><ul><li>shape为[M, N]，其中N = x最后一维 / 2。</li><li>数据类型必须与x一致。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td>分组索引，用于控制分组量化边界。</td>
      <td><ul><li>shape为[G]，采用cumsum模式，表示每个group的行数累积值。传入空指针时表示不分组。</li><li>当前不支持传入空Tensor。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>可选属性</td>
      <td>SwiGLU激活侧选择。</td>
      <td><ul><li>True表示左半部分为hidden，右半部分为gate。</li><li>False表示右半部分为hidden，左半部分为gate。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td>表示数据转换的模式，对应公式中的round_mode。</td>
      <td><ul><li>仅支持dstDtype为36，对应输出y1Out和y2Out数据类型为FLOAT8_E4M3FN时，仅支持{"rint"}。</li><li>传入空指针时，采用"rint"模式。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>可选属性</td>
      <td>表示mxscale1Out和mxscale2Out的计算方法。</td>
      <td><ul><li>取值范围：{1}，代表cuBLAS实现。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_dtype</td>
      <td>可选属性</td>
      <td>表示指定数据转换后y1Out和y2Out的类型。</td>
      <td><ul><li>输入范围为 {36}，分别对应输出y1Out和y2Out的数据类型为{36: FLOAT8_E4M3FN}。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_dtype_value</td>
      <td>可选属性</td>
      <td>预留参数。</td>
      <td><ul><li>maxDtypeValue取值仅支持0。</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1_out</td>
      <td>输出</td>
      <td>表示SwiGLU结果量化-1轴后的对应结果，对应公式中的<i>P<sub>i</sub></i>。</td>
      <td><ul><li>shape为[M, 2N]。</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale1_out</td>
      <td>输出</td>
      <td>表示-1轴每个分组对应的量化尺度，对应公式中的mxscale1。</td>
      <td><ul><li>shape为[M, (ceil(N/32)+2-1)/2, 2]，需进行偶数pad，pad填充值为0。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2_out</td>
      <td>输出</td>
      <td>表示SwiGLU结果量化-2轴后的对应结果，对应公式中的<i>P<sub>j</sub></i>。</td>
      <td><ul><li>shape为[M, 2N]。</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale2_out</td>
      <td>输出</td>
      <td>表示-2轴每个分组对应的量化尺度，对应公式中的mxscale2。</td>
      <td><ul><li>当groupIndexOptional存在时，shape为 [floor(M/64)+G, N, 2]。</li><li>需进行偶数pad，pad填充值为0。</li><li>mxscale2Out输出需要对每两行数据进行交织处理。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x的最后一维必须能被64整除。
- FP8输出类型（FLOAT8_E4M3FN）仅支持"rint"舍入模式。
- groupIndexOptional采用cumsum模式，每个值表示对应group的行数累积值，groupIndexOptional的每个元素值需要大于0且最后一个元素值要等于M。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :------- | :------- | :--- |
| aclnn调用 | [test_aclnn_swiglu_backward_mx_quant_with_dual_axis](./examples/arch35/test_aclnn_swiglu_backward_mx_quant_with_dual_axis.cpp) | 通过[aclnnSwigluBackwardMxQuantWithDualAxis](./docs/aclnnSwigluBackwardMxQuantWithDualAxis.md)接口方式调用SwigluBackwardMxQuantWithDualAxis算子。 |
