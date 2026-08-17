# AvgPool1d

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：对输入tensor的最后一维做一维平均池化（1D average pooling），支持不对称padding、ceil_mode输出长度修正和count_include_pad分母选择。

- 计算公式：

设输入x的shape为(..., L)，pads=[pad_left, pad_right]，padding后长度L_padded = L + pad_left + pad_right。输出长度L_out按以下公式计算：

$$
L_{out} = \begin{cases} \left\lfloor \frac{L_{padded} - ksize}{strides} \right\rfloor + 1, & ceil\_mode = False \\ \left\lceil \frac{L_{padded} - ksize}{strides} \right\rceil + 1, & ceil\_mode = True \end{cases}
$$

当pad_left > 0且最后一个窗口的起始位置落在padding区时（即(L_out - 1) * strides >= L + pad_left），L_out减1。

每个输出窗口的值为：

$$
y[..., w] = \frac{\sum_{i=0}^{ksize-1} x_{padded}[..., w \times strides + i]}{divisor}
$$

其中divisor在count_include_pad=True时为窗口在padded空间的实际元素数（ceil_mode下可能小于ksize），在count_include_pad=False时为窗口与原始数据区的交集大小（最小为1）。

## 参数说明

<table><thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>待进行一维平均池化计算的入参，shape为(..., L)，L为池化方向维度长度。</td>
    <td>float16、float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>一维平均池化计算结果，shape为(..., L_out)，与x同dtype。</td>
    <td>float16、float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ksize</td>
    <td>属性</td>
    <td>池化窗口大小。</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>strides</td>
    <td>属性</td>
    <td>池化窗口滑动步长。</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pads</td>
    <td>属性</td>
    <td>左右padding大小，格式为[pad_left, pad_right]，支持不对称padding。</td>
    <td>ListInt</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ceil_mode</td>
    <td>属性</td>
    <td>为True时用ceil模式计算输出长度，为False时用floor模式。默认False。</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>count_include_pad</td>
    <td>属性</td>
    <td>为True时分母含padding元素，为False时仅含有效元素。默认False。</td>
    <td>bool</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- 输入x的rank需大于等于3，支持3D~8D。
- ksize需大于等于1。
- strides需大于等于1（stride=0将报错）。
- pads的元素需大于等于0，长度为2。
- 输入x仅支持float16、float32类型。
- 数据格式仅支持ND。
- 输出y与输入x同dtype。
- 支持空tensor输入（L=0时输出空tensor）。
- 当ceil_mode=True且最后一个窗口的起始位置落在padding区时（即(L_out - 1) * strides >= L + pad_left），L_out减1。
- 当窗口内无有效元素（data_num==0）时将报错（除零错误）。
- 确定性实现：采用单线程内顺序累加，不使用原子操作，相同输入多次执行结果一致。
- NaN输入按IEEE 754规则传播，Inf输入按IEEE 754规则传播。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_avg_pool1_d.cpp">test_geir_avg_pool1_d</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
