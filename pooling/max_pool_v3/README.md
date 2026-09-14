# MaxPoolV3

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|<term>Ascend 950PR/Ascend 950DT</term>   |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    x     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 接口功能：
对于dim=3或4维的输入张量，进行最大池化（max pooling）操作。
- 计算公式：
  - 当ceilMode=False时，out tensor的shape中H和W维度推导公式：

    $$
    [H_{out}, W_{out}]=[\lfloor{\frac{H_{in}+  padding\_size_{Htop} + padding\_size_{Hbottom} - {dilation\_size \times(k_h - 1) - 1}}{s_h}}\rfloor + 1,\lfloor{\frac{W_{in}+ padding\_size_{Wleft} + padding\_size_{Wright} - {dilation\_size \times(k_w - 1) - 1}}{s_w}}\rfloor + 1]
    $$

  - 当ceilMode=True时，out tensor的shape中H和W维度推导公式：

    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in}+  padding\_size_{Htop} + padding\_size_{Hbottom} - {dilation\_size \times(k_h - 1) - 1}}{s_h}}\rceil + 1,\lceil{\frac{W_{in}+ padding\_size_{Wleft} + padding\_size_{Wright} - {dilation\_size \times(k_w - 1) - 1}}{s_w}}\rceil + 1]
    $$

    - 滑窗左上角起始位处在下或右侧pad填充位上或者界外（无法取到有效值）时，舍弃该滑窗结果，在上述推导公式基础上对应空间轴shape需减去1：

      $$
      \begin{cases}
      H_{out}=H_{out} - 1& \text{if } (H_{out}-1)*s_h>=H_{in}+padding\_size_{Htop} \\
      W_{out}=W_{out} - 1& \text{if } (W_{out}-1)*s_w>=W_{in}+padding\_size_{Wleft}  \\
      \end{cases}\\
      $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
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
      <td>4维输入tensor。</td>
      <td>FLOAT16、BF16、FLOAT32、DOUBLE、INT32、INT64、UINT8、INT16、INT8、UINT16、QINT8</td>
      <td>NHWC、NCHW</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>最大池化的窗口大小，长度为4的列表。N和C维度的ksize必须为1，H和W维度的ksize必须大于0。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>滑动窗口的步长，长度为4的列表。N和C维度的stride必须为1。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>属性</td>
      <td>指定padding的方式。支持"SAME"、"VALID"、"CALCULATED"，默认为"CALCULATED"。"SAME"时补零使输出shape等于ceil(input shape / stride)；"VALID"时不补零；"CALCULATED"时使用pads计算输出shape。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>沿着空间轴方向开始和结束的位置填充，对应公式中的padding_size。长度为4，指定[top, bottom, left, right]的填充量。仅在padding_mode为"CALCULATED"时生效。默认值为{0,0,0,0}。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>输入输出的数据格式，支持"NHWC"和"NCHW"，默认为"NCHW"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>global_pooling</td>
      <td>属性</td>
      <td>是否使用全局池化。为True时忽略ksize和pads，输出H和W均为1。默认False。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>属性</td>
      <td>计算输出形状的取整模式。True时向上取整，False时向下取整。padding_mode为"SAME"或"VALID"时仅支持False。默认False。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>4维输出tensor，与输入x具有相同的数据类型和格式。</td>
      <td>FLOAT16、BF16、FLOAT32、DOUBLE、INT32、INT64、UINT8、INT16、INT8、UINT16、QINT8</td>
      <td>NHWC、NCHW</td>
    </tr>
  </tbody></table>

## 约束说明

- **值域限制说明：**
  - ksize：长度为4的列表。H和W维度的ksize必须大于0，N和C维度的ksize必须为1。H和W维度的ksize乘积应小于等于255。
  - strides：长度为4的列表。N和C维度的stride必须为1，H和W维度的stride必须大于0。
  - padding_mode：支持"SAME"、"VALID"、"CALCULATED"三种模式。
  - pads：长度为4，按[top, bottom, left, right]指定填充量，仅在padding_mode为"CALCULATED"时生效。pads应大于等于0且小于对应的kernel size。
  - global_pooling：为True时，输出H和W均为1，ksize和pads被忽略。
  - ceil_mode：padding_mode为"SAME"或"VALID"时仅支持False。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_max_pool](examples/test_aclnn_max_pool.cpp) | 通过[aclnnMaxPool](docs/aclnnMaxPool.md)接口方式调用MaxPoolV3算子。 |
