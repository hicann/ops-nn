# MaxPoolV3

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|<term>Ascend 950PR/Ascend 950DT</term>   |     √    |

## 功能说明

- 接口功能：
对于dim=3 或4维的输入张量，进行最大池化（max pooling）操作。
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
      <td>kernelShape</td>
      <td>输入</td>
      <td>最大池化的窗口大小。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>输入</td>
      <td>窗口移动的步长。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>autoPad</td>
      <td>输入</td>
      <td>指定padding的方式。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>输入</td>
      <td>沿着空间轴方向开始和结束的位置填充，对应公式中的padding_size。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>输入</td>
      <td>沿着核空间轴方向的膨胀值，对应公式中的dilation_size。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>输入</td>
      <td>计算输出形状的取整模式。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出的tensor。</td>
      <td>FLOAT16、FLOAT、BF16、INT32、INT64、UINT8、INT16、INT8、UINT16。</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明
- **值域限制说明：**
  - kernelShape：对应公式中的k_h和k_w，长度为1或2，且数组元素必须都大于0。
  - strides：对应公式中的s_h和s_w，数组长度为0、1或2，且数组元素必须都大于0。当数组长度为0时，strides取默认值为1。
  - autoPad：其中0代表"NOTSET"，并且只支持数值0。
  - pads：长度为0、1、2或4。当数组长度为0时，不进行填充。当数组长度为1时，H_top、H_bottom、W_left、W_right填充同一个值。当数组长度为2时，H_top、H_bottom分别填充数组第1个值，W_left、W_right分别填充数组第2个值。当数组长度为4时，按[H_top、W_left、H_bottom、W_right]位置关系进行填充。单个空间轴方向填充量之和需小于等于对应方向kernelShape。
  - dilations：只支持数值为1的输入场景。长度为0、1、2或4。
  - ceilMode：取值为0时，代表False，向下取整；非0值时，代表True，向上取整。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_max_pool](examples/test_aclnn_max_pool.cpp) | 通过[aclnnMaxPool](docs/aclnnMaxPool.md)接口方式调用MaxPoolV3算子。 |