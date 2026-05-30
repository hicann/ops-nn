# MaxPoolV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|<term>Ascend 950PR/Ascend 950DT</term>   |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |     x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     x     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |     x     |
| <term>Atlas 推理系列产品</term>                             |     x     |
| <term>Atlas 训练系列产品</term>                             |     x     |

## 功能说明

- 接口功能：
对于4维的输入张量，进行最大池化（max pooling）操作。支持NHWC和NCHW两种数据格式，支持SAME和VALID两种填充模式。
- 计算公式：
  - 当padding="VALID"时，out tensor的shape中H和W维度推导公式：

    $$
    [H_{out}, W_{out}]=[\lfloor{\frac{H_{in} - k_h}{s_h}}\rfloor + 1,\lfloor{\frac{W_{in} - k_w}{s_w}}\rfloor + 1]
    $$

  - 当padding="SAME"时，out tensor的shape中H和W维度推导公式：

    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in}}{s_h}}\rceil, \lceil{\frac{W_{in}}{s_w}}\rceil]
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
      <td>输入tensor，维度为4。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>输入</td>
      <td>最大池化的窗口大小，长度为4的列表。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>输入</td>
      <td>窗口移动的步长，长度为4的列表。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性</td>
      <td>填充模式，取值为"SAME"或"VALID"。SAME模式表示输出大小为输入大小除以步长向上取整，VALID模式表示输出大小为输入大小减去窗口大小再除以步长加1。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>数据格式，支持"NHWC"或"NCHW"，默认为"NHWC"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出的tensor。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- **值域限制说明：**
  - x：输入tensor维度必须为4。
  - ksize：长度为4的列表，其中ksize[0] = 1或ksize[3] = 1，ksize[1] * ksize[2] <= 255。
  - strides：长度为4的列表，其中strides[0] = 1或strides[3] = 1，strides[1] <= 63，strides[0] >= 1，strides[2] <= 63，strides[2] >= 1。
  - padding：取值必须为"SAME"或"VALID"。
  - data_format：取值必须为"NHWC"或"NCHW"，默认为"NHWC"。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_max_pool_v2](examples/test_geir_max_pool_v2.cpp) | 通过[算子IR](op_graph/max_pool_v2_proto.h)接口方式调用MaxPoolV2算子。 |