# AvgPoolV2Grad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    x     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    x     |
| <term>Atlas 推理系列产品</term>                             |    x     |
| <term>Atlas 训练系列产品</term>                             |    x     |

## 功能说明

- 接口功能：
二维平均池化的反向传播，计算二维平均池化正向传播的输入梯度。

- 计算公式：
  - 对于前向平均池化输出梯度，计算原始输入的梯度：
    
    $$
    out\_grad(i,j) = \frac{1}{divisor} \sum_{k \in pooling\_window} input\_grad(k)
    $$
    
    其中divisor为池化窗口大小（若exclusive为true，则不包含padding区域；若指定divisor_override则使用该值）。

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
      <td>orig_input_shape</td>
      <td>输入</td>
      <td>原始输入的形状，描述前向AvgPoolV2的输入维度[N,C,H,W]或[N,H,W,C]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_grad</td>
      <td>输入</td>
      <td>平均池化输出的梯度张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>池化窗口的大小，长度为4，数值大于0。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>滑动窗口的步长，长度为4，数值大于0。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>属性</td>
      <td>填充算法类型，支持"VALID"、"SAME"或"CALCULATED"。默认"CALCULATED"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>输入特征图的填充大小，默认{0,0,0,0}。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>数据格式，支持"NCHW"或"NHWC"。默认"NCHW"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>global_pooling</td>
      <td>属性</td>
      <td>是否使用全局池化，若为true则忽略ksize和pads。默认false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>属性</td>
      <td>推导的输出out的shape是否向上取整。默认false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>exclusive</td>
      <td>属性</td>
      <td>计算平均池化时是否排除padding点。默认true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisor_override</td>
      <td>属性</td>
      <td>指定除数覆盖值，若为0则使用池化区域大小。默认0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out_grad</td>
      <td>输出</td>
      <td>输出的梯度张量，形状与orig_input_shape相同。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- **值域限制说明：**
  - orig_input_shape：必须是一维张量，长度为4，描述[N,C,H,W]或[N,H,W,C]。
  - input_grad：4D张量，支持FLOAT16、FLOAT、BFLOAT16类型。
  - ksize：长度为 4，必须大于0，指定池化窗口大小。
  - strides：长度为 4，必须大于0，指定滑动步长。
  - padding_mode：只支持"VALID"、"SAME"、"CALCULATED"三种模式。
  - pads：长度为 4，必须大于0，仅在padding_mode为"CALCULATED"时生效。不超过ksize对应位置的1/2。
  - data_format：支持"NCHW"或"NHWC"。
  - global_pooling：若为true，ksize 和 pads 将被忽略。
  - ceil_mode：若为true，使用ceil计算输出形状；否则使用floor。
  - exclusive：若为true，计算均值时排除padding区域。
  - divisor_override：若非0，将作为除数使用；否则使用池化区域大小。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_avg_pool_v2_grad](examples/test_aclnn_avg_pool_v2_grad.cpp) | 通过[aclnnAvgPool2dBackward](../../pooling/avg_pool3_d_grad/docs/aclnnAvgPool2dBackward.md)接口方式调用AvgPoolV2Grad算子。 |