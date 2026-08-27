# MaxPoolV3Grad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：MaxPoolV3Grad是MaxPoolV3（最大池化）算子的反向传播（梯度计算）算子，用于在神经网络训练过程中将损失函数对池化输出的梯度回传到池化输入。对于每个池化输出位置，其梯度仅传递给前向计算中产生该最大值的输入位置（argmax位置），其余输入位置的梯度为0。

- argmax位置通过从orig_input重新计算每个池化窗口的最大值确定（不依赖orig_output）。当窗口内存在多个输入位置等于最大值时，仅首个匹配位置接收梯度（first-wins策略，与TensorFlow一致）。

- 计算公式：
  反向传播将梯度回传到 argmax 位置，对于每个池化输出位置 $(ho, wo)$，从 orig_input 对应窗口中找到 argmax 位置 $(h^*, w^*)$：

  $$
  out\_grad[n, c, h^*, w^*] \mathrel{+}= grad[n, c, ho, wo]
  $$

  $$
  out\_grad[n, c, h, w] \mathrel{+}= 0 \quad \text{for } (h, w) \neq (h^*, w^*)
  $$

  out_grad 的 H/W 维度与 orig_input 一致，orig_output 的 H/W 维度按 padding_mode 推导如下。设 H 维池化窗口大小为 $k_h$、步长为 $s_h$，W 维池化窗口大小为 $k_w$、步长为 $s_w$：

  - 当 padding_mode 为 "SAME" 时，输出尺寸为：

    $$
    H_{out} = \lceil \frac{H_{in}}{s_h} \rceil, \quad W_{out} = \lceil \frac{W_{in}}{s_w} \rceil
    $$

    总 padding 大小为：

    $$
    pad_h = \max((H_{out} - 1) \times s_h + k_h - H_{in}, 0), \quad pad_w = \max((W_{out} - 1) \times s_w + k_w - W_{in}, 0)
    $$

    其中 $pad_{top} = \lfloor pad_h / 2 \rfloor$、$pad_{bottom} = pad_h - pad_{top}$，$pad_{left} = \lfloor pad_w / 2 \rfloor$、$pad_{right} = pad_w - pad_{left}$。

  - 当 padding_mode 为 "VALID" 时，无 padding，输出尺寸为：

    $$
    pad_{top} = pad_{bottom} = pad_{left} = pad_{right} = 0
    $$

    $$
    H_{out} = \lfloor \frac{H_{in} - k_h}{s_h} \rfloor + 1, \quad W_{out} = \lfloor \frac{W_{in} - k_w}{s_w} \rfloor + 1
    $$

  - 当 padding_mode 为 "CALCULATED" 时，$pad_{top} = pads[0]$、$pad_{bottom} = pads[1]$、$pad_{left} = pads[2]$、$pad_{right} = pads[3]$：

    - 当 ceil_mode 为 false 时，输出尺寸为：

      $$
      H_{out} = \lfloor \frac{H_{in} + pad_{top} + pad_{bottom} - k_h}{s_h} \rfloor + 1
      $$

      $$
      W_{out} = \lfloor \frac{W_{in} + pad_{left} + pad_{right} - k_w}{s_w} \rfloor + 1
      $$

    - 当 ceil_mode 为 true 时，输出尺寸为：

      $$
      H_{out} = \lceil \frac{H_{in} + pad_{top} + pad_{bottom} - k_h}{s_h} \rceil + 1
      $$

      $$
      W_{out} = \lceil \frac{W_{in} + pad_{left} + pad_{right} - k_w}{s_w} \rceil + 1
      $$

      若滑窗左上角起始位置落在下侧/右侧 pad 填充位或界外（无法取到有效值），则该滑窗结果被舍弃，对应空间轴 shape 减 1：

      $$
      \begin{cases}
      H_{out} = H_{out} - 1 & \text{if } (H_{out} - 1) \times s_h \ge H_{in} + pad_{top} \\
      W_{out} = W_{out} - 1 & \text{if } (W_{out} - 1) \times s_w \ge W_{in} + pad_{left}
      \end{cases}
      $$

  - 当 global_pooling 为 true 时，池化窗口覆盖整个 H/W 维度：

    $$
    k_h = H_{in}, \quad k_w = W_{in}, \quad pad_{top} = pad_{bottom} = pad_{left} = pad_{right} = 0, \quad H_{out} = W_{out} = 1
    $$

## 参数说明

|参数名|输入/输出/属性|描述|数据类型|数据格式|
|-----|-----------|----|---------|------|
|orig_input|输入|前向MaxPoolV3的原始输入，4D张量。data_format为"NCHW"时布局为(N, C, H, W)，为"NHWC"时布局为(N, H, W, C)。|FLOAT16、FLOAT、BFLOAT16、INT32、INT64、UINT8、INT16、INT8、UINT16|ND|
|orig_output|输入|前向MaxPoolV3的原始输出，4D张量。shape由orig_input和ksize/strides/padding推导，dtype与orig_input一致。|FLOAT16、FLOAT、BFLOAT16、INT32、INT64、UINT8、INT16、INT8、UINT16|ND|
|grad|输入|输出梯度，4D张量，shape和dtype与orig_output完全一致。|FLOAT16、FLOAT、BFLOAT16、INT32、INT64、UINT8、INT16、INT8、UINT16|ND|
|out_grad|输出|输入梯度，4D张量，shape和dtype与orig_input一致。|FLOAT16、FLOAT、BFLOAT16、INT32、INT64、UINT8、INT16、INT8、UINT16|ND|
|ksize|属性|池化窗口大小，长度为4的列表。N维和C维必须为1，H/W维取值范围为[1, 255]。global_pooling为true时忽略。|INT64|-|
|strides|属性|池化步长，长度为4的列表。N维和C维必须为1，H/W维取值范围为[1, 63]。|INT64|-|
|padding_mode|属性|padding模式，支持"CALCULATED"、"SAME"、"VALID"，默认"CALCULATED"。|STRING|-|
|pads|属性|padding大小，长度为4的列表，顺序为[pad_top, pad_bottom, pad_left, pad_right]，仅在padding_mode为"CALCULATED"时生效，每个值必须大于等于0，默认[0, 0, 0, 0]。|INT64|-|
|data_format|属性|逻辑数据格式，支持"NCHW"、"NHWC"，默认"NCHW"。决定H、W、C维度的索引位置，输入输出均为4D ND格式，不做5HD格式转换。|STRING|-|
|global_pooling|属性|是否全局池化，默认false。为true时忽略ksize和pads，池化窗口覆盖整个H/W维度，输出H/W维度为1。|BOOL|-|
|ceil_mode|属性|是否使用ceil模式计算输出尺寸，默认false。仅在padding_mode为"CALCULATED"时生效。|BOOL|-|

## 约束说明

- **数据格式约束：**
  - 输入输出张量为4D ND格式，支持NCHW和NHWC两种逻辑排布，由data_format属性指定。
  - data_format属性（NCHW/NHWC）决定H、W、C维度的索引位置，算子内部按该格式直接计算4D地址，不做5HD格式转换。
  - 非连续张量通过AutoContiguous自动转为连续后计算。

- **数据类型约束：**
  - orig_input、orig_output、grad三个输入必须使用相同的dtype，输出out_grad也使用相同dtype。
  - 支持FLOAT16、FLOAT、BFLOAT16、INT32、INT64、UINT8、INT16、INT8、UINT16九种dtype。
  - 不支持跨类型计算。

- **shape约束：**
  - grad的shape必须与orig_output的shape完全一致。
  - orig_output的N维和C维必须与orig_input一致（N/C维在池化过程中不变）。
  - out_grad的shape与orig_input的shape完全一致。

- **ksize约束：**
  - ksize数组长度必须为4。
  - N维和C维的ksize必须为1（不参与池化）。NCHW格式下ksize[0]和ksize[1]为1，NHWC格式下ksize[0]和ksize[3]为1。
  - H/W维ksize取值范围为[1, 255]，超出范围报错。
  - global_pooling为true时忽略ksize，池化窗口取输入H/W维大小。

- **strides约束：**
  - strides数组长度必须为4。
  - N维和C维的strides必须为1。NCHW格式下strides[0]和strides[1]为1，NHWC格式下strides[0]和strides[3]为1。
  - H/W维strides取值范围为[1, 63]，超出范围报错。

- **padding_mode约束：**
  - 仅支持"CALCULATED"、"SAME"、"VALID"三种模式，其他值报错。
  - pads属性仅在padding_mode为"CALCULATED"时生效，pads长度必须为4，顺序为[pad_top, pad_bottom, pad_left, pad_right]，每个值必须大于等于0。
  - ceil_mode仅在padding_mode为"CALCULATED"时生效；SAME、VALID模式下忽略ceil_mode。

- **global_pooling约束：**
  - global_pooling支持true和false。为true时池化窗口覆盖整个H/W维度，输出H/W维度为1，忽略ksize和pads。

- **特殊值处理：**
  - NaN输入：窗口内存在数值（非NaN）时，NaN不参与最大值比较；窗口内全为NaN时，梯度回传给行优先扫描遇到的第一个有效输入位置（与TensorFlow MaxPoolGrad一致）。
  - Inf输入：Inf作为最大值正常参与比较和梯度传播。
  - +0.0与-0.0遵循IEEE 754标准，比较相等。
  - 多个输入位置具有相同最大值时，仅第一个最大值位置接收梯度（first-wins策略）。
  - padding区域不接收梯度。

- **边界条件处理：**
  - 空张量（shape含0维）：返回全0的out_grad，shape与orig_input一致。
  - 非连续张量：通过AutoContiguous自动转为连续后计算。
  - ceil_mode为true时部分窗口可能超出输入边界，超出部分不接收梯度。

- **确定性保证：**
  - 每个线程独占一个out_grad元素做梯度聚合，不存在对同一地址的并发写，无需原子加即可保证确定性。
  - argmax选择由first-wins策略保证确定。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| GE图模式 | [test_geir_max_pool_v3_grad](examples/test_geir_max_pool_v3_grad.cpp) | 通过[算子IR](op_graph/max_pool_v3_grad_proto.h)构图方式调用MaxPoolV3Grad算子，详见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)。 |
