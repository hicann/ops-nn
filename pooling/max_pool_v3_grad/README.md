# MaxPoolV3Grad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：MaxPoolV3Grad是MaxPoolV3（最大池化）算子的反向传播（梯度计算）算子，用于在神经网络训练过程中将损失函数对池化输出的梯度回传到池化输入。对于每个池化输出位置，其梯度仅传递给前向计算中产生该最大值的输入位置（argmax位置），其余输入位置的梯度为0。采用first-wins策略，当窗口内有多个输入位置等于最大值时，仅首个匹配位置接收梯度。

- 计算公式：

对于每个池化输出位置$(ho, wo)$，找到argmax位置$(h^*, w^*)$：

$$
out\_grad[n, c, h^*, w^*] \mathrel{+}= grad[n, c, ho, wo]
$$

$$
out\_grad[n, c, h, w] \mathrel{+}= 0 \quad \text{for } (h, w) \neq (h^*, w^*)
$$

其中argmax位置通过比较orig_input和orig_output确定：若$orig\_input[n, c, h, w] == orig\_output[n, c, ho, wo]$，则$(h, w)$是argmax位置。

## 参数说明

|参数名|输入/输出/属性|描述|数据类型|数据格式|
|-----|-----------|----|---------|------|
|orig_input|输入|前向MaxPoolV3的原始输入，4D张量。data_format为"NCHW"时布局为(N, C, H, W)，为"NHWC"时布局为(N, H, W, C)。|FLOAT16、FLOAT|ND|
|orig_output|输入|前向MaxPoolV3的原始输出，4D张量。shape由orig_input和ksize/strides/padding推导，dtype与orig_input一致。|FLOAT16、FLOAT|ND|
|grad|输入|输出梯度，4D张量，shape和dtype与orig_output完全一致。|FLOAT16、FLOAT|ND|
|out_grad|输出|输入梯度，4D张量，shape和dtype与orig_input一致。|FLOAT16、FLOAT|ND|
|ksize|属性|池化窗口大小，长度为4的列表。N维和C维必须为1，H/W维取值范围为[1, 255]。|INT64|-|
|strides|属性|池化步长，长度为4的列表。N维和C维必须为1，H/W维取值范围为[1, 63]。|INT64|-|
|padding_mode|属性|padding模式，支持"CALCULATED"、"SAME"、"VALID"，默认"CALCULATED"。|STRING|-|
|pads|属性|padding大小[pad_top, pad_bottom, pad_left, pad_right]，仅在padding_mode为"CALCULATED"时生效，每个值必须大于等于0，默认[0, 0, 0, 0]。|INT64|-|
|data_format|属性|逻辑数据格式，支持"NCHW"、"NHWC"，默认"NCHW"。决定H、W、C维度的索引位置，输入输出均为4D ND格式，不做5HD格式转换。|STRING|-|
|global_pooling|属性|是否全局池化，不支持全局池化的梯度计算，必须为false，默认false。|BOOL|-|
|ceil_mode|属性|是否使用ceil模式计算输出尺寸，仅在padding_mode为"CALCULATED"时生效，默认false。|BOOL|-|

## 约束说明

- **数据格式约束：**
  - 输入输出张量为4D ND格式，支持NCHW和NHWC两种逻辑排布，由data_format属性指定。
  - data_format属性（NCHW/NHWC）决定H、W、C维度的索引位置，算子内部按该格式直接计算4D地址，不做5HD格式转换。
  - 非连续张量通过AutoContiguous自动转为连续后计算。

- **数据类型约束：**
  - orig_input、orig_output、grad三个输入必须使用相同的dtype，输出out_grad也使用相同dtype。
  - 仅支持float16和float32两种dtype，暂不支持bfloat16。
  - 不支持跨类型计算。

- **shape约束：**
  - grad的shape必须与orig_output的shape完全一致。
  - orig_output的N维和C维必须与orig_input一致（N/C维在池化过程中不变）。
  - out_grad的shape与orig_input的shape完全一致。

- **ksize约束：**
  - ksize数组长度必须为4。
  - N维和C维的ksize必须为1（不参与池化）。NCHW格式下ksize[0]和ksize[1]为1，NHWC格式下ksize[0]和ksize[3]为1。
  - H/W维ksize取值范围为[1, 255]，超出范围报错。

- **strides约束：**
  - strides数组长度必须为4。
  - N维和C维的strides必须为1。NCHW格式下strides[0]和strides[1]为1，NHWC格式下strides[0]和strides[3]为1。
  - H/W维strides取值范围为[1, 63]，超出范围报错。

- **padding_mode约束：**
  - 仅支持"CALCULATED"、"SAME"、"VALID"三种模式，其他值报错。
  - pads属性仅在padding_mode为"CALCULATED"时生效，pads长度必须为4，顺序为[pad_top, pad_bottom, pad_left, pad_right]，每个值必须大于等于0。
  - 当padding_mode为"SAME"或"VALID"时，ceil_mode必须为false。

- **global_pooling约束：**
  - global_pooling必须为false，不支持全局池化的梯度计算，设为true时报错。

- **特殊值处理：**
  - NaN输入：NaN不参与比较（NaN==任何值返回false），对应位置梯度为0。
  - Inf输入：Inf作为最大值正常参与比较和梯度传播（Inf==Inf为true）。
  - +0.0与-0.0遵循IEEE 754标准，比较相等（+0.0==-0.0为true）。
  - 多个输入位置具有相同最大值时，仅第一个最大值位置接收梯度（first-wins策略，与TensorFlow一致）。
  - padding区域不接收梯度。

- **边界条件处理：**
  - 空张量（shape含0维）：返回全0的out_grad，shape与orig_input一致。
  - 非连续张量：通过AutoContiguous自动转为连续后计算。
  - ceil_mode=True时部分窗口可能超出输入边界，超出部分不接收梯度。

- **确定性保证：**
  - 非重叠路径（stride>=kernel）：无写入冲突，直接赋值，完全确定。
  - 重叠路径（stride<kernel）：使用FP32 atomicAdd累加保证精度，argmax选择由first-wins策略保证确定。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| GE图模式 | [test_geir_max_pool_v3_grad](examples/test_geir_max_pool_v3_grad.cpp) | 通过[算子IR](op_graph/max_pool_v3_grad_proto.h)构图方式调用MaxPoolV3Grad算子，详见[算子调用](../../docs/zh/invocation/quick_op_invocation.md)。 |
