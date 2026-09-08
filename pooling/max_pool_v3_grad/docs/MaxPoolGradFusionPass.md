# MaxPoolGradFusionPass

## 融合模式

<!-- npu="950" id1 -->
网络中的MaxPoolGrad算子在支持的型号上没有对应的算子二进制实现。该融合规则将图中符合条件的MaxPoolGrad算子整体替换为MaxPoolV3Grad算子，输入、输出保持一一对应，属性按等价语义改写。MaxPoolGradFusionPass采用的属性转换规则，与MaxPoolFusionPass将MaxPool转换为MaxPoolV3时采用的属性转换规则保持一致。如下图所示。

![](../../../docs/zh/figures/MaxPoolGradFusionPass_1.png)

<!-- end id1 -->

融合前后的输入输出对应关系如下。

| **融合前MaxPoolGrad** | **融合后MaxPoolV3Grad** | **必选/可选** | **说明** |
| --- | --- | --- | --- |
| x1（输入0） | orig_input（输入0） | 必选 | 前向池化的原始输入。 |
| x2（输入1） | orig_output（输入1） | 必选 | 前向池化的原始输出。 |
| grad（输入2） | grad（输入2） | 必选 | 反向传入的梯度。 |
| y（输出0） | out_grad（输出0） | 必选 | 反向输出的梯度，输出描述沿用融合前节点的输出描述。 |

融合前后的属性对应关系如下。标记为必选的属性在融合前的MaxPoolGrad节点上必须存在且满足说明中的取值要求，否则不融合。

| **融合后MaxPoolV3Grad属性** | **取值来源** | **必选/可选** | **说明** |
| --- | --- | --- | --- |
| ksize | MaxPoolGrad的ksize属性 | 必选 | 属性名和属性值均由MaxPoolGrad原样透传至MaxPoolV3Grad，元素个数必须为4。 |
| strides | MaxPoolGrad的strides属性 | 必选 | 属性名和属性值均由MaxPoolGrad原样透传至MaxPoolV3Grad，元素个数必须为4。 |
| padding_mode | MaxPoolGrad的padding属性 | 必选 | 由MaxPoolGrad的padding属性转换得到：属性名改为padding_mode，属性值保持不变。padding仅支持SAME和VALID，其他取值不融合。 |
| pads | 固定值`{0, 0, 0, 0}` | - | MaxPoolGrad无该属性，融合时新增；padding_mode为SAME或VALID时，pads不生效。 |
| data_format | MaxPoolGrad的data_format属性 | 必选 | 属性名和属性值均由MaxPoolGrad原样透传至MaxPoolV3Grad，仅支持NCHW和NHWC。 |
| global_pooling | 固定值`false` | - | MaxPoolGrad无该属性，融合时新增，并设置为非全局池化。 |
| ceil_mode | 固定值`false` | - | MaxPoolGrad无该属性，融合时新增，并设置为向下取整模式。 |

> [!NOTE] 说明
>
> data_format以MaxPoolGrad节点的属性为准。节点输入描述中的物理Format可能为ND，不以该物理Format判断数据排布，也不会因其为ND而拒绝融合。

## 使用约束

- 融合在InferShape之后的阶段执行，匹配的源算子类型为MaxPoolGrad，替换后的目标算子类型为MaxPoolV3Grad。
- 完整链路支持的数据类型为FLOAT16、FLOAT32、BFLOAT16。
- 匹配的图结构为单个MaxPoolGrad节点，具有x1、x2、grad三个输入和y一个输出，输入输出个数不满足时不匹配。
- 该融合规则默认开启，当前不支持关闭。

## 支持的型号

<!-- npu="950" id2 -->
Ascend 950PR/Ascend 950DT
<!-- end id2 -->
