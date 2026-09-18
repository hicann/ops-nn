# PSROIPoolingGradV2D

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |     √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |     √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

Ascend950使用本目录新增的arch35实现；Atlas A2/A3训练及推理系列产品使用CANN已有的TBE实现。两者的数据格式和约束不同，不能将Ascend950算子包直接用于A2/A3。

| 实现 | x/y数据类型和存储格式 | rois数据类型和存储格式 | ROI数量约束 |
| :-- | :-- | :-- | :-- |
| Ascend950（本目录arch35） | FLOAT32、ND或NCHW，x/y格式须一致 | FLOAT32、ND | R为非负整数，支持R=0 |
| Atlas A2/A3（CANN已有TBE） | FLOAT32、NC1HWC0，原始格式为NCHW | FLOAT32、ND | R为正整数且为16的倍数 |

## 功能说明

PSROIPoolingGradV2D将PS ROI Pooling的上游梯度按位置敏感通道映射和bin面积散射回输入特征图空间。输入`x`的逻辑shape为`[B*R, O, G, G]`，`rois`的shape为`[B, 5, R]`，输出`y`的逻辑shape为`[B, O*G*G, H, W]`，其中`O=output_dim`、`G=group_size`、`[H,W]=input_size`。

Ascend950实现中，同一输出元素的重叠贡献按固定的batch/ROI顺序累加；ROI bin边界使用分离的FP32乘法与加法，禁止FMA收缩，以保证离散覆盖区域和结果确定性。该确定性说明不扩展至已有TBE实现。

## 算子原型

公共图模式原型见[psroi_pooling_grad_v2_d_proto.h](./op_graph/psroi_pooling_grad_v2_d_proto.h)，注册名为`PSROIPoolingGradV2D`。输入顺序为`x`、`rois`，输出为`y`；`spatial_scale`、`output_dim`、`group_size`、`input_size`均为必需属性。

公共`REG_OP`为兼容既有接口保留FLOAT16/FLOAT32声明；这不代表具体产品实现支持FLOAT16。上述实现均仅支持FLOAT32，且`x`、`rois`、`y`的数据类型必须一致。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :-- | :--: | :-- | :--: | :--: |
| x | 输入 | 上游梯度，逻辑shape为`[B*R, O, G, G]`。 | FLOAT32 | 950为ND或NCHW；A2/A3为NC1HWC0 |
| rois | 输入 | shape为`[B, 5, R]`，五个平面依次存储`batch_id, x1, y1, x2, y2`，即第r个ROI为`rois[b, :, r]`。 | FLOAT32 | ND |
| spatial_scale | 属性 | ROI坐标到特征图坐标的缩放比例，必须有限且大于0。 | FLOAT | - |
| output_dim | 属性 | 池化输出通道数O，必须大于0。 | INT | - |
| group_size | 属性 | 位置敏感分组大小G，取值范围为`[1, 127]`。 | INT | - |
| input_size | 属性 | 输出空间尺寸`[H, W]`，必须恰有两个元素且均大于0。 | LIST_INT | - |
| y | 输出 | 输入特征图梯度，逻辑shape为`[B, O*G*G, H, W]`。 | FLOAT32 | 950为ND或NCHW，须与x一致；A2/A3为NC1HWC0 |

## 约束说明

- ROI四个坐标必须有限且大于等于0；`batch_id`必须为整数且位于`[0, B)`。调用方须保证这些张量值约束，shape推导不会校验张量内容。
- Ascend950：`x`和`y`必须均为4维ND或均为4维NCHW，不能混用；两种格式使用相同的连续4维布局。`rois`必须为3维ND且`rois.shape[1] == 5`；`x.shape == [B*R, output_dim, group_size, group_size]`。
- Ascend950图模式构图使用`x/y=NCHW、rois=ND`，ND计算入口仍保持支持。
- Ascend950：`R == 0`时返回正确shape的全零输出；相同合法输入多次运行的结果bit-exact。
- Atlas A2/A3：`x`、`y`的逻辑格式为NCHW，物理存储格式为NC1HWC0，物理shape分别为`[B*R, ceil(O/16), G, G, 16]`、`[B, ceil(O*G*G/16), H, W, 16]`；`rois`为`[B, 5, R]`的ND张量，R必须为正的16倍数。
- A2/A3的支持来自CANN已有实现，需安装对应产品的CANN算子包；本目录的Ascend950构建与示例不提供A2/A3内核，也未声明两种实现之间的bit-exact一致性。

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
    <td><a href="./examples/test_geir_psroi_pooling_grad_v2_d.cpp">test_geir_psroi_pooling_grad_v2_d</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
