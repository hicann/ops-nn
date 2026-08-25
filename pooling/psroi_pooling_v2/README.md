# PSROIPoolingV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

PSROIPoolingV2对输入特征图执行位置敏感RoI平均池化。对于每个RoI，算子将每个输出通道和池化位置映射到唯一输入通道，在保留RoI空间布局的同时完成区域聚合，适用于R-FCN等位置敏感目标检测网络。

输入`x`的逻辑shape为`[N,C,H,W]`，输入`rois`的shape为`[N,5,R]`。令`O=output_dim`、`G=group_size`，要求`C=O*G*G`，输出`y`的shape为`[N*R,O,G,G]`。每个输出位置对应的输入通道为：

$$
c_{\mathrm{in}} = o \times G^2 + p_h \times G + p_w
$$

算子先对RoI坐标执行`rint`，再对右下端点加1并乘以`spatial_scale`，随后裁剪到特征图范围，对对应分组区域求平均。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :----- | :------------ | :--- | :------- | :------- |
| `x` | 输入 | 输入特征图，shape为`[N,C,H,W]`。 | `float16`、`float32` | ND |
| `rois` | 输入 | RoI数据，shape为`[N,5,R]`，五个平面依次为`batch_id,x1,y1,x2,y2`。 | 与`x`相同 | ND |
| `y` | 输出 | 池化结果，shape为`[N*R,output_dim,group_size,group_size]`。 | 与`x`相同 | ND |
| `spatial_scale` | 属性 | 原图RoI坐标到特征图坐标的缩放比例，必填。 | Float | - |
| `output_dim` | 属性 | 输出通道数，必填。 | Int | - |
| `group_size` | 属性 | 位置敏感分组大小，输出空间为`group_size*group_size`，必填。 | Int | - |

## 约束说明

- `x`必须是4D ND Tensor，`rois`必须是3D ND Tensor，且`rois.shape[1]`必须为5。
- `x.shape[0]`必须等于`rois.shape[0]`，输入通道数必须满足`C=output_dim*group_size²`。
- `x`、`rois`和`y`仅支持全`float16`或全`float32`两种组合，不支持混合dtype。
- `spatial_scale`必须为有限正数，`output_dim`必须大于0，`group_size`的取值范围为`[1,127]`。
- `rois`中的坐标必须有限且非负，`round(batch_id)`必须位于`[0,N-1]`。
- 支持`N=0`或`R=0`的空输出；已知的`C`、`H`和`W`维必须大于0。
- 坐标处理顺序固定为“取整→右下端点加1→缩放”，不得交换取整与缩放顺序。
- 前向计算为确定性实现；`x`中的IEEE NaN/Inf按浮点语义自然传播。

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
    <td><a href="./examples/test_geir_psroi_pooling_v2.cpp">test_geir_psroi_pooling_v2</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
