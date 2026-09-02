# ROIPooling

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

- 算子功能：对输入特征图按ROI（Region of Interest）区域进行最大池化，输出固定尺寸的池化结果，用于目标检测等任务。

- 计算公式：

对每个ROI n（rois[n] = [batch_idx, x1, y1, x2, y2]）、通道c、池化位置(ph,pw)：

**1. ROI坐标映射**（使用分离的h/w缩放因子，roundf取整为int，终点坐标无+1偏移）：

$$
\text{roi\_start}_w = \text{round}(x1 \cdot \text{spatial\_scale\_w}), \quad \text{roi\_start}_h = \text{round}(y1 \cdot \text{spatial\_scale\_h})
$$

$$
\text{roi\_end}_w = \text{round}(x2 \cdot \text{spatial\_scale\_w}), \quad \text{roi\_end}_h = \text{round}(y2 \cdot \text{spatial\_scale\_h})
$$

> 以上坐标均为**整数**（int类型），+1偏移在ROI尺寸上（见下），不在坐标上。

**2. ROI尺寸**（int运算，+1在宽度/高度上，malformed ROI强制非空）：

$$
\text{roi\_w} = \max(\text{roi\_end}_w - \text{roi\_start}_w + 1, 1), \quad \text{roi\_h} = \max(\text{roi\_end}_h - \text{roi\_start}_h + 1, 1)
$$

**3. Bin切分与边界裁剪**（bin_size为float，基于int的roi_w/roi_h）：

$$
\text{bin\_size}_w = \frac{\text{roi\_w}}{\text{pooled\_w}}, \quad \text{bin\_size}_h = \frac{\text{roi\_h}}{\text{pooled\_h}}
$$

$$
\text{bin}_{x1} = \text{clamp}(\lfloor pw \cdot \text{bin\_size}_w \rfloor + \text{roi\_start}_w, 0, W)
$$

$$
\text{bin}_{x2} = \text{clamp}(\lceil (pw+1) \cdot \text{bin\_size}_w \rceil + \text{roi\_start}_w, 0, W)
$$

（y方向同理，clamp到[0, H]）

> floor/ceil基于相对偏移（不含roi_start），再加整数roi_start。

**4. Max Pooling**（空bin输出0）：

$$
y[n, c, ph, pw] = \begin{cases} 0 & \text{if } \text{bin}_{x2} \le \text{bin}_{x1} \text{ or } \text{bin}_{y2} \le \text{bin}_{y1} \\ \max_{h, w \in [\text{bin}_{y1}, \text{bin}_{y2}) \times [\text{bin}_{x1}, \text{bin}_{x2})} x[\text{batch\_idx}, c, h, w] & \text{otherwise} \end{cases}
$$

其中+1偏移在ROI尺寸上（`roi_width = roi_end - roi_start + 1`），是ROI Pooling通用标准（源自Fast R-CNN）

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
    <td>特征图，shape为[N,C,H,W]，公式中的x。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rois</td>
    <td>输入</td>
    <td>ROI框，shape为[K,5]，每行为[batch_idx,x1,y1,x2,y2]。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>roi_actual_num</td>
    <td>输入</td>
    <td>每个batch的实际ROI数量，可选输入，当前版本不参与计算。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>池化结果，shape为[K,C,pooled_h,pooled_w]，公式中的y。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>pooled_h</td>
    <td>属性</td>
    <td>池化输出高度。</td>
    <td>Int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pooled_w</td>
    <td>属性</td>
    <td>池化输出宽度。</td>
    <td>Int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>spatial_scale_h</td>
    <td>属性</td>
    <td>高度方向缩放因子。</td>
    <td>Float</td>
    <td>-</td>
  </tr>
  <tr>
    <td>spatial_scale_w</td>
    <td>属性</td>
    <td>宽度方向缩放因子。</td>
    <td>Float</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- 输入x仅支持4维[N,C,H,W]。
- 输入rois仅支持2维[K,5]，每行为[batch_idx,x1,y1,x2,y2]。
- roi_actual_num为可选输入，当前版本不参与计算。
- pooled_h和pooled_w必须大于0。
- spatial_scale_h和spatial_scale_w必须大于0。
- x与rois的dtype必须一致。
- 输出y的shape为[K,C,pooled_h,pooled_w]。
- ROI坐标用roundf取整为int，+1偏移在ROI尺寸上（roi_width = roi_end - roi_start + 1）。
- malformed ROI强制非空，roi_size取max(size,1)（int运算）。
- 空bin输出0。

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
    <td><a href="./examples/test_geir_roi_pooling.cpp">test_geir_roi_pooling</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
