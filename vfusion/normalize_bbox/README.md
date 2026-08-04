# NormalizeBBox

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：将目标检测预选框的绝对像素坐标，按对应图像的高（height）与宽（width）逐batch归一化到`[0, 1]`区间。每个框由4个坐标构成，y系坐标除以height、x系坐标除以width。
- 计算公式：

  对第b个batch，令$h = shapeHw[b, 0]$、$w = shapeHw[b, 1]$（int32转float）：

  $$
  y_{b,i} = boxes_{b,i} / [h, w, h, w]
  $$

  其中$i$遍历该batch内的每个框。boxes第0维为batch，coord轴（大小为4）与batch之间可有任意多个中间维（其乘积为每batch的框数num），boxes的rank支持2~8。`reversedBox`仅改变coord轴在内存中的位置，不改变数学语义：

  - `reversedBox = false`：boxes形如$(batch, \ldots, 4)$，coord轴在最后一维（如$(batch, num, 4)$、$(batch, H, W, 4)$）。
  - `reversedBox = true`：boxes形如$(batch, 4, \ldots)$，coord轴在第1维（如$(batch, 4, num)$、$(batch, 4, H, W)$）。

## 参数说明

<table style="table-layout: auto; width: 100%">
<thead>
    <tr>
    <th style="white-space: nowrap">参数名</th>
    <th style="white-space: nowrap">输入/输出/属性</th>
    <th style="white-space: nowrap">描述</th>
    <th style="white-space: nowrap">数据类型</th>
    <th style="white-space: nowrap">数据格式</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>boxes</td>
    <td>输入</td>
    <td>表示传入的框绝对坐标张量，公式中的boxes。rank 2-8，dim0=batch，coord轴大小为4，中间维乘积为num。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>shapeHw</td>
    <td>输入</td>
    <td>逐batch的图像尺寸，[0]=height、[1]=width。rank 2，dim1 == 3。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>reversedBox</td>
    <td>属性</td>
    <td>可选，boxes坐标排布开关，默认false。</td>
    <td>BOOL</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>表示归一化后的输出张量，公式中的y。dtype/shape与boxes一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
    </tr>
</tbody>
</table>

## 约束说明

- `shapeHw`的batch（dim0）必须与`boxes`的batch（dim0）一致。
- `boxes`的rank须在[2, 8]范围内。`reversedBox = false`时`boxes`末维须为4；`reversedBox = true`时`boxes`的dim1须为4。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式接口  | [test_geir_normalize_bbox.cpp](examples/arch35/test_geir_normalize_bbox.cpp) | 通过GE图模式构建NormalizeBBox算子图并执行RunGraph验证。 |
