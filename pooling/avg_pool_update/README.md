# AvgPoolUpdate

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |     √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |     √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     √    |
| <term>Atlas 推理系列产品</term>                               |     √    |
| <term>Atlas 训练系列产品</term>                               |     √    |

## 功能说明

- 算子功能：计算平均池化的更新值。将求和池化结果除以池化窗口实际覆盖的有效元素个数，得到平均值。x1为求和池化输出，x2为原始输入feature map（仅用于获取输入空间尺寸，不参与数值计算）。

- 计算公式：

$$
y = x1 \oslash \text{mean\_matrix}
$$

其中mean_matrix为池化除数矩阵：

$$
\text{mean\_matrix}[h, w] = \text{mean\_h} \times \text{mean\_w}
$$

$$
\text{mean\_h} = \max(\min(\min(h \cdot s_h - p_t + k_h,\; (H_{out}-1-h) \cdot s_h - p_b + k_h),\; \min(k_h,\; H_{in})),\; 1)
$$

$$
\text{mean\_w} = \max(\min(\min(w \cdot s_w - p_l + k_w,\; (W_{out}-1-w) \cdot s_w - p_r + k_w),\; \min(k_w,\; W_{in})),\; 1)
$$

式中，$s_h$、$s_w$为strides的H/W分量，$p_t$、$p_b$、$p_l$、$p_r$为pads的上/下/左/右分量，$k_h$、$k_w$为ksize的H/W分量，$H_{in}$、$W_{in}$为x2的空间输入尺寸，$H_{out}$、$W_{out}$为x1的空间输出尺寸。mean_matrix计算使用int64中间变量，除法前cast为x1的dtype。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>x1</td>
      <td>输入</td>
      <td>求和池化输出结果，公式中的x1。shape为(N, C, H_out, W_out)（NCHW）或(N, H_out, W_out, C)（NHWC）。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>原始输入feature map，公式中的x2。仅用于获取输入空间尺寸H_in/W_in，不参与数值计算。shape为(N, C, H_in, W_in)（NCHW）或(N, H_in, W_in, C)（NHWC）。</td>
      <td>INT4、INT8、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>平均池化更新结果，公式中的y。shape与x1相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>池化窗口大小，长度为4的列表，各维度顺序与data_format一致。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>池化步长，长度为4的列表，各维度顺序与data_format一致。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>属性</td>
      <td>填充模式，取值范围：CALCULATED、VALID、SAME。默认值为CALCULATED。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>填充值，长度为4的列表(top, bottom, left, right)。仅在padding_mode为CALCULATED时生效。默认值为{0, 0, 0, 0}。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>数据格式，取值范围：NCHW、NHWC。默认值为NHWC。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>属性</td>
      <td>是否使用ceil模式计算输出尺寸。true为ceil，false为floor。默认值为false。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>exclusive</td>
      <td>属性</td>
      <td>是否排除padding区域计入窗口。true为排除padding（使用实际覆盖元素个数），false为使用常量窗口大小。默认值为true。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- **exclusive约束**：exclusive必须为true。当exclusive为false时算子无意义（池化因子为常量，无需更新），会报错退出。
- **padding_mode与ceil_mode约束**：当padding_mode为VALID且ceil_mode为false时，算子无意义（池化因子为常量），会报错退出。
- **输入维度约束**：x1和x2必须为4D张量，支持NCHW和NHWC两种data_format。不支持其他维度数。
- **ksize约束**：ksize为长度4的列表，各分量必须为正整数。H/W分量（k_h, k_w）的取值由data_format决定位置。
- **strides约束**：strides为长度4的列表，各分量必须为正整数。H/W分量（s_h, s_w）的取值由data_format决定位置。
- **pads约束**：pads为长度4的列表(top, bottom, left, right)，各分量必须为非负整数。仅在padding_mode为CALCULATED时生效；padding_mode为VALID时pads被忽略（置为0）；padding_mode为SAME时pads由框架自动计算。
- **data_format约束**：仅支持NCHW和NHWC。
- **padding_mode约束**：仅支持CALCULATED、VALID、SAME三种取值。
- **dtype约束**：x1和y的dtype必须相同（FLOAT16或FLOAT）。x2的dtype独立，支持INT4、INT8、FLOAT16、FLOAT（x2仅用于获取输入空间尺寸，不参与数值计算）。

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
    <td><a href="examples/arch35/test_geir_avg_pool_update.cpp">test_geir_avg_pool_update</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
