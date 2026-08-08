# MaxPool3D

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- 算子功能：对五维输入张量的深度、高度和宽度维执行三维最大池化。

- 计算公式：

  $$
  y(n,c,o_d,o_h,o_w)=\max_{(i_d,i_h,i_w)\in R(o_d,o_h,o_w)}x(n,c,i_d,i_h,i_w)
  $$

  其中，$R(o_d,o_h,o_w)$ 表示由池化窗口、步长、膨胀率和 padding 共同确定的有效输入区域。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 330px">
  <col style="width: 250px">
  <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>五维输入张量，逻辑布局由data_format指定。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND、NDC1HWC0</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>池化窗口大小，支持包含1、3或5个整数。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>池化窗口步长，支持包含1、3或5个整数。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性</td>
      <td>padding模式，支持VALID、SAME和CALCULATED。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>可选属性</td>
      <td>显式padding，顺序为front、back、top、bottom、left、right，默认全0。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>可选属性</td>
      <td>池化窗口膨胀率，支持包含1、3或5个整数，默认全1。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>可选属性</td>
      <td>CALCULATED模式下是否采用向上取整，0表示否，非0表示是，默认0。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td>输入输出的逻辑布局，支持NCDHW和NDHWC，默认NDHWC。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>最大池化结果，数据类型与x一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND、NDC1HWC0</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入和输出的逻辑维数必须为5。
- `ksize`、`strides` 和 `dilation` 的空间维取值必须大于0。
- 当 `ksize` 或 `strides` 包含5个元素时，N维和C维的值必须为1。
- `pads` 仅在 `padding="CALCULATED"` 时生效，必须包含6个非负整数。
- 算子不支持广播。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_max_pool3_d](./examples/test_aclnn_max_pool3_d.cpp) | 通过[aclnnMaxPool3D](./docs/aclnnMaxPool3D.md)接口以NDHWC布局调用MaxPool3D算子。 |
| aclnn调用 | [test_aclnn_max_pool3_d_ncdhw](./examples/test_aclnn_max_pool3_d_ncdhw.cpp) | 通过[aclnnMaxPool3D](./docs/aclnnMaxPool3D.md)接口以NCDHW布局调用MaxPool3D算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| :--- | :--- | :--- | :--- | :--- |
| lusonglier | 个人开发者 | MaxPool3D | 2026/07/18 | MaxPool3D算子适配开源仓 |
