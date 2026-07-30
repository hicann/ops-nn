# AscendAntiQuantV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：根据输入的scale和offset对输入x进行反量化。

- 计算公式：
  - sqrt\_mode为true，offset为None时，计算公式为：

    $$
    y = cast\_to\_dst\_type((x) * scale * scale)
    $$

  - sqrt\_mode为true，offset不为None时，计算公式为：

    $$
    y = cast\_to\_dst\_type((x + offset) * scale * scale)
    $$

  - sqrt\_mode为false，offset为None时，计算公式为：

    $$
    y = cast\_to\_dst\_type((x) * scale)
    $$

  - sqrt\_mode为false，offset不为None时，计算公式为：

    $$
    y = cast\_to\_dst\_type((x + offset) * scale)
    $$

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
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示算子输入的Tensor，对应公式中的x；</li><li>不支持空Tensor；</li><li>当数据类型是INT4时，shape的尾轴为偶数。</li></ul></td>
      <td>INT4、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td><ul><li>表示反量化中的scale值。对应公式中的scale；</li><li>不支持空Tensor；</li><li>scale的维数必须与x相同，或者是1维；</li><li>如果x是1维，scale的形状必须是[1]或与x相同；</li><li>如果scale是1维，其大小必须是1、或x[-1]；</li><li>如果scale是多维，最多只能有一个非1的维度，且这个非1的维度只能是-1轴；</li></ul></td>
      <td>FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>可选输入</td>
      <td><ul><li>表示反量化中的offset值。对应公式中的offset；</li><li>不支持空Tensor；</li><li>数据类型和shape需要与scale保持一致。</li></ul></td>
      <td>和scale一致</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>表示输出的数据类型；</li><li>支持取值1、27，分别表示FLOAT16、BFLOAT16。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sqrt_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示scale参与计算的逻辑。对应公式中的sqrt_mode；</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示反量化的计算输出。对应公式中的y；</li><li>shape和输入x一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

输入x：不支持空Tensor；当数据类型是INT4时，shape的尾轴为偶数。
scale：不支持空Tensor。scale的维数必须与x相同，或者是1维；如果x是1维，scale的形状必须是[1]或与x相同；如果scale是1维，其大小必须是1、或x[-1]；如果scale是多维，最多只能有一个非1的维度，且这个非1的维度只能是-1轴。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_ascend_anti_quant](examples/test_aclnn_ascend_anti_quant.cpp) | 通过[aclnnAscendAntiQuant](docs/aclnnAscendAntiQuant.md)接口方式调用AscendAntiQuantV2算子。 |    |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| newnew | 个人开发者 | AscendAntiQuantV2 | 2026/06/22 | AscendAntiQuantV2算子适配开源仓 |
