# AscendAntiQuantV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

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
      <td>INT4、INT8、HIFLOAT8、FLOAT8_E4M3、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td><ul><li>表示反量化中的scale值。对应公式中的scale；</li><li>不支持空Tensor；</li><li>scale的维数必须与x相同，或者是1维；</li><li>如果x是1维，scale的形状必须是[1]或与x相同；</li><li>如果scale是1维，其大小必须是1、x[-1]或x[-2]；</li><li>如果scale是多维，最多只能有一个非1的维度，且这个非1的维度只能是-1或-2轴；</li><li>当输入x的类型为HIFLOAT8、FLOAT8_E4M3、FLOAT8_E5M2时，scale只支持FLOAT32。</li></ul></td>
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
      <td><ul><li>表示scale参与计算的逻辑。对应公式中的sqrt_mode；</li><li>当x的数据类型为HIFLOAT8、FLOAT8_E4M3、FLOAT8_E5M2时，sqrt_mode为false。</li></ul></td>
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

- <term>Atlas 推理系列产品</term>：
  - 数据类型：
    - 入参`x`仅支持INT8。
    - 入参`scale`、`offset`和出参`y`不支持BFLOAT16。
  - 入参`dstType`仅支持取值1，表示FLOAT16。
  - 入参`scale`、`offset`仅支持一维，且只能等于x尾轴大小或1，当x为int32类型时，必须为x尾轴大小的8倍。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品/Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 入参`x`仅支持INT4、INT8。
  - 入参`scale`、`offset`仅支持一维，且只能等于x尾轴大小或1，当x为int32类型时，必须为x尾轴大小的8倍。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_ascend_anti_quant](examples/test_aclnn_ascend_anti_quant.cpp) | 通过[aclnnAscendAntiQuant](docs/aclnnAscendAntiQuant.md)接口方式调用AscendAntiQuantV2算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/ascend_anti_quant_v2_proto.h)构图方式调用AscendAntiQuantV2算子。         |
