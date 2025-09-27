# GroupQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对输入x进行分组量化操作。
- 计算公式：

  $$
  y = round((x * scale) + offsetOptional)
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
      <td>需要做量化的输入，对应公式中的`x`。shape支持2维。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>量化中的scale值，对应公式中的`scale`。shape支持2维，且第2维与x shape的第2维相等。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>输入</td>
      <td>量化中的groupIndex值。shape支持1维，且维度与scale shape的第1维相等。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>可选输入</td>
      <td>量化中的offset值，对应公式中的`offsetOptional`。并且数据类型与`scale`一致。`offsetOptional`为1个数。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>指定输出的数据类型，支持取值2，29，分别表示INT8，INT4。<!--INT32(3)acln
      n中有--></li><li>默认值为DT_INT8。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示LayerNorm的结果输出`y`，对应公式中的y。shape与`x`的shape一致。</td>
      <td>INT4、INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入`scale`与输入`offset`的数据类型一致。
- 如果属性`dstType`为29(INT4)，那么输入`y`的shape的最后一维需要能被2整除。
- 输入`groupIndex`必须是非递减序列，最大值必须与输入`x`的shape的第1维大小相等。
- 输入`offsetOptional`的shape当前仅支持[1, ]或[ , ]。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_group_quant](examples/test_aclnn_group_quant.cpp) | 通过[aclnnGroupQuant](docs/aclnnGroupQuant.md)接口方式调用GroupQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/group_quant_proto.h)构图方式调用GroupQuant算子。         |

<!--[test_geir_group_quant](examples/test_geir_group_quant.cpp)-->