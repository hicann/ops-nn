# AscendQuantV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对输入x进行量化操作，支持设置axis以指定scale和offset对应的轴，scale和offset的shape需要满足和axis指定x的轴相等或1。axis当前支持设置最后两个维度。
- 计算公式：
  - sqrt\_mode为false时，计算公式为：
    
    $$
    y = round((x * scale) + offset)
    $$

  - sqrt\_mode为true时，计算公式为：
    
    $$
    y = round((x * scale * scale) + offset)
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
      <td>需要执行量化的输入，对应公式中的`x`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>量化中的scale值，对应公式中的`scale`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>可选输入</td>
      <td>反量化中的offset值，对应公式中的`offset`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sqrt_mode</td>
      <td>可选属性</td>
      <td><ul><li>指定scale参与计算的逻辑，对应公式中的`sqrtMode`。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td><ul><li>指定cast到int8输出的转换方式。支持取值round，ceil，trunc，floor。</li><li>默认值为"round"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>指定输出的数据类型。</li><li>默认值为DT_INT8。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>指定scale和offset对应x的维度。</li><li>默认值为-1。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示AscendQuantV2的结果输出`y`，对应公式中的`y`。shape与输入`x`的shape一致。</td>
      <td>INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 推理系列产品</term>：
  - 数据类型：
    - 输入Tensor数据类型不支持BFLOAT16。
    - 输出Tensor数据类型仅支持INT8。
  - round_mode，支持取值round，ceil，trunc，floor。
  - dst_type：支持取值2，表示INT8。
  - axis：只支持指定x的最后一个维度（假设输入x维度是xDimNum，axis取值是-1或xDimNum-1）。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 输出Tensor数据类型仅支持INT8、INT4。
  - dst_type：支持取值2，29，分别表示INT8、INT4。
  - axis：支持指定x的最后两个维度（假设输入x维度是xDimNum，axis取值范围是[-2，-1]或[xDimNum-2，xDimNum-1]）。
- <term>Ascend 950PR/Ascend 950DT</term>：
   - round_mode：dst_type表示FLOAT8_E5M2或FLOAT8_E4M3FN时，只支持round。dst_type表示HIFLOAT8时，支持round和hybrid。dst_type表示其他类型时，支持round，ceil，trunc和floor。
   - axis：支持指定x的最后两个维度（假设输入x维度是xDimNum，axis取值范围是[-2，-1]或[xDimNum-2，xDimNum-1]）。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_ascend_quant](examples/test_aclnn_ascend_quant.cpp) | 通过[aclnnAscendQuant](docs/aclnnAscendQuant.md)接口方式调用AscendQuantV2算子。 |
| aclnn接口  | [test_aclnn_ascend_quant_v3](examples/test_aclnn_ascend_quant_v3.cpp) | 通过[aclnnAscendQuantV3](docs/aclnnAscendQuantV3.md)接口方式调用AscendQuantV2算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/ascend_quant_v2_proto.h)构图方式调用AscendQuantV2算子。         |