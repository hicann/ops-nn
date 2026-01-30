# GroupedDynamicBlockQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：根据传入的分组索引的起始值（group_list）对各个group以基本块的粒度进行量化，量化为（FP8/HiFP8），并输出量化参数scale（FP32）。

- 计算公式：

  $$
    input\_max = block\_reduce\_max(abs(input))
  $$

  $$
    scale = min(input\_max/FP8\_MAX(HiF8\_MAX), 1/min\_scale)
  $$

  $$
    y = cast\_to\_[HiF8/FP8](input/scale)
  $$


## 参数说明

<table style="undefined;table-layout: fixed; width: 1401px"><colgroup>
  <col style="width: 149px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 200px">
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
      <td>表示算子输入的Tensor，形如[M, N]或[B, M, N]，对应公式中的input。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>输入</td>
      <td>表示在M轴上每个group的偏移（cumsum模式）。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min_scale</td>
      <td>输入</td>
      <td>表示参与scale计算的最小scale值，对应公式中的min_scale。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td>表示最后由高bit数据cast到目标数据类型的近似模式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>输入</td>
      <td>表示数据转换后y的数据类型。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>row_block_size</td>
      <td>输入</td>
      <td>表示指定M轴上的量化粒度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>col_block_size</td>
      <td>输入</td>
      <td>表示指定N轴上的量化粒度。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>输入</td>
      <td>表示group_list的功能类型。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示量化后的输出Tensor，对应公式中的y。</td>
      <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输出</td>
      <td>表示每个分组对应的量化尺度，对应公式中的scale。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_grouped_dynamic_block_quant](examples/arch35/test_aclnn_grouped_dynamic_block_quant.cpp) | 通过[aclnnGroupedDynamicBlockQuant](docs/aclnnGroupedDynamicBlockQuant.md)接口方式调用GroupedDynamicBlockQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/grouped_dynamic_block_quant_proto.h)构图方式调用GroupedDynamicBlockQuant算子。         |