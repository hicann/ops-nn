# DynamicBlockQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对输入张量，通过给定的row_block_size和col_block_size将输入划分成多个数据块，以数据块为基本粒度进行量化。在每个块中，先计算出当前块对应的量化参数scale，并根据scale对输入进行量化。输出最终的量化结果，以及每个块的量化参数scale。

- 计算公式：
  
  $$
  input\_max = block\_reduce\_max(abs(x))
  $$
  
  $$
  scale = min((FP8\_MAX/HiF8\_MAX / INT8\_MAX) / input\_max, 1/min_scale)
  $$

  $$
  y = cast\_to\_[FP8/HiF8/INT8](x / scale)
  $$
  
  其中block\_reduce\_max代表求每个block中的最大值。

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
      <td>表示算子输入的Tensor，对应公式中的`x`。shape支持2-3维。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min_scale</td>
      <td>可选属性</td>
      <td><ul><li>表示参与scale计算的最小scale值，对应公式中的`min_scale`。必须为正浮点数。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示量化的舍入模式，支持取值rint、round。</li><li>默认值为rint。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>指定输出y的数据类型。支持取值2、34、35、36，分别代表ACL_INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN。</li><li>默认值为35，对应的数据类型为FLOAT8_E5M2。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>row_block_size</td>
      <td>可选属性</td>
      <td><ul><li>表示行维度上每个块包含的元素数量。支持取值1，128。</li><li>默认值为1。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>col_block_size</td>
      <td>可选属性</td>
      <td><ul><li>表示列维度上每个块包含的元素数量。</li><li>默认值为128。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示量化后的张量，对应公式中的`y`。shape与输入`x`的shape保持一致，数据类型由参数`dst_type`的取值决定。</td>
      <td>INT8、HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输出</td>
      <td>表示量化使用的缩放张量，对应公式中的`scale`。shape为[ceil(x.rows/row_block_size), ceil(x.cols/col_block_size)]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 参数`round_mode`只支持rint。
    - 参数`dst_type`仅支持取值2，代表ACL_INT8。
    - 参数`row_block_size`仅支持取值1。
    - 参数`y`的数据类型仅支持INT8。

  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 参数`x`、`y`、`scale`的shape仅支持2维。
    - 参数`round_mode`的取值与参数`y`的数据类型存在对应关系：
      - 当输出`y`的数据类型是HIFLOAT8时，参数`round_mode`支持设置为round。
      - 当输出`y`的数据类型是FLOAT8_E4M3FN、FLOAT8_E5M2时，参数`round_mode`支持设置为rint。
    - 参数`dst_type`支持取值34、35、36，分别代表HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN。
    - 参数`y`的数据类型不支持INT8。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_dynamic_block_quant](examples/test_aclnn_dynamic_block_quant.cpp) | 通过[aclnnDynamicBlockQuant](docs/aclnnDynamicBlockQuant.md)接口方式调用DynamicBlockQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/dynamic_block_quant_proto.h)构图方式调用DynamicBlockQuant算子。         |

<<<<<<< HEAD
<!--图模式仅支持Ascend 950PR/Ascend 950DT-->
=======
<!--图模式仅支持910_95-->
>>>>>>> e2e609c4 (update)
<!-- [test_geir_dynamic_block_quant](examples/test_geir_dynamic_block_quant.cpp)-->