# AddRmsNormQuantV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能描述

- 算子功能：RmsNorm是大模型常用的标准化操作，相比LayerNorm，其去掉了减去均值的部分。AddRmsNormQuant算子将RmsNorm前的Add算子以及RmsNorm归一化的输出给到1个或2个Quantize算子融合起来，减少搬入搬出操作。AddRmsNormQuantV2算子相较于AddRmsNormQuant在RmsNorm计算过程中增加了偏置项bias参数，即计算公式中的`bias`。

- 计算公式：

  $$
  x_i={x1}_i+{x2}_i
  $$

  $$
  y_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i + bias, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  resOut_i=\frac{1}{\operatorname{Rms}(x_i)} * x_i * gamma_i
  $$

  - div_mode为True时：

    $$
    y1=round((y/scales1)+zero\_points1)
    $$

    $$
    y2=round((y/scales2)+zero\_points2)
    $$

  - div_mode为False时：

    $$
    y1=round((y*scales1)+zero\_points1)
    $$

    $$
    y2=round((y*scales2)+zero\_points2)
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
      <td>x1</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量，对应公式中的`x1`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量，对应公式中的`x2`。shape与`x1`保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量。对应公式中的`gamma`。shape需要与`x1`需要Norm的维度保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>输入</td>
      <td>表示量化过程中得到y1的scales张量，对应公式中的`scales1`。当参数`div_mode`的值为True时，该参数的值不能为0。</td>
      <td>FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y2的scales张量，对应公式中的`scales2`。当参数`div_mode`的值为True时，该参数的值不能为0。shape与`scales1`保持一致。</td>
      <td>FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y1的offset张量，对应公式中的`zero_points1`。shape需要与`gamma`保持一致。</td>
      <td>BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y2的offset张量，对应公式中的`zero_points2`。shape需要与`gamma`保持一致。</td>
      <td>BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>表示标准化过程中的偏置项，公式中的`bias`。shape需要与`gamma`保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>表示需要进行量化的elewise轴，其他的轴做broadcast，指定的轴不能超过输入`x1`的维度数。当前仅支持-1，传其他值均不生效。</li><li>默认值为-1。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>用于防止除0错误，对应公式中的`epsilon`。</li><li>默认值为1e-6。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>可选属性</td>
      <td><ul><li>公式中决定量化公式是否使用除法的参数，对应公式中的`div_mode`。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y1`。shape需要与输入`x1`/`x2`一致。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y2`。shape需要与输入`x1`/`x2`一致。当`scales2`为空时，该输出的值无效。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>表示x1和x2的和，对应公式中的`x`。shape与输入`x1`/`x2`一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>resOut</td>
      <td>输出</td>
      <td>表示进行RmsNorm之后的结果，对应公式中的`resOut`。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 可选输出`x`和`resOut`，必须且只能选择其一进行输出。
- 当需要输出`y2`时，此时要求`gamma`与`scale`的shape保持一致，且需要与`x1`需要Norm的维度保持一致，可选输出只能输出`x`。
- 当需要输出`x`时，且参数`scales1`和`zero_points1`的shape为[1]，且`gamma`的shape为1维且与x1的最后一维相等或者`gamma`的shape为2维且第一维为1、第二维为`x1`的最后一维时，此时`scales2`和`zero_points2`不生效。
- 当需要输出`resOut`时，参数`scales1`和`zero_points1`的shape为[1]，且`gamma`的shape为1维且与x1的最后一维相等或者`gamma`的shape为2维且第一维为1、第二维为`x1`的最后一维，且`bias`和`x`必须传空指针，此时`scales2`和`zero_points2`不生效。

- **边界值场景说明**

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入是inf时，输出为inf。当输入是NaN时，输出为NaN。

- **维度的边界说明**

  参数`x1`、`x2`、`gamma`、`bias`、`scales1`、`scales2`、`zero_points1`、`zero_points2`、`y1`、`y2`、`x`、`resOut`的shape中每一维大小都不大于INT32的最大值2147483647。

- **数据格式说明**

    所有输入输出Tensor的数据格式推荐使用ND格式，其他数据格式会由框架默认转换成ND格式进行处理。

- **各产品型号支持数据类型说明**
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

     | x1数据类型 | x2数据类型 | gamma数据类型 | scales1数据类型 | scales2数据类型 | zero_points1数据类型 | zero_points2数据类型 | bias数据类型 | y1数据类型 | y2数据类型 | x数据类型 | resOut数据类型 |
     | - | - | - | - | - | - | - | - | - | - | - | - |
     | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | FLOAT16 | INT8 | INT8 | FLOAT16 | FLOAT16 |
     | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | INT8 | INT8 | BFLOAT16 | BFLOAT16 |

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_add_rms_norm_quant_v2](examples/test_aclnn_add_rms_norm_quant_v2.cpp) | 通过[aclnnAddRmsNormQuantV2](docs/aclnnAddRmsNormQuantV2.md)接口方式调用AddRmsNormQuantV2算子。 |

<!--
| 图模式 | -  | 通过[算子IR](op_graph/add_rms_norm_quant_proto.h)构图方式调用AddRmsNormQuantV2算子。         |
-->

<!--[test_geir_add_rms_norm_quant](examples/test_geir_add_rms_norm_quant.cpp)-->