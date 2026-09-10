# AddRmsNormQuantV2

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：RmsNorm是大模型常用的标准化操作，相比LayerNorm，其去掉了减去均值的部分。AddRmsNormQuant算子将RmsNorm前的Add算子以及RmsNorm归一化的输出给到1个或2个Quantize算子融合起来，减少搬入搬出操作。AddRmsNormQuantV2算子相较于AddRmsNormQuant在RmsNorm计算过程中增加了偏置项bias参数，即计算公式中的`bias`，同时新增可选输出`resOut`，表示量化前的RmsNorm归一化结果（不含bias）。

- 计算公式：

  $$
  x_i={x1}_i+{x2}_i
  $$

  $$
  y_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i + bias, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  resOut_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i
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
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量，对应公式中的`x2`。shape与`x1`保持一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量。对应公式中的`gamma`。shape需要与`x1`需要Norm的维度保持一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>输入</td>
      <td>表示量化过程中得到y1的scales张量，对应公式中的`scales1`。当参数`div_mode`的值为True时，该参数的值不能为0。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>可选输入</td>
      <td>表示第二路量化的scales张量，对应公式中的`scales2`。可不传。当参数`div_mode`的值为True时，该参数的值不能为0。shape与`scales1`保持一致。缺省行为参见<a href="#产品差异说明">产品差异说明</a>。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>可选输入</td>
      <td>表示第一路量化的offset张量，对应公式中的`zero_points1`。可不传，不传时不叠加offset。shape需要与`gamma`保持一致。</td>
      <td>INT32、FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>可选输入</td>
      <td>表示第二路量化的offset张量，对应公式中的`zero_points2`。可不传，不传时不叠加offset。shape需要与`gamma`保持一致。与`scales2`的合法组合参见<a href="#产品差异说明">产品差异说明</a>。</td>
      <td>INT32、FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>表示标准化过程中的偏置项，对应公式中的`bias`。可不传，不传时不叠加偏置。shape需要与`gamma`保持一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y1`。shape需要与输入`x1`/`x2`一致。</td>
      <td>INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示第二路量化输出Tensor，对应公式中的`y2`。shape需要与输入`x1`/`x2`一致。该输出的有效条件参见<a href="#产品差异说明">产品差异说明</a>。</td>
      <td>INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>可选输出</td>
      <td>表示`x1`和`x2`之和，对应公式中的`x`。shape与输入`x1`/`x2`一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>resOut</td>
      <td>可选输出</td>
      <td>表示不含`bias`的RmsNorm结果，对应公式中的`resOut`。当`output_res`为True时输出，shape与输入`x1`/`x2`一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>表示需要进行量化的element-wise轴，其他轴进行广播。当前仅支持-1，传入其他值不生效。</li><li>默认值为-1。</li></ul></td>
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
      <td><ul><li>表示量化计算使用除法还是乘法，对应公式中的`div_mode`。</li><li>默认值为True。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>可选属性</td>
      <td><ul><li>表示`y1`和`y2`的数据类型。</li><li>默认值为DT_INT8。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_res</td>
      <td>可选属性</td>
      <td><ul><li>表示是否输出不含`bias`的RmsNorm结果`resOut`。</li><li>默认值为False。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

### 产品差异说明

参数表中的数据类型和数据格式为所有支持产品能力的并集。下表中的“x类参数”包括`x1`、`x2`、`gamma`、`bias`、`x`和`resOut`；“scale参数”包括`scales1`和非空的`scales2`；“zero point参数”包括非空的`zero_points1`和`zero_points2`。同一合法组合内，同类参数的数据类型保持一致。

#### 数据类型与属性差异

| 产品 | x类参数/scale参数/zero point参数合法组合 | `y1`、`y2`数据类型（`dst_type`） | `div_mode` |
| --- | --- | --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | FLOAT16/FLOAT32/INT32、BFLOAT16/FLOAT32/INT32、FLOAT32/FLOAT32/FLOAT32、FLOAT16/FLOAT16/FLOAT16、BFLOAT16/BFLOAT16/BFLOAT16、FLOAT16/FLOAT32/FLOAT32、BFLOAT16/FLOAT32/FLOAT32 | INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN；`y1`与`y2`的数据类型保持一致 | True、False |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | FLOAT16/FLOAT32/INT32、BFLOAT16/BFLOAT16/BFLOAT16、BFLOAT16/FLOAT32/INT32 | INT8 | 原生V2路径仅支持True；aclnn API满足V1回退条件时支持False |
| <term>Atlas 推理系列产品</term> | FLOAT16/FLOAT32/INT32 | INT8 | 原生V2路径仅支持True；aclnn API满足V1回退条件时支持False |

对于Atlas A3训练系列产品/Atlas A3推理系列产品、Atlas A2训练系列产品/Atlas A2推理系列产品和Atlas推理系列产品，通过aclnn API调用且`div_mode`为False时，接口不进入原生V2路径。当输出`x`、不输出`resOut`且其他参数满足V1回退路径约束时，接口回退到V1路径并按乘法模式执行。GE图模式不适用该回退机制，`div_mode`仅支持True。

#### shape与参数组合差异

| 产品 | 静态shape能力 | 动态shape能力 | 第二路量化 | 其他用户可观察限制 |
| --- | --- | --- | --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | 输入、输出均支持ND格式 | 输入、输出均支持ND格式 | `scales2`或`zero_points2`任一非空时，`y2`为有效输出。仅`zero_points2`非空时，`scales2`按1处理，$y2=round(y+zero\_points2)$ | - |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | 输入、输出均支持ND格式 | 输入、输出均支持ND格式 | `y2`仅在`scales2`非空时有效；`zero_points2`非空时，`scales2`必须同时非空 | 输入为inf时，输出为inf；输入为NaN时，输出为NaN |
| <term>Atlas 推理系列产品</term> | 输入、输出均支持ND格式 | 输入、输出均支持ND格式 | `y2`仅在`scales2`非空时有效；`zero_points2`非空时，`scales2`必须同时非空 | 输入不支持inf和NaN；`gamma`包含的归一化元素个数不能小于32 |

## 约束说明

- 可选输出`x`和`resOut`，必须且只能选择其一进行输出。
- 当需要输出`y2`时，此时要求`gamma`与`scale1`的shape保持一致，且需要与`x1`需要Norm的维度保持一致，可选输出只能输出`x`。
- 当需要输出`x`时，且参数`scales1`和`zero_points1`的shape为[1]，且`gamma`的shape为1维且与x1的最后一维相等或者`gamma`的shape为2维且第一维为1、第二维为`x1`的最后一维时，此时`scales2`和`zero_points2`不生效。
- 当需要输出`resOut`时，参数`scales1`和`zero_points1`的shape为[1]，且`gamma`的shape为1维且与x1的最后一维相等或者`gamma`的shape为2维且第一维为1、第二维为`x1`的最后一维，且`bias`和`x`必须传空指针，此时`scales2`和`zero_points2`不生效。

- **维度的边界说明**

  参数`x1`、`x2`、`gamma`、`bias`、`scales1`、`scales2`、`zero_points1`、`zero_points2`、`y1`、`y2`、`x`、`resOut`的shape中每一维大小都不大于INT32的最大值2147483647。

- **数据格式说明**

    所有输入输出Tensor的数据格式推荐使用ND格式，其他数据格式会由框架默认转换成ND格式进行处理。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| aclnn API | [test_aclnn_add_rms_norm_quant_v2](examples/test_aclnn_add_rms_norm_quant_v2.cpp) | 通过[aclnnAddRmsNormQuantV2接口文档](docs/aclnnAddRmsNormQuantV2.md)中的两段式接口调用AddRmsNormQuantV2算子。 |
| GE图模式 | - | 通过[AddRmsNormQuantV2算子IR](op_graph/add_rms_norm_quant_v2_proto.h)构图方式调用AddRmsNormQuantV2算子。 |
