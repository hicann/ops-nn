# FlatQuant

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

- 算子功能：该融合算子为输入矩阵x一次进行两次小矩阵乘法，即右乘输入矩阵kroneckerP2，左乘输入矩阵kroneckerP1，然后针对矩阵乘的结果进行量化处理。目前支持pertoken和pergroup量化方式，分别对应int4和float4_e2m1量化输出类型。

- 矩阵乘计算公式：

  1. 输入x右乘kroneckerP2：

     $$
     x' = x @ kroneckerP2
     $$

  2. kroneckerP1左乘x'：

     $$
     x'' = kroneckerP1@x'
     $$

- 量化计算方式：

  pertoken量化方式：

  1. 沿着x''的0维计算最大绝对值并除以(7 / clipRatio)以计算需量化为INT4格式的量化因子：

     $$
     quantScale = [max(abs(x''[0,:,:])),max(abs(x''[1,:,:])),...,max(abs(x''[M-1,:,:]))]/(7 / clipRatio)
     $$

  2. 计算输出的out：

     $$
     out = x'' / quantScale
     $$

  pergroup量化方式

  1. 矩阵乘后x''的shape为[M,N1,N2],在计算pergroup量化方式其中的mx_quantize时，需reshape为[M,N1*N2],记为x2

  2. 在x2第二维上按照groupsize进行分组，包含元素e0,e1...e31。计算出emax

     $$
     emax = max(e0,e1....e31)
     $$

  3. 计算出reduceMaxValue和sharedExp

     $$
     reduceMaxValue = log2(reduceMax(x2),groupSize=32)
     $$

     $$
     sharedExp[M,N1*N2/32] = reduceMaxValue -emax
     $$

  4. 计算quantScale

     $$
     quantScale[M,N1*N2/32] = 2 ^ {sharedExp[M,N1*N2/32]}
     $$

  5. 每groupsize共享一个quantScale，计算out

     $$
     out = x2 / quantScale
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
      <td>输入的原始数据，对应公式中的`x`。shape为[M, N1, N2]，其中，M不超过262144，N1和N2不超过256。`out`的数据类型为INT4，N2必须是偶数。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kronecker_p1</td>
      <td>输入</td>
      <td>输入的计算矩阵1，对应公式中的`kroneckerP1`。shape为[N1, N1]，N1与`x`中N1维一致，数据类型与入参`x`的数据类型一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kronecker_p2</td>
      <td>输入</td>
      <td>输入的计算矩阵2，对应公式中的`kroneckerP2`。shape为[N2, N2]，N2与`x`中N2维一致，数据类型与入参`x`的数据类型一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>可选输入</td>
      <td>代表输入`x`的量化分组大小分布。当group_list_type为0或1时，shape为[G]，当group_list_type为2时，shape为[G, 2]，G表示分组数，G需要小于等于1024。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clip_ratio</td>
      <td>可选属性</td>
      <td><ul><li>用于控制量化的裁剪比例，输入数据范围为(0, 1]。</li><li>默认值为1。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_dtype</td>
      <td>可选属性</td>
      <td><ul><li>用于控制量化方式，当值为DT_FLOAT4_E2M1时表示为pergroup量化方式，当值为DT_INT32时表示pertoken量化方式。</li><li>默认值为DT_INT32。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>可选属性</td>
      <td><ul><li>表示maxType的取值，对应公式中的Amax。支持取值0.0和6.0-12.0。</li><li>取值为0.0代表不使用该参数。</li><li>取值为6.0-12.0代表目标数据类型的最大值。</li><li>仅支持在FLOAT4_E2M1数据类型时设置该值。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>可选属性</td>
      <td><ul><li>代表groupList输入的分组方式。当group_list不为nullptr时支持取值0，1，2。</li><li>默认值为0。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量，对应公式中的`out`。数据类型为INT4时，shape与入参`x`一致。数据类型为INT32时，shape为[M,N1,N2/8]。数据类型为FLOAT4_E2M1时，shape为[M,N1*N2]。</td>
      <td>INT4、INT32、FLOAT4_E2M1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_scale</td>
      <td>输出</td>
      <td>输出的量化因子，对应公式中的`quantScale`。当输出类型为FLOAT时，shape为[M]，M与`x`中M维一致。当输出类型为FLOAT8_E8M0时，shape为[M,ceilDiv(N1*N2,64),2]。</td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 参数group_list、group_list_type需要满足如下约束：
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - group_list仅支持nullptr。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - group_list不为nullptr时：
      - group_list_type必须从[0, 1, 2]中取值。
      - 当group_list_type为0或1时，shape为[G]，当group_list_type为2时，shape为[G, 2]，G表示分组数，G需要小于等于1024。
      - group_list的数值需要满足以下条件，否则无法保证输出是否符合预期：
        - 当group_list_type为0时，group_list必须为非负单调非递减数列，表示分组后每组大小的cumsum结果（累计和），最后一个值应小于等于x中tensor的第一维。
        - 当group_list_type为1时，group_list必须为非负数列，表示分组后每组大小，数值的总和应小于等于x中tensor的第一维。
        - 当group_list_type为2时，group_list必须为非负数列，数据排布为[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]，其中groupSize为分组后每组大小，第二列数值的总和应小于等于x中tensor的第一维。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_flat_quant](examples/test_aclnn_flat_quant.cpp) | 通过[aclnnFlatQuant](docs/aclnnFlatQuant.md)接口方式调用FlatQuant算子。 |
| aclnn接口  | [test_aclnn_flat_quant_v2](examples/test_aclnn_flat_quant_v2.cpp) | 通过[aclnnFlatQuantV2](docs/aclnnFlatQuantV2.md)接口方式调用FlatQuant算子。 |
| aclnn接口  | [test_aclnn_flat_quant_v3](examples/test_aclnn_flat_quant_v3.cpp) | 通过[aclnnFlatQuantV3](docs/aclnnFlatQuantV3.md)接口方式调用FlatQuant算子。 |
| PyTorch API | - | 通过[flat_quant](./docs/torchapi_flat_quant.md)接口调用FlatQuant算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/flat_quant_proto.h)构图方式调用FlatQuant算子。         |
