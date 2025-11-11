# TransQuantParamV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：完成量化计算参数scale数据类型的转换，将FLOAT32的数据类型转换为硬件需要的UINT64类型。

- 计算公式：

  1. `out`为64位格式，初始为0。

  2. 若`round_mode`为1，`scale`按bit位round到高19位，`round_mode`为0不做处理。
     $$
     scale = Round(scale)
     $$

  3. `scale`按bit位取高19位截断，存储于`out`的bit位32位处，并将46位修改为1。
     
     $$
     out = out\ |\ (scale\ \&\ 0XFFFFE000)\ |\ (1\ll46)
     $$

  4. 根据`offset`取值进行后续计算：
     - 若`offset`不存在，不再进行后续计算。
     - 若`offset`存在：
       1. 将`offset`值处理为int，范围为[-256, 255]。
     
          $$
          offset = Max(Min(INT(Round(offset)),255),-256)
          $$

       2. 再将`offset`按bit位保留9位并存储于out的37到45位。
          $$
          out = (out\ \&\ 0x4000FFFFFFFF)\ |\ ((offset\ \&\ 0X1FF)\ll37)
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
      <td>scale</td>
      <td>输入</td>
      <td>量化参数，对应公式中的`scale`。shape是1维（t，），t = 1或n，以及2维（1，n）其中n与matmul计算中的右矩阵的shape n一致。用户需要保证scale数据中不存在NaN和inf。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>可选输入</td>
      <td>量化可选参数，对应公式中描述中的`offset`。shape是1维（t，），以及2维（1，n），t = 1或n，其中n与matmul计算中的右矩阵的shape n一致。用户需要保证offset数据中不存在NaN和inf。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td><ul><li>量化计算中FP32填充到FP19的round模式，支持以下取值：0（aclnn
      的V3兼容V2），1（提升计算精度）。其他值非法。对应公式中描述中的`round_mode`。</li><li>默认值为0。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化后的输出结果，对应公式中的`out`。当输入scale的shape为1维时，y的shape也为1维，该维度的shape大小为scale与offset(若不为nullptr)单维shape大小的最大值，当输入scale的shape为2维时，y的shape与输入scale的shape维度和大小完全一致。</td>
      <td>UINT64</td><!--opdef只有一个数据类型，aclnn有两个，多了一个INT64-->
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该接口对应的aclnn接口支持与matmul类算子（如[aclnnQuantMatmulV4](aclnnQuantMatmulV4.md)）配套使用。
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该接口对应的aclnn接口不支持与grouped matmul类算子（如aclnnGroupedMatmulV4）配套使用。
- 关于scale、offset、y的shape说明如下：
  - 当无offset时，y shape与scale shape一致。
    - 若y作为matmul类算子输入，shape支持1维(1,)、(n,)或2维(1, n)，其中n与matmul计算中右矩阵（对应参数x2）的shape n一致。
    - 若y作为grouped matmul类算子输入，仅在分组模式为m轴分组时使用（对应参数groupType为0），shape支持1维(g,)或2维(g, 1)、(g, n)，其中n与grouped matmul计算中右矩阵（对应参数x2）的shape n一致，g与grouped matmul计算中分组数（对应参数groupListOptional的shape大小）一致。
  - 当有offset时，仅作为matmul类算子输入：
    - offset，scale，y的shape支持1维(1,)、(n,)或2维(1, n)，其中n与matmul计算中右矩阵（对应参数x2）的shape n一致。
    - 当输入scale的shape为1维，y的shape也为1维，且shape大小为scale与offset单维shape大小的最大值。
    - 当输入scale的shape为2维，y的shape与输入scale的shape维度和大小完全一致。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_trans_quant_param_v2](examples/test_aclnn_trans_quant_param_v2.cpp) | 通过[aclnnTransQuantParamV2](docs/aclnnTransQuantParamV2.md)接口方式调用TransQuantParamV2算子。 |
| aclnn接口  | [test_aclnn_trans_quant_param_v3](examples/test_aclnn_trans_quant_param_v3.cpp) | 通过[aclnnTransQuantParamV3](docs/aclnnTransQuantParamV3.md)接口方式调用TransQuantParamV2算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/trans_quant_param_v2_proto.h)构图方式调用TransQuantParamV2算子。         |

<!--没有IR原型，只有op_ref,这一条补充啥啊？9.17之前反馈没有IR原型，但是9.17发现IR原型归档了，同步补充了一条-->
<!--[test_geir_trans_quant_param_v2](examples/test_geir_trans_quant_param_v2.cpp)-->