# GroupedDynamicMxQuant

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

- 接口功能：根据传入的分组索引的起始值，对传入的数据进行分组的float8的动态量化。

- 计算公式：
  - 将输入x在第0维上先按照groupIndex进行分组，每个group内按k = blocksize个数分组，一组k个数 {{x<sub>i</sub>}<sub>i=1</sub><sup>k</sup>} 计算出这组数对应的量化尺度mxscale_pre, {mxscale_pre, {P<sub>i</sub>}<sub>i=1</sub><sup>k</sup>}, 计算公式为下面公式(1)(2)。
  $$
  shared\_exp = floor(log_2(max_i(|V_i|))) - emax  \tag{1} 
  $$
  $$
  mxscale\_pre = 2^{shared\_exp}  \tag{2}
  $$
  - 这组数每一个除以mxscale，根据round_mode转换到对应的dst_type，得到量化结果y, 计算公式为下面公式(3)。
  $$
  P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize \tag{3}
  $$
  
  ​	量化后的P<sub>i</sub>按对应的x<sub>i</sub>的位置组成输出y，mxscale_pre按对应的groupIndex分组，分组内第一个维度pad为偶数，组成输出mxscale。
  
  - emax: 对应数据类型的最大正则数的指数位。
  
    |   DataType    | emax |
    | :-----------: | :--: |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |


## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>Device侧的aclTensor，计算公式中的输入x。shape仅支持2维。支持非连续的Tensor，支持空Tensor。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>groupIndex</td>
        <td>输入</td>
        <td>Device侧的aclTensor，量化分组的起始索引。shape仅支持1维。支持非连续的Tensor，支持空Tensor。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>roundMode</td>
        <td>输入</td>
        <td>host侧的string，公式中的round_mode，数据转换的模式，仅支持"rint"模式。</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstType</td>
        <td>输入</td>
        <td>host侧的int64_t，公式中的dst_type，指定数据转换后y的类型，输入范围为{35, 36}，分别对应输出y的数据类型为{35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}。</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blocksize</td>
        <td>输入</td>
        <td>host侧的int64_t，公式中的blocksize，指定每次量化的元素个数，仅支持32。</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>Device侧的aclTensor，公式中的输出y，输入x量化后的对应结果。需与dstType对应，shape仅支持2维，支持空Tensor，Shape和输入x一致。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>mxscale</td>
        <td>输出</td>
        <td>Device侧的aclTensor，公式中的mxscale_pre组成的输出mxscale，每个分组对应的量化尺度。需与dstType对应，shape仅支持3维，支持空Tensor，Shape和输入x一致。假设x的shape为 $[m,n]$，groupedIndex的shape为 $[g]$，则mxscale的shape为 $[(m/(blocksize * 2)+g), n, 2]$</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## 约束说明

无
## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_dynamic_mx_quant](./examples/test_aclnn_grouped_dynamic_mx_quant.cpp) | 通过[aclnnGroupedDynamicMxQuant](./docs/aclnnGroupedDynamicMxQuant.md)接口方式调用GroupedDynamicMxQuant算子。 |
