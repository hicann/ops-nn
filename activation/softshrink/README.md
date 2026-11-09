# Softshrink

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

> 说明：
> - Ascend 950PR/950DT 由本仓 Ascend C 实现提供；Atlas A3/A2、Atlas推理/训练由 canndev TBE 实现承接；Atlas 200I/500 A2 无对应二进制配置，不支持。
> - Ascend 950PR/950DT、Atlas A3/A2 支持 FLOAT、FLOAT16、BFLOAT16；Atlas推理/训练仅支持 FLOAT、FLOAT16，不支持 BFLOAT16。

## 功能说明

- 算子功能：对输入张量应用Softshrink激活函数。

- 计算公式：

$$
Softshrink(x)=\begin{cases}
x-λ, if x > λ \\
x+λ, if x < -λ \\
0, otherwise \\
\end{cases}
$$

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
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input_x</td>
      <td>输入</td>
      <td>待进行Softshrink计算的输入张量，公式中的x。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>输出</td>
      <td>Softshrink计算的输出张量，shape与input_x一致。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>属性</td>
      <td>Softshrink的阈值参数，公式中的λ，默认值为0.5。取值为NaN或负数时接口返回非法参数错误；+Inf合法。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 数据格式仅支持ND。
- 输入input_x的dtype为fp16/fp32/bf16，输出output_y与input_x的dtype一致；Atlas推理/训练不支持BF16（仅FLOAT、FLOAT16）。
- 输入支持任意rank（0~8维）；0维标量输入会被视作shape为(1,)的1维tensor处理。
- lambd仅支持FLOAT标量，取值范围为[0, +∞)：允许0、正数和+Inf，负数或NaN会返回参数非法错误。
- 本仓Ascend C实现中，输入input_x为NaN时输出为0，不进行NaN透传；依赖NaN传播的调用方需注意该差异。
- 支持空Tensor（元素个数为0），直接返回同shape的空输出。
- 支持动态shape（含-1维度）：tiling在运行期按实际shape计算切分，无需重新编译。
- 默认确定性实现，相同输入始终产生相同输出。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_softshrink.cpp](./examples/test_aclnn_softshrink.cpp) | 通过[aclnnSoftshrink](./docs/aclnnSoftshrink.md)接口方式调用Softshrink算子。 |
| GE图模式 | [softshrink_proto.h](./op_graph/softshrink_proto.h) | 通过算子IR构图方式调用Softshrink算子。 |

> 样例的编译与运行请参考[aclnnSoftshrink调用示例](./docs/aclnnSoftshrink.md#调用示例)及CANN样例编译运行说明。
