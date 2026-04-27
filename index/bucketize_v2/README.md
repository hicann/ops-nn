# BucketizeV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    x     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    x     |
| <term>Atlas 推理系列产品 </term>                              |    x     |
| <term>Atlas 训练系列产品</term>                               |    x     |

## 功能说明

- 算子功能：用于将张量中的值按照给定的边界进行离散化, 根据给定的边界将输入张量中的每个值映射到对应的分箱区间，并返回该值所属区间索引。
- 计算流程：
对于输入值 $x$ 和边界数组 $boundaries = [b₀, b₁, ..., bₙ₋₁]$，输出索引 $yi$ 满足：

**当 right=false（默认，左闭右开区间）：**
如果 $x < b₀$，则 $yi = 0$
如果 $bᵢ₋₁ ≤ x < bᵢ$，则 $yi = i$
如果 $x ≥ bₙ₋₁$，则 $yi = n$

**当 right=true（左开右闭区间）：**
如果 $x ≤ b₀$，则 $yi = 0$
如果 $bᵢ₋₁ < x ≤ bᵢ$，则 $yi = i$
如果 $x > bₙ₋₁$，则 $yi = n$

**示例说明：**
假设 $boundaries = [1, 3, 5]$，对于输入值 $x=3$：
- 当 right=false 时，$x$ 满足 $1 ≤ 3 < 5$，落入区间 $[1, 5)$，输出索引为 1
- 当 right=true 时，$x$ 满足 $1 < 3 ≤ 3$，落入区间 $(1, 3]$，输出索引为 1

对于边界值 $x=1$：
- 当 right=false 时，$x$ 满足 $1 ≤ 1 < 3$，落入区间 $[1, 3)$，输出索引为 1
- 当 right=true 时，$x$ 满足 $x ≤ 1$，输出索引为 0

## 参数说明

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 215px">
  <col style="width: 163px">
  <col style="width: 287px">
  <col style="width: 439px">
  <col style="width: 135px">
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
        <td>输入的张量，公式中的x。</td>
        <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>boundaries</td>
        <td>输入</td>
        <td>输入的张量，表示给定的边界，公式中的boundaries。</td>
        <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>输入的结果，公式中的y。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>right</td>
        <td>可选属性</td>
        <td><ul><li>用于指定区间是否包含右边界。</li><li>默认值为false。</ul></td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>out_int32</td>
        <td>可选属性</td>
        <td><ul><li>用于指定输出的数据类型，true表示输出y的dtype为int32，false表示输出y的dtype为int64。</li><li>默认值为false。</ul></td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## 约束说明

boundaries中的数必须是升序，且不重复。