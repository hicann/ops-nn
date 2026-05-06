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

    **当 right=false（默认，左开右闭区间）：**
    如果 $x \le b_0$，则 $yi = 0$
    如果 $b_{i-1} < x \le b_i$，则 $yi = i$
    如果 $x > b_{n-1}$，则 $yi = n$

    **当 right=true（左闭右开区间）：**
    如果 $x < b_0$，则 $yi = 0$
    如果 $b_{i-1} \le x < b_i$，则 $yi = i$
    如果 $x \ge b_{n-1}$，则 $yi = n$

    注意：本算子语义与 `numpy.digitize` 相反（numpy 的 `right=False` 是左闭右开）。

    **示例说明：**
    假设 $boundaries = [1, 3, 5]$，对于输入值 $x=3$：
    - 当 right=false 时，$x$ 满足 $1 < 3 \le 3$，落入区间 $(1, 3]$，输出索引为 1
    - 当 right=true 时，$x$ 满足 $3 \le 3 < 5$，落入区间 $[3, 5)$，输出索引为 2

    对于边界值 $x=1$：
    - 当 right=false 时，$x$ 满足 $x \le 1$，落入 $(-\infty, 1]$，输出索引为 0
    - 当 right=true 时，$x$ 满足 $1 \le 1 < 3$，落入区间 $[1, 3)$，输出索引为 1

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

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_bucketize](examples/arch35/test_aclnn_bucketize.cpp) | 通过[aclnnBucketize](./docs/aclnnBucketize.md)接口方式调用BucketizeV2算子。 |
| 图模式调用 | [test_geir_bucketize_v2](./examples/arch35/test_geir_bucketize_v2.cpp) | 通过[算子IR](./op_graph/bucketize_v2_proto.h)构图方式调用BucketizeV2算子。 |
