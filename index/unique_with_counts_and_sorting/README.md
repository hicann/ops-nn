# UniqueWithCountsAndSorting

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：将输入张量x展平后进行全局去重，返回唯一元素y，可选择返回原输入到y的反向索引indices及每个唯一元素的出现次数counts。
- 计算流程：
  设展平后的输入为v，元素个数为N，去重后的元素个数为U。sorted为true时，y按升序排列；启用对应输出时，满足：

  $$y[indices[i]] = v[i], \quad 0 \le i < N$$

  $$counts[j] = \sum_{i=0}^{N-1} \mathbf{1}(indices[i] = j), \quad 0 \le j < U$$

  例如输入x=[3, 1, 3, 2]，sorted=true、return_inverse=true、return_counts=true时，得到y=[1, 2, 3]、indices=[2, 0, 2, 1]、counts=[1, 1, 2]。

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
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待去重的输入张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>唯一元素，数据类型与x相同，shape为[U]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输出</td>
      <td>return_inverse为true时，返回每个输入元素在y中的索引，从0开始，shape与x相同。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counts</td>
      <td>输出</td>
      <td>return_counts为true时，返回y中每个元素在x中的出现次数，shape为[U]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>return_inverse</td>
      <td>可选属性</td>
      <td>是否返回反向索引，默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_counts</td>
      <td>可选属性</td>
      <td>是否返回出现次数，默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sorted</td>
      <td>可选属性</td>
      <td>是否要求唯一元素按升序排列，默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out_idx</td>
      <td>可选属性</td>
      <td>索引及计数的数据类型，当前仅支持DT_INT64，默认值为DT_INT64。</td>
      <td>TYPE</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

输入张量x不能为空，且维度数必须大于0。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_unique](../unique/examples/test_aclnn_unique.cpp) | 通过[aclnnUnique](../unique/docs/aclnnUnique.md)或[aclnnUnique2](../unique/docs/aclnnUnique2.md)接口调用UniqueWithCountsAndSorting算子。 |
| 图模式调用 | [test_geir_unique_with_counts_and_sorting](./examples/test_geir_unique_with_counts_and_sorting.cpp) | 通过[算子IR](./op_graph/unique_with_counts_and_sorting_proto.h)构图方式调用UniqueWithCountsAndSorting算子。 |
