# UnsortedSegmentSum

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term> Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能: 对一个张量分段求和
- 公式说明：
    $$
    y[i] = {\sum}_{j...:segment_ids[j...]==i} x[j...]
    $$
    其中，对满足 segment_ids[j...] == i 的所有位置 j... 求和。若某个段 i 没有对应的元素，则 y[i] = 0。
   
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
      <td>x</td>
      <td>输入</td>
      <td>输入数据</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT32、INT64、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>segment_ids</td>
      <td>输入</td>
      <td>分段索引</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_segments</td>
      <td>输入</td>
      <td>分段个数</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出值信息</td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT32、INT64、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_unsorted_segment_sum](examples/test_geir_unsorted_segment_sum.cpp) | 通过[算子IR](op_graph/unsorted_segment_sum_proto.h)构图方式调用UnsortedSegmentSum算子。 |
