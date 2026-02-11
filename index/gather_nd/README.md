# GatherNd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对于维度为**r≥1**的输入张量`self`，和维度**q≥1**的输入张量`indices`，将数据切片收集到维度为 **(q-1) + (r - indices_shape[-1])** 的输出张量out中。indices是一个**q**维的整型张量，可视作一个**q-1**维的由**索引对**构成的特殊张量（每个**索引对**是一个长度为**indices_shape[-1]**的一维张量，每个**索引对**指向**self**中一个切片）。
- 计算流程：
  - 如果indices_shape[-1] > r，不合法场景。
  - 如果indices_shape[-1] = r，则输出张量out的维度为q-1，即out的shape为 [indices_shape[0:q-1]]，out中元素为self的索引对位置的元素。（见例1）
  - 如果indices_shape[-1] < r，则输出张量out的维度为 (q-1) + (r - indices_shape[-1])，设c=indices_shape[-1]，即out的shape为 [indices_shape[0:q-1],self_shape[c:r]] ，`out`由`self`的索引对位置的切片组成。（见例2、例3、例4）

  关于**r**、**q**、**indices_shape[-1]** 的一些限制条件如下：
  - 必须满足r≥1，q≥1。
  - indices_shape[-1]的值必须满足在1（包含）和r（包含）之间。
  - `indices`的每个元素，必须在[-s, s-1]范围内(s为self_shape各个轴上的值)，即-self_shape[i]≤indices[...,i]≤self_shape[i]-1。

- 示例：

  ```
  例1：
    self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
    indices: [[0, 0], [1, 1]]   # indices_shape=[2, 2], q=2, indices_shape[-1]=2
    out: [0, 3]                 # out_shape=[2]
  例2：
    self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
    indices: [[1], [0]]         # indices_shape=[2, 1], q=2, indices_shape[-1]=1
    out: [[2, 3], [0, 1]]       # out_shape=[2, 2]
  例3：
    self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
    indices: [[0, 1], [1, 0]]                  # indices_shape=[2, 2], q=2, indices_shape[-1]=2
    out: [[2, 3], [4, 5]]                      # out_shape=[2, 2]
  例4：
    self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
    indices: [[[0, 1]], [[1, 0]]]              # indices_shape=[2, 1, 2], q=3, indices_shape[-1]=2
    out: [[[2, 3]], [[4, 5]]]                  # out_shape=[2, 1, 2]
  ```

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1479px"><colgroup>
    <col style="width: 183px">
    <col style="width: 120px">
    <col style="width: 265px">
    <col style="width: 197px">
    <col style="width: 114px">
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
        <td>self</td>
        <td>输入</td>
        <td>输入aclTensor。</td>
        <td>INT64、INT32、INT8、UINT8、BOOL、FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT16、UINT16、UINT32、UINT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>输入</td>
        <td>输入aclTensor。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>negativeIndexSupport</td>
        <td>输入</td>
        <td>onnx模型是否存在负索引。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>输出aclTensor。</td>
        <td>与self一致</td>
        <td>ND</td>
      </tr>
    </tbody></table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型不支持DOUBLE、INT16、UINT16、UINT32、UINT64。
- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型不支持BFLOAT16、DOUBLE、INT16、UINT16、UINT32、UINT64。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_gather_nd](examples/arch35/test_aclnn_gather_nd.cpp) | 通过[aclnnGatherNd](./docs/aclnnGatherNd.md)接口方式调用GatherNd算子。 |
