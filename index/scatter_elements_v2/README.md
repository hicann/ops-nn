# ScatterElementsV2

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|

## 功能说明

- 算子功能: 将tensor src中的值按指定的轴和方向和对应的位置关系逐个替换/累加/累乘至tensor self中。

- 示例：
  对于一个3D tensor， self会按照如下的规则进行更新：

  ```
  self[index[i][j][k]][j][k] += src[i][j][k] # 如果 dim == 0 && reduction == 1
  self[i][index[i][j][k]][k] *= src[i][j][k] # 如果 dim == 1 && reduction == 2
  self[i][j][index[i][j][k]] = src[i][j][k]  # 如果 dim == 2 && reduction == 0
  ```

  在计算时需要满足以下要求：
  - self、index和src的维度数量必须相同。
  - 对于每一个维度d，有index.size(d) <= src.size(d)的限制。
  - 对于每一个维度d，如果d != dim, 有index.size(d) <= self.size(d)的限制。
  - dim的值的大小必须在 [-self的维度数量, self的维度数量-1] 之间。
  - self的维度数应该小于等于8。
  - index中对应维度dim的值大小必须在[0, self.size(dim)-1]之间。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>self</td>
      <td>输入</td>
      <td>公式中的`self`，Device侧的aclTensor。</td>
      <td>UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>用来scatter的维度，数据类型为INT64。</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
      <tr>
      <td>index</td>
      <td>输入</td>
      <td>公式中的`index`，Device侧的aclTensor。</td>
      <td>INT32、INT64。</td>
      <td>ND</td>
    </tr>
      <td>src</td>
      <td>输入</td>
      <td>公式中的`src`，Device侧的aclTensor。</td>
      <td>UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>输入</td>
      <td>Host侧的整型，选择应用的reduction操作。</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出。</td>
      <td>FLOAT、FLOAT16、DOUBLE、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_scatter](./examples/test_aclnn_scatter.cpp) | 通过[aclnnScatter&aclnnInplaceScatter](./docs/aclnnScatter&aclnnInplaceScatter.md)接口方式调用ScatterElementsV2算子。 |
