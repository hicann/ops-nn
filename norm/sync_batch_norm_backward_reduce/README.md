# SyncBatchNormBackwardReduce

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：SyncBatchNormBackwardReduce用于反向传播过程中计算BatchNorm操作的所需的权重梯度gradWeight和中间量sumDyXmu。

- 计算公式：

$$
sumDyXmu = {sumDyDxPad} - {sumDy} * {mean}
$$

$$
gradWeight = ({sumDyDxPad} - {sumDy} * {mean}) * invertStd
$$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
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
      <td>sum_dy</td>
      <td>输入</td>
      <td>表示正向输出梯度的累加和，对应公式中的`sumDy`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_dy_dx_pad</td>
      <td>输入</td>
      <td>对应公式中的`sumDyDxPad`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示输入数据均值，对应公式中的`mean`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>invert_std</td>
      <td>输入</td>
      <td>表示输入数据标准差倒数，对应公式中的`invertStd`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_dy_xmu</td>
      <td>输出</td>
      <td>表示正向输出梯度与输入中心化后数据乘积之和，对应公式中的`sumDyXmu`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示缩放参数的梯度，对应公式中的`gradWeight`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_batch_norm_reduce_backward](../sync_batch_norm_backward_reduce/examples/test_aclnn_batch_norm_reduce_backward.cpp) | 通过[aclnnBatchNormReduceBackward](../sync_batch_norm_backward_reduce/docs/aclnnBatchNormReduceBackward.md)接口方式调用SyncBatchNormBackwardReduce算子。 |
| 图模式 | [test_geir_sync_batch_norm_backward_reduce](../sync_batch_norm_backward_reduce/examples/test_geir_sync_batch_norm_backward_reduce.cpp) | 通过[算子IR](op_graph/sync_batch_norm_backward_reduce_proto.h)构图方式调用SyncBatchNormBackwardReduce算子。         |
