# SyncBatchNormBackwardElemt

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

- 算子功能：SyncBatchNormBackwardElemt算子用于计算输入张量的元素级梯度，以便在反向传播过程中更新模型参数。


- 计算公式：

$$
gradInput = ({gradOut} - {meanDy}) - ((input - mean) * (invstd^{2} *   {meanDyXmu})) * invstd * weight
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
      <td>grad_output</td>
      <td>输入</td>
      <td>表示正向输出的微分，对应公式中的`gradOut`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>save_input</td>
      <td>输入</td>
      <td>表示进行BatchNorm计算的输入，对应公式中的`input`。</td>
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
      <td>invstd</td>
      <td>输入</td>
      <td>表示输入数据标准差倒数，对应公式中的`invstd`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示权重Tensor，对应公式中的`weight`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean_dy</td>
      <td>输入</td>
      <td>表示输出梯度的样本均值和的平均值，对应公式中的`sumDy`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean_dy_xmu</td>
      <td>输入</td>
      <td>表示样本均值和与输入梯度乘积的平均值，对应公式中的`sumDyXmu`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>表示输入Tensor的梯度，对应公式中的`gradInput`。</td>
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
| aclnn接口  | [test_aclnn_batch_norm_elemt_backward](../sync_batch_norm_backward_elemt/examples/test_aclnn_batch_norm_elemt_backward.cpp) | 通过[aclnnBatchNormElemtBackward](../sync_batch_norm_backward_elemt/docs/aclnnBatchNormElemtBackward.md)接口方式调用SyncBatchNormBackwardElemt算子。 |
| 图模式 | [test_geir_sync_batch_norm_backward_elemt](../sync_batch_norm_backward_elemt/examples/test_geir_sync_batch_norm_backward_elemt.cpp) | 通过[算子IR](op_graph/sync_batch_norm_backward_elemt_proto.h)构图方式调用SyncBatchNormBackwardElemt算子。         |