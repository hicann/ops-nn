# CrossEntropyGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|×|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|

## 功能说明

- 算子功能：完成 CrossEntropy 梯度计算。

- 计算公式：

$$
grad\_input = softmax(logits) - one\_hot(target)
$$

其中 softmax 为稳定版（减去每行最大值），one_hot(target) 在 target 索引位置为 1，其余为 0。

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
      <td>logits</td>
      <td>输入</td>
      <td>预测 logits，形状为 [*, num_classes]。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>目标类别索引，形状与 logits 的 batch 维度一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>CrossEntropy 梯度计算结果。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- logits: FLOAT 仅可与 target: INT32 搭配使用。
- logits: FLOAT16 仅可与 target: INT32 搭配使用。
- logits: BFLOAT16 仅可与 target: INT32 搭配使用。
- 当 logits 最后维度（num_classes）过大导致 UB 无法容纳完整行时，自动启用列分割模式。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/test_aclnn_cross_entropy_grad.cpp">test_aclnn_cross_entropy_grad</a></td>
    <td>参见算子调用完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
