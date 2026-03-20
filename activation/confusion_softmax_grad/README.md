# ConfusionSoftmaxGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                     |    √     |
| <term>Atlas 推理系列产品 </term>                            |    √     |
| <term>Atlas 训练系列产品</term>                             |    √     |

## 功能说明

- 算子功能：融合算子，将mul、sum、sub进行融合。

- 计算公式：

  $$
  y = grad - sum(grad \cdot x, axis=-1)
  $$

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 813px"><colgroup>
  <col style="width: 103px">
  <col style="width: 144px">
  <col style="width: 194px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>grad</td>
      <td>输入</td>
      <td>反向传播的梯度值，即上一层的输出梯度。公式中的grad。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>公式中的x，数据类型和shape与grad一致。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的y，数据类型和shape与grad一致</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    </tbody>
     </table>
- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT32。 

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_confusion_softmax_grad](./examples/test_geir_confusion_softmax_grad.cpp)   | 通过[算子IR](./op_graph/confusion_softmax_grad_proto.h)构图方式调用ConfusionSoftmaxGrad算子。 |
