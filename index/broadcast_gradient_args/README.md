# BroadcastGradientArgs

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×   |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 算子功能：

  BroadcastGradientArgs 是 TensorFlow 中用于计算梯度传播所需的广播维度索引的算子。

  它的核心功能是：在反向传播过程中，根据两个张量在正向传播时的原始形状，自动识别出它们因广播机制而扩展的维度，并输出需要在哪些维度上对梯度进行约简，以便将梯度从广播后的形状还原为每个原始张量的形状。

- 示例：

  ```
  例1：
    原始张量a的shape为[2, 1, 4, 1, 6]
    原始张量b的shape为[2, 3, 1, 5, 1]
    x1_data: [2, 1, 4, 1, 6]   # x1_shape=[5]
    x2_data: [2, 3, 1, 5, 1]   # x2_shape=[5]
    
    y1_data: [1, 3]            # y1_shape=[2]
    y2_data: [2, 4]            # y2_shape=[2]

  例2：
    原始张量a的shape为[4, 1, 6]
    原始张量b的shape为[2, 3, 1, 5, 1]
    x1_data: [4, 1, 6]         # x1_shape=[3]
    x2_data: [2, 3, 1, 5, 1]   # x2_shape=[5]
    
    y1_data: [0, 1, 3]         # y1_shape=[3]
    y2_data: [2, 4]            # y2_shape=[2]

  例3：
    原始张量a的shape为[2, 1, 4, 1, 6]
    原始张量b的shape为[2, 1, 4, 1, 6]
    x1_data: [2, 1, 4, 1, 6]   # x1_shape=[5]
    x2_data: [2, 1, 4, 1, 6]   # x2_shape=[5]
    
    y1_data: []            # y1_shape=[0]
    y2_data: []            # y2_shape=[0]
  ```

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x1</td>
      <td>输入</td>
      <td>shape必须是1维，data是示例中原始张量a的shape。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>shape必须是1维，data是示例中原始张量b的shape，数据类型与x1的数据类型保持一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>shape必须是1维，表示x1对应的张量shape中需要广播的索引，数据类型与x1的数据类型保持一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>shape必须是1维，表示x2对应的张量shape中需要广播的索引，数据类型与x1的数据类型保持一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_broadcast_gradient_args](./examples/test_geir_broadcast_gradient_args.cpp)   | 通过[算子IR](./op_graph/broadcast_gradient_args_proto.h)构图方式调用BroadcastGradientArgs算子。 |
