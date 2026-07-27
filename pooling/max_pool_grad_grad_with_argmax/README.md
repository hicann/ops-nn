# MaxPoolGradGradWithArgmax

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>                   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>                   |     ×    |
|  <term>Atlas 推理系列产品</term>                           |     ×    |
|  <term>Atlas 训练系列产品</term>                           |     ×    |

## 功能说明

- 算子功能：最大池化（MaxPool）的二阶梯度反向传播算子。固定局部索引模式（include_batch_in_index=false），根据`argmax`索引从`grad`张量中按batch内局部索引收集（gather）值到`output`张量。

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
      <td>原始池化输入张量，形状为[N, H, W, C]，固定NHWC格式。</td>
        <td>DOUBLE、FLOAT32、FLOAT16、INT16、INT32、INT64、INT8、UINT16、UINT32、UINT64、UINT8、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad</td>
        <td>输入</td>
        <td>对x的梯度张量，形状和数据类型与x相同。</td>
         <td>DOUBLE、FLOAT32、FLOAT16、INT16、INT32、INT64、INT8、UINT16、UINT32、UINT64、UINT8、BFLOAT16</td>
       <td>ND</td>
     </tr>
     <tr>
       <td>argmax</td>
      <td>输入</td>
      <td>池化最大值索引张量，形状为池化后输出shape，值为batch内局部索引。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>二阶梯度输出，形状与argmax相同，数据类型与x相同。</td>
        <td>DOUBLE、FLOAT32、FLOAT16、INT16、INT32、INT64、INT8、UINT16、UINT32、UINT64、UINT8、BFLOAT16</td>
       <td>ND</td>
     </tr>
     <tr>
       <td>ksize</td>
      <td>属性（必选）</td>
      <td>池化窗口大小，长度为4的列表。ksize[0]=1且ksize[3]=1。</td>
      <td>ListInt</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性（必选）</td>
      <td>窗口移动步长，长度为4的列表。strides[0]=1且strides[3]=1。</td>
      <td>ListInt</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性（必选）</td>
      <td>填充算法，取值为"SAME"或"VALID"。</td>
      <td>String</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入仅支持4维，固定NHWC格式。
- x.shape == grad.shape，argmax.shape == y.shape。
- ksize[0] == 1且strides[0] == 1（batch维度不池化），ksize[3] == 1且strides[3] == 1（通道维度不池化）。
- argmax为batch内局部索引（include_batch_in_index=false），值范围为[0, H*W*C)。
- 支持空tensor（totalElements == 0时直接返回空output）。
- x、grad、y的数据类型必须一致。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式接口 | [test_geir_max_pool_grad_grad_with_argmax](examples/arch35/test_geir_max_pool_grad_grad_with_argmax.cpp) | 通过IR [MaxPoolGradGradWithArgmax](./op_graph/max_pool_grad_grad_with_argmax_proto.h)构图方式调用MaxPoolGradGradWithArgmax算子。 |
