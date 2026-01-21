# AdaptiveMaxPool3d

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|昇腾910_95 AI处理器|×|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|
|Atlas 200/300/500 推理产品|×|

## 功能说明

- 算子功能：根据输入的output_size计算每次kernel的大小，对输入x进行3维最大池化操作，输出池化后的值y和索引indices。

- 计算公式：
  y tensor中对于DHW轴上每个位置为$(l,m,n)$的元素来说，其计算公式为：
  
  $$
  D^{l}_{left} = floor((l*D)/D_o)
  $$
  
  $$
  D^{l}_{right} = ceil(((l+1)*D)/D_o)
  $$
  
  $$
  H^{m}_{left} = floor((m*H)/H_o)
  $$
  
  $$
  H^{m}_{right} = ceil(((m+1)*H)/H_o)
  $$
  
  $$
  W^{n}_{left} = floor((n*W)/W_o)
  $$
  
  $$
  W^{n}_{right} = ceil(((n+1)*W)/W_o)
  $$
  
  $$
  y(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{max} input(N,C,i,j,k)
  $$
  
  $$
  indices(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{argmax} input(N,C,i,j,k)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>待进行AdaptiveMaxPool3d计算的入参。D轴H轴W轴3个维度的乘积不能大于int32的最大表示。数据类型与出参`y`的保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>表示输出结果在D，H，W维度上的空间大小。数据类型与入参`x`的保持一致。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行AdaptiveMaxPool3d计算的出参。shape与出参`indices`的保持一致，数据类型与入参`x`的保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输出</td>
      <td>表示`y`元素在输入`x`中的索引位置。shape与出参`y`的保持一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明
Shape描述：
  - self.shape = (N, C, Din, Hin, Win)
  - outputSize = [Dout, Hout, Wout]
  - outputOut.shape = (N, C, Dout, Hout, Wout)
  - indicesOut.shape = (N, C, Dout, Hout, Wout)


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_adaptive_max_pool3d.cpp](examples/test_aclnn_adaptive_max_pool3d.cpp) | 通过[aclnnAdaptiveMaxPool3d](docs/aclnnAdaptiveMaxPool3d.md)接口方式调用AdaptiveMaxPool3d算子。 |
