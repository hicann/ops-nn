# ChamferDistanceGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 算子功能：ChamferDistance（倒角距离）的反向算子，根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。
- 计算公式：
  
  假设有两个点集：  xyz1=[B,N,2], xyz2=[B,M,2]
  
  - ChamferDistance（倒角距离）正向算子计算公式为：
    
    $dist1_i=Min((x_{1_i}−x_2)^2+(y_{1_i}−y_2)^2)，x_2, y_2∈xyz2$
    $dist2_i=Min((x_{2_i}-x_1)^2+(y_{2_i}-y_1)^2)，x_1,y_1∈xyz1$
  
  - 反向算子即为对该公式求导，计算公式为：
    - $dist1_i 对x_{1_i} $的导数$=2*grad\_dist1*(x_{1_i}-x_2)$

      其中：$x_{1_i}∈xyz1$，$x_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的横坐标，单点求导公式如上，因为单点梯度更新的位置是连续的，所以考虑多点并行计算。
      
    - $dist1_i 对y_{1_i} $的导数$=2*grad\_dist1*(y_{1_i}-y_2)$
      
      其中$y_{1_i}∈xyz1$，$y_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的纵坐标，单点求导公式如上，因为单点梯度更新的位置是连续的，所以也可以考虑多点并行计算。
      
    - $dist1_i 对x_2 $的导数$=-2*grad\_dist1*(x_{1_i}-x_2)$
      
      其中$x_{1_i}∈xyz1，x_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的横坐标，单点求导公式如上，因为单点梯度需要根据最小距离值对应的索引值去更新，所以这块无法并行只能单点计算。
    
    - $dist1_i 对y_2 $的导数$=-2*grad\_dist1*(y_{1_i}-y_2)$
      
      其中$y_{1_i}∈xyz1，y_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的纵坐标，单点求导公式如上，因为单点梯度需要根据最小值对应的索引值去更新，所以这块也无法并行只能单点计算。
  
  对应$dist2_i$对$x_{2_i} 、x_1、y_{2_i} 、y_1$的导数和上述过程类似，这里不再赘述。
  
  最终计算公式如下，i∈[0,n)：

  $grad_xyz1[2*i] = 2*grad\_dist1*(x_{1_i}-x_2) - 2*grad\_dist1*(x_{1_i}-x_2)$
  
  $grad_xyz1[2*i+1] = 2*grad\_dist1*(y_{1_i}-y_2) - 2*grad\_dist1*(y_{1_i}-y_2)$
  
  $grad_xyz2[2*i] = 2*grad\_dist2*(x_{1_i}-x_2) - 2*grad\_dist2*(x_{1_i}-x_2)$
  
  $grad_xyz2[2*i+1] = 2*grad\_dist2*(y_{1_i}-y_2) - 2*grad\_dist2*(y_{1_i}-y_2)$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 60px">
  <col style="width: 70px">
  <col style="width: 150px">
  <col style="width: 80px">
  <col style="width: 80px">
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
      <td>xyz1</td>
      <td>输入</td>
      <td>算子正向输入的张量。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>xyz2</td>
      <td>输入</td>
      <td>算子正向输入的张量。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx1</td>
      <td>输入</td>
      <td>正向算子输出的张量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx2</td>
      <td>输入</td>
      <td>正向算子输出的张量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_dist1</td>
      <td>输入</td>
      <td>正向输出dist1的反向梯度，也是反向算子的初始梯度。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_dist2</td>
      <td>输入</td>
      <td>正向输出dist2的反向梯度，也是反向算子的初始梯度。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_xyz1</td>
      <td>输出</td>
      <td>梯度更新之后正向算子输入xyz1对应的梯度。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_xyz2</td>
      <td>输出</td>
      <td>梯度更新之后正向算子输入xyz2对应的梯度。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

xyz1、xyz2的shape为(B, N, 2)，grad_dist1、grad_dist2、idx1、idx2的shape为(B, N)。所有输入shape中的B和N数值保持一致。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_chamfer_distance_backward](./examples/test_aclnn_chamfer_distance_grad.cpp) | 通过[aclnnChamferDistanceBackward](./docs/aclnnChamferDistanceBackward.md)接口方式调用ChamferDistanceGrad算子。    |


