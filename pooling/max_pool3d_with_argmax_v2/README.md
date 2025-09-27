# MaxPool3dWithArgmaxV2

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|

## 功能说明

- 算子功能：对于输入信号的输入通道，提供3维最大池化（Max pooling）操作，输出池化后的值out和索引indices。
- 计算公式：
  
  * output tensor中每个元素的计算公式：
    
    $$
    out(N_i, C_j, d, h, w) = \max\limits_{{k\in[0,k_{D}-1],m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times d + k, stride[1]\times h + m, stride[2]\times w + n)
    $$
  * out tensor的shape推导公式 (默认ceilMode=false，即向下取整)：
    
    $$
    [N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lfloor{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rfloor + 1,\lfloor{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rfloor + 1, \lfloor{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rfloor + 1]
    $$
  * out tensor的shape推导公式 (默认ceilMode=true，即向上取整)：
    
    $$
    [N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lceil{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rceil + 1,\lceil{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rceil + 1, \lceil{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rceil + 1]
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 500px">
  <col style="width: 300px">
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
      <td>self</td>
      <td>输入</td>
      <td>待进行MaxPool3dWithArgmaxV2计算的入参。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>输入</td>
      <td>表示最大池化的窗口大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>表示窗口移动的步长。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>输入</td>
      <td>表示每一条边补充的层数，补充的位置填写“负无穷”。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>输入</td>
      <td>表示控制窗口中元素的步幅。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>输入</td>
      <td>为True时表示计算输出形状时用向上取整的方法，为False时则表示向下取整。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输入</td>
      <td>表示池化后的结果。数据类型、数据格式需要与`self`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输出</td>
      <td>最大值的索引位置组成的Tensor。shape需要与`out`一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明
- 输入数据排布不支持NDHWC。
- kernelSize、stride、padding、dilation、ceilMode参数需要保证输出out shape中不存在小于1的轴。
- 当ceilMode为True的时候，如果滑动窗口全部在右侧padding区域上，这个输出结果将被忽略。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_max_pool3d_with_argmax.cpp](examples/test_aclnn_max_pool3d_with_argmax.cpp) | 通过[aclnnMaxPool3dWithArgmax](docs/aclnnMaxPool3dWithArgmax.md)接口方式调用MaxPool3dWithArgmaxV2算子。 |

