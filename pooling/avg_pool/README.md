# AvgPool

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|√|
|Atlas 训练系列产品|√|

## 功能说明

- 接口功能：对输入Tensor进行窗口为$kH * kW$、步长为$sH * sW$的二维平均池化操作，其中$k$为kernelSize，表示池化窗口的大小，$s$为stride，表示池化操作的步长。
- 计算公式：
  输入input（$N,C,H,W$）、输出out（$N,C,H_{out},W_{out}$）和池化步长（$strides$）、池化窗口大小($kH,kW$)的关系是

  $$
  H_{out}=\lfloor \frac{H_{in}+2*paddings[0]-kH}{strides[0]}+1 \rfloor
  $$

  $$
  W_{out}=\lfloor \frac{W_{in}+2*paddings[1]-kW}{strides[1]}+1 \rfloor
  $$

  $$
  out(N_i,C_i,h,w)=\frac{1}{kH*kW}\sum_{m=0}^{kH-1}\sum_{n=0}^{kW-1}input(N_i,C_i,strides[0]*h+m,strides[1]*w+n)
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
      <td>输入tensor，公式中的input</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出tensor，公式中的output</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>表示池化窗口大小，公式中的`kernelSize`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>表示池化操作的步长，公式中的`stride`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>表示在输入的D、H方向上pads补0的层数，公式中的`padding`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td>表示输入的数据类型，默认为NHWC</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无。


## 调用说明

无。
