# AvgPool3D

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|
|Kirin X90 处理器系列产品|√|
|Kirin 9030 处理器系列产品|√|
## 功能说明

- 算子功能：对输入Tensor进行窗口为$kD * kH * kW$、步长为$sD * sH * sW$的三维平均池化操作，其中$k$为kernelSize，表示池化窗口的大小，$s$为stride，表示池化操作的步长。

- 计算公式：
  输入input($N,C,D_{in},H_{in},W_{in}$)、输出out($N,C,D_{out},H_{out},W_{out}$)和池化步长($stride$)、池化窗口大小kernelSize($kD,kH,kW$)的关系是

$$
D_{out}=\lfloor \frac{D_{in}+2*padding[0]-kernelSize[0]}{stride[0]}+1 \rfloor
$$

$$
H_{out}=\lfloor \frac{H_{in}+2*padding[1]-kernelSize[1]}{stride[1]}+1 \rfloor
$$

$$
W_{out}=\lfloor \frac{W_{in}+2*padding[2]-kernelSize[2]}{stride[2]}+1 \rfloor
$$

$$
out(N_i,C_i,d,h,w)=\frac{1}{kD*kH*kW}\sum_{k=0}^{kD-1}\sum_{m=0}^{kH-1}\sum_{n=0}^{kW-1}input(N_i,C_i,stride[0]*d+k,stride[1]*h+m,stride[2]*w+n)
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
      <td>表示待转换的张量，公式中的`input`。</td>
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
      <td>表示在输入的D、H、W方向上pads补0的层数，公式中的`padding`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>可选属性</td>
      <td>表示计算输出shape时，向下取整（False），否则向上取整。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>count_include_pad</td>
      <td>可选属性</td>
      <td>表示平均计算中包括零填充（True），否则不包括。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisor_override</td>
      <td>可选属性</td>
      <td>如果指定，它将用作平均计算中的除数，当值为0时，该属性不生效。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td>输入数据格式，支持"NDHWC"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行AvgPool3D计算的出参，公式中的`out`。数据类型、数据格式需要与`x`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_avgpool3d.cpp](examples/test_aclnn_avgpool3d.cpp) | 通过[aclnnAvgPool3d](docs/aclnnAvgPool3d.md)接口方式调用AvgPool3d算子。 |
