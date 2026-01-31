# MaxPoolGradWithArgmaxV3

##  产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>                   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>                   |     ×    |
|  <term>Atlas 推理系列产品</term>                           |     ×    |
|  <term>Atlas 训练系列产品</term>                           |     ×    |

## 功能说明

- 算子功能：正向最大池化MaxPoolGradWithArgmaxV3的反向。

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
      <td>输入x</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>梯度Tensor</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>输入</td>
      <td>输入grad</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>输入属性</td>
      <td>池化窗口大小</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>输入属性</td>
      <td>窗口移动步长</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>输入属性</td>
      <td>每一条边补充的层数</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>输入属性</td>
      <td>输出argmax的数据类型</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>输入属性</td>
      <td>控制窗口中元素步幅</td>
      <td>Int</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>输入属性</td>
      <td>为true是用向上取整的方法</td>
      <td>Bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>输入属性</td>
      <td>数据格式</td>
      <td>String</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_max_pool_grad_with_argmax_v3](examples/test_aclnn_max_pool_grad_with_argmax_v3.cpp) | 通过[aclnnMaxPool2dWithIndicesBackward](../max_pool3d_grad_with_argmax/docs/aclnnMaxPool2dWithIndicesBackward.md)接口方式调用aclnnMaxPoolGradWithArgmaxV3算子。|
