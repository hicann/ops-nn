# BN3DTrainingReduceGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：BN3DTrainingReduceGrad是3D Batch Normalization（BN3D，对5D张量按通道归一化）训练反向传播的elementwise收尾算子。该算子将1D通道参数张量沿通道轴广播到5D，把损失对BN前向输出y的梯度grads、前置归约段BN3DTrainingUpdateGrad产出的diff_scale/diff_offset以及前向统计量scale/batch_mean/batch_variance逐元素合成，得到损失对前向输入x的梯度dx（输出y）。支持NCDHW与NDHWC双布局（通道轴分别位于dim1与dim4），常用于BN3D训练反向链的最后一环。本算子无跨元素归约。

- 计算公式：

  $$
  y_{n,c,d,h,w} = \left( grads_{n,c,d,h,w} - \frac{diff\_scale_c \cdot (x_{n,c,d,h,w} - batch\_mean_c)}{num \cdot data\_sqrt_c} - \frac{diff\_offset_c}{num} \right) \cdot \frac{scale_c}{data\_sqrt_c}
  $$

  其中：

  - num = N·D·H·W：除通道数C外所有维的乘积（NCDHW与NDHWC两种布局下一致）；
  - data_sqrt_c = sqrt(batch_variance_c + epsilon)：方差加epsilon后的平方根；
  - n、c、(d, h, w)分别为批、通道与空间坐标，1D参数张量沿通道轴广播到5D；
  - batch_variance为有偏方差口径（E[x²]−E[x]²）；
  - 计算完成后输出y回落grads的dtype（float16为cast回落、bfloat16为round回落），与grads同shape、format、dtype。

- float16/bfloat16输入在kernel内先升位为float32参与全部中间运算，结果再回落原dtype。

## 参数说明

<table style="undefined;table-layout: fixed; width: 910px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>grads</td>
      <td>输入</td>
      <td>损失对BN前向输出y的梯度，公式中的grads。shape、数据格式、数据类型需与x完全一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>NCDHW、NDHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>BN前向输入，公式中的x。shape、数据格式、数据类型需与grads完全一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>NCDHW、NDHWC</td>
    </tr>
    <tr>
      <td>diff_scale</td>
      <td>输入</td>
      <td>前置BN3DTrainingUpdateGrad输出，损失对scale的梯度，公式中的diff_scale。shape为(C,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diff_offset</td>
      <td>输入</td>
      <td>前置BN3DTrainingUpdateGrad输出，损失对offset的梯度，公式中的diff_offset。shape为(C,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>BN前向缩放参数γ，公式中的scale。shape为(C,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>输入</td>
      <td>x的批均值（前置BN3DTrainingUpdate输出），公式中的batch_mean。shape为(C,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>输入</td>
      <td>x的批方差（前置BN3DTrainingUpdate输出，有偏口径），公式中的batch_variance。shape为(C,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>数值稳定性常量，加在batch_variance上再开方，公式中的epsilon，必须大于0。</li><li>默认值为0.0001。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>损失对前向输入x的梯度dx，公式中的y。shape、数据格式、数据类型与grads一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>NCDHW、NDHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- grads，x和y必须同为5维且shape、数据格式、数据类型完全一致。
- diff_scale、diff_offset、scale、batch_mean、batch_variance必须为1维，长度等于通道数C，数据类型必须为FLOAT。NCDHW布局通道轴为dim1，NDHWC布局通道轴为dim4。
- grads、x、y不支持空Tensor（任一维为0时返回错误）。
- epsilon必须大于0。
- batch_variance必须使用有偏方差口径（E[x²]−E[x]²）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---------------- | --------------- | ----------- |
| GE图模式 | [test_geir_bn3_d_training_reduce_grad](examples/arch35/test_geir_bn3_d_training_reduce_grad.cpp) | 通过[算子IR](op_graph/bn3_d_training_reduce_grad_proto.h)构图方式调用BN3DTrainingReduceGrad算子 |
