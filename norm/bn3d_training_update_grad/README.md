# BN3DTrainingUpdateGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     √    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：BN3DTrainingUpdateGrad是3D Batch Normalization（BN3D，对5D张量按通道归一化）训练反向传播的逐通道参数梯度归约算子。该算子把损失对BN前向输出y的梯度grads与前向输入x，结合逐通道统计量batch_mean、batch_variance，沿批与空间维（N、D、H、W）归约，得到损失对缩放参数gamma的梯度diff_scale与对偏置参数beta的梯度diff_offset。通道轴由grads的数据格式确定（NCDHW/NCHW位于dim1，NHWC位于末轴），常用于BN3D训练反向链中参数梯度的收集环节。

- float16/bfloat16输入在kernel内先升位为float32参与全部中间运算与归约，输出diff_scale/diff_offset恒为float32。

- 计算公式：

  $$
  x\_norm_{n,c,d,h,w} = \frac{x_{n,c,d,h,w} - batch\_mean_c}{\sqrt{batch\_variance_c + epsilon}}
  $$

  $$
  diff\_scale_c = \sum_{n,d,h,w} grads_{n,c,d,h,w} \cdot x\_norm_{n,c,d,h,w}
  $$

  $$
  diff\_offset_c = \sum_{n,d,h,w} grads_{n,c,d,h,w}
  $$

  其中：

  - c为通道坐标，n、(d, h, w)分别为批与空间坐标，逐通道统计量沿通道轴广播；
  - batch_variance为有偏方差口径（E[x²]−E[x]²），epsilon与前向BN3D保持一致；
  - diff_scale、diff_offset恒为float32，shape为通道数C。

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
      <td>NCDHW、NCHW、NHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>BN前向输入，公式中的x。shape、数据格式、数据类型需与grads完全一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>NCDHW、NCHW、NHWC</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>输入</td>
      <td>逐通道均值，公式中的batch_mean。shape为通道数C。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>输入</td>
      <td>逐通道有偏方差，公式中的batch_variance。shape为通道数C。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>属性</td>
      <td>可选属性，方差加的小常数，默认0.0001，需与前向BN3D一致。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diff_scale</td>
      <td>输出</td>
      <td>损失对缩放参数gamma的梯度，公式中的diff_scale。shape为通道数C。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diff_offset</td>
      <td>输出</td>
      <td>损失对偏置参数beta的梯度，公式中的diff_offset。shape为通道数C。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- grads与x的数据类型、shape、数据格式必须完全一致。
- batch_mean与batch_variance的通道维必须与grads的通道数C一致，非通道维恒为1。
- grads/x的数据格式只支持NCDHW、NCHW、NHWC具名格式，不支持ND。

## 调用说明

| 调用方式 | 调用接口 | 说明 |
| -------- | -------- | ---- |
| 图模式调用（GEIR） | 通过`op::BN3DTrainingUpdateGrad`构图，参见`examples/arch35/test_geir_bn3d_training_update_grad.cpp` | 通过Ascend Graph（GE IR）方式调用，编译并执行图。 |
