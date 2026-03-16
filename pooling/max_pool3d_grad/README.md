# MaxPool3DGrad

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|×|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|

## 功能说明

算子功能：正向最大池化后反向传播，将梯度回填到每个窗口最大值的坐标处，相同坐标处累加。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
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
      <td>orig_x</td>
      <td>输入</td>
      <td>待进行MaxPool3DGrad计算的入参，表示正向的输入Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>orig_y</td>
      <td>输入</td>
      <td>表示正向输入中最大元素的索引位置。数据格式需要与`orig_x`一致，shape需要与`grads`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>grads</td>
      <td>输入</td>
      <td>待进行MaxPool3DGrad计算的入参，表示当前节点的梯度。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>表示最大池化的窗口大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>表示池化操作的步长。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>可选属性</td>
      <td>pad模式，本次仅支持SAME和VALID,填补的数据为-inf，默认SAME。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>当pad模式为CALCULATED时，输入的orig_x的shape进行左右上下前后的扩展，填补的数据为-inf，暂不生效。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td>表示支持的数据格式。</td>
      <td>-</td>
      <td>本次仅支持NCDHW，后续支持NDHWC</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行MaxPool3DGrad计算的出参。shape、数据格式需要与`orig_x`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW</td>
    </tr>
  </tbody></table>

## 约束说明
无。


## 调用说明
不涉及。
