# TensorScatterSub

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

> 说明：Ascend 950PR/950DT 由本目录 kernel 直接支持；Atlas A2/A3 系列产品通过图模式融合规则（TensorScatterSubFusionPass，融合为 TensorMove + ScatterNdSub）支持，无独立 kernel。

## 功能说明

- 算子功能：将x拷贝到输出y，再按照indices指定的位置，逐条将updates中对应的slice从y中减去。重复索引按输入顺序串行累减，结果确定。

- 计算公式：

$$
y = copy(x)\\
for\ i \in [0, N):\ y[indices_i] \mathrel{-}= updates_i
$$

其中N为indices.shape[:-1]的乘积，即slice条数。

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
      <td>待进行tensor_scatter_sub计算的源tensor，公式中的x。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引tensor，最后一维长度K表示每条索引覆盖x的前K个维度。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>待减去的slice数据，公式中的updates。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>计算结果tensor，shape与x相同。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- x、updates、y的dtype必须一致；indices的dtype为INT32或INT64。
- y的shape与x的shape完全相同。
- x的维度范围为1D~8D。
- 记K=indices.shape[-1]，必须满足K<=x的维度数；updates的shape必须等于indices.shape[:-1]拼接x.shape[K:]。
- indices中第j个索引分量的取值必须在x第j维的合法范围[0,x.shape[j])内，越界索引属于非法输入，kernel内跳过该条目。
- indices存在重复索引时，按输入顺序逐条串行累减，结果确定。
- x、indices、updates允许为空tensor（元素数为0），此时y为x的拷贝。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_tensor_scatter_sub.cpp">test_geir_tensor_scatter_sub</a></td>
    <td>通过<a href="./op_graph/tensor_scatter_sub_proto.h">算子IR</a>构图方式调用TensorScatterSub算子。</td>
  </tr>
</tbody>
</table>
