# aclnnEdgeSoftmax

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     x    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对输入图的每个节点的所有入边特征，按节点分组进行 softmax 归一化。
- 计算公式：对于一个节点 $i$，

  $$
  a_{ij} = \frac{\exp(z_{ij})}{\sum_{j \in \mathcal{N}(i)} \exp(z_{ij})}
  $$

  其中 $z_{ij}$ 是边 $j \rightarrow i$ 的信号，在 softmax 的上下文中也称为 logits。$\mathcal{N}(i)$ 是指向节点 $i$ 的所有节点的集合。

## 参数说明

<table style="undefined;table-layout: fixed; width: 919px"><colgroup>
  <col style="width: 130px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>边特征张量</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx</td>
      <td>输入</td>
      <td>目标节点索引</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>n</td>
      <td>输入</td>
      <td>节点总数</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>归一化后边特征</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无
