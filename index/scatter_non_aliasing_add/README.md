# ScatterNonAliasingAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：ScatterNonAliasingAdd执行"先拷贝、后散射累加"操作。将输入tensor x全量拷贝到输出y，然后根据indices指定的多维索引位置，将updates中的值累加到y的对应位置。输出y与输入x不共享内存（Non-Aliasing）。

- 计算公式：

$$
y = \text{copy}(x)
$$
$$
y[\text{indices}[i]] \mathrel{+}= \text{updates}[i], \quad i = 0, 1, \ldots, N-1
$$

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
      <td>源tensor，输出为其拷贝后散射累加updates。</td>
      <td>FLOAT16、FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>多维索引tensor，shape[-1] = K ≤ rank(x)，指定散射目标位置。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>待累加的更新值tensor，shape = indices.shape[:-1] + x.shape[K:]。</td>
      <td>FLOAT16、FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出tensor，shape = x.shape，dtype = x.dtype。</td>
      <td>FLOAT16、FLOAT、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输出y与输入x不共享内存（Non-Aliasing），y是独立分配的tensor。
- 所有输入输出均为FORMAT_ND（连续存储），不支持非连续tensor。
- 重复索引按indices原始顺序多次累加，结果确定性。
- indices中每个索引值必须满足0 ≤ index < x.shape[dim]，不支持负索引和越界索引。算子不对索引做合法性校验，调用方需保证索引合法，否则结果未定义。
- 支持空tensor（shape含0维），此时返回x的拷贝，不执行scatter add。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | - | 通过[算子IR](op_graph/scatter_non_aliasing_add_proto.h)接口方式调用ScatterNonAliasingAdd算子。 |
