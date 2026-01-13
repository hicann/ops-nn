# IndexFillD

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 接口功能：沿输入self的给定轴dim，将index指定位置的值使用value进行替换。
- 示例：
输入self为:

  &emsp;&emsp;[[1, 2, 3],

  &emsp;&emsp;&nbsp;[4, 5, 6],

  &emsp;&emsp;&nbsp;[7, 8, 9]]

  若dim = 0，index = [0, 2]，value = 0时，算子的计算结果为：

    &emsp;&emsp;[[0, 0, 0],

    &emsp;&emsp;&nbsp;[4, 5, 6],

    &emsp;&emsp;&nbsp;[0, 0, 0]]

  若dim = 1，index = [0, 2]，value = 0时，算子的计算结果为：

    &emsp;&emsp;[[0, 2, 0],

    &emsp;&emsp;&nbsp;[0, 5, 0],

    &emsp;&emsp;&nbsp;[0, 8, 0]]

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
        <th>输入/输出</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>输入</td>
        <td>功能示例中的self，即待被在指定位置的值用value替换的张量。</td>
        <td>FLOAT16、FLOAT、INT32、INT64、BOOL、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>输入</td>
        <td>指定了self将要填充的维度。当self为1-8维时，dim的取值范围在[-self.dim(), self.dim())，当self为0维时，dim的取值范围在[-1, 1)。</td>
        <td>int64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>输入</td>
        <td>指定self在dim维度将要填充的下标。其中的元素值小于self对应dim的维度大小。</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>指定填充的数据值。需要可转化为self的数据类型。</td>
        <td>与self一致</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>指定的输出张量。</td>
        <td>与self一致</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## 约束说明

无
## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_index_fill_tensor](./examples/test_aclnn_index_fill_tensor.cpp) | 通过[aclnnIndexFillTensor](./docs/aclnnIndexFillTensor&aclnnInplaceIndexFillTensor.md)接口方式调用IndexFillD算子。 |
