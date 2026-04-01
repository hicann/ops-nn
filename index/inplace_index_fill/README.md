# InplaceIndexFill

##  产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：沿输入x的给定轴dim，将indices指定位置的值使用value进行原地填充替换。
- 示例：
输入x为:

  &emsp;&emsp;[[1, 2, 3],

  &emsp;&emsp;&nbsp;[4, 5, 6],

  &emsp;&emsp;&nbsp;[7, 8, 9]]

  若dim = 0，indices = [0, 2]，value = 0时，算子的计算结果为：

    &emsp;&emsp;[[0, 0, 0],

    &emsp;&emsp;&nbsp;[4, 5, 6],

    &emsp;&emsp;&nbsp;[0, 0, 0]]

  若dim = 1，indices = [0, 2]，value = 0时，算子的计算结果为：

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
        <td>x</td>
        <td>输入</td>
        <td>功能示例中的x，即待被在指定位置的值用value原地替换的张量。</td>
        <td>FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BOOL、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>输入</td>
        <td>指定x在dim维度将要填充的下标。其中的元素值小于x对应dim的维度大小，支持负数索引。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>指定填充的数据值。数据类型需与x一致。</td>
        <td>与x一致</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dim</td>
        <td>属性</td>
        <td>指定了x将要填充的维度。当x为1-8维时，dim的取值范围在[-x.dim(), x.dim())，当x为0维时，dim的取值范围在[-1, 1)。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x（输出）</td>
        <td>输出</td>
        <td>原地填充后的输出张量，shape与输入x一致。</td>
        <td>与输入x一致</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## 约束说明

- x与value的数据类型必须一致。
- indices的数据类型仅支持INT32和INT64。

## 调用说明

无