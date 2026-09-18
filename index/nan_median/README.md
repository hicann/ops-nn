# NanMedian

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：忽略NaN后，返回输入张量x在指定维度dim上的中位数及其在原输入中的索引；有效元素个数为偶数时取两个中间值中较小的值。
- 计算流程：
  沿dim维度排除NaN后，将剩余元素按升序排列，记有效元素个数为n、排序后的值为s、对应的原始索引为p。当n大于0时：

  $$m = \left\lfloor \frac{n - 1}{2} \right\rfloor, \quad y = s[m], \quad indices = p[m]$$

  例如输入张量x=[[1, NaN, 3, 2], [4, 6, NaN, 5]]，dim=1，得到y=[[2], [5]]，indices=[[3], [3]]。

  若某个切片中的元素全部为NaN，则对应的中位数为NaN。对于整数输入，计算结果与Median相同。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 215px">
  <col style="width: 163px">
  <col style="width: 287px">
  <col style="width: 439px">
  <col style="width: 135px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待计算的输入张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>中位数。数据类型与x相同，shape与x相同但dim维度的长度为1。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输出</td>
      <td>选中元素在输入张量dim维度上的索引，从0开始。shape与y相同。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>可选属性</td>
      <td><ul><li>指定计算的维度，取值范围为[-rank(x), rank(x) - 1]。</li><li>默认值为-1，表示最后一维。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

输入张量x不能为空，且维度数必须大于0。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_nan_median](./examples/test_aclnn_nan_median.cpp) | 通过[aclnnNanMedian](./docs/aclnnNanMedian.md)接口忽略NaN后计算所有元素的中位数。 |
| aclnn调用 | [test_aclnn_nan_median_dim](./examples/test_aclnn_nan_median_dim.cpp) | 通过[aclnnNanMedianDim](./docs/aclnnNanMedianDim.md)接口在指定维度上忽略NaN后计算中位数及索引，通过keepDim控制输出是否保留归约维度。 |
