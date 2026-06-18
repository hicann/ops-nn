# CeluV2

 ## 产品支持情况
 	 
|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：激活函数Celu（Continuously Differentiable Exponential Linear Units）。
- 计算公式：

$$
CELU(x) = \max(0,x) + \min(0, \alpha \ast (\exp(x/\alpha) - 1))
$$

其中：
- x: 输入张量

- alpha: CELU公式中的激活系数，默认值为1.0

## 参数说明

<table style="table-layout: auto; width: 100%">
<thead>
    <tr>
    <th style="white-space: nowrap">参数名</th>
    <th style="white-space: nowrap">输入/输出</th>
    <th style="white-space: nowrap">描述</th>
    <th style="white-space: nowrap">数据类型</th>
    <th style="white-space: nowrap">数据格式</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>公式中的'x'，表示CELU激活函数的输入张量。支持空Tensor，支持非连续Tensor，维度支持0-8。</td>
    <td>FLOAT16、FLOAT32、DT_BF16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>alpha</td>
    <td>属性</td>
    <td>CELU公式中的激活系数，默认值为1.0，不能为0。</td>
    <td>FLOAT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>表示x经CELU计算得到的输出张量，shape和dtype与输入x一致。支持非连续Tensor。</td>
    <td>FLOAT16、FLOAT32、DT_BF16</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_celu.cpp](examples/test_aclnn_celu.cpp) | 通过[aclnnCelu&aclnnInplaceCelu](docs/aclnnCelu&aclnnInplaceCelu.md)接口方式调用CeluV2算子。 |
