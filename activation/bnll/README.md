# BNLL

 ## 产品支持情况
 	 
|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：激活函数BNLL（Binomial Normal Log Likelihood），即数值稳定版本的softplus。
- 计算公式：

$$
BNLL(x) = \ln(1 + e^x)
$$

等价分段实现（保证数值稳定性）：

$$
y_i = \begin{cases} x_i + \ln(1 + e^{-x_i}) & \text{if } x_i > 0 \\ \ln(1 + e^{x_i}) & \text{otherwise} \end{cases}
$$

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
    <td>公式中的'x'，表示BNLL激活函数的输入张量。支持空Tensor，维度支持0-8。</td>
    <td>FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>表示x经BNLL计算得到的输出张量，shape和dtype与输入x一致。</td>
    <td>FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE图模式  | [test_geir_bnll.cpp](examples/test_geir_bnll.cpp) | 通过GE IR接口方式调用BNLL算子。 |
