# Relu6

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |

## 功能说明

- 算子功能：激活函数Relu6，按元素对输入张量进行激活计算，将每个元素值限制在`[0, 6]`区间内。
- 计算公式：

$$
Relu6(x) = \min(\max(x, 0), 6)
$$

等价于分段函数：

$$
Relu6(x) =
\begin{cases}
0, & x < 0 \\
x, & 0 \le x \le 6 \\
6, & x > 6
\end{cases}
$$

其中：

- x: 输入张量

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
    <td>公式中的'x'，表示Relu6激活函数的输入张量。</td>
    <td>FLOAT16、FLOAT32、INT32、DT_BF16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>表示x经Relu6计算得到的输出张量，shape和dtype与输入x一致。</td>
    <td>FLOAT16、FLOAT32、INT32、DT_BF16</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- 输入x和输出y的shape必须相同。
- 最高支持8维张量。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式(GE IR)  | [test_geir_relu6.cpp](examples/test_geir_relu6.cpp) | 通过GE IR图模式方式调用Relu6算子。 |
