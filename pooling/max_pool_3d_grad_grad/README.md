# MaxPool3DGradGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：MaxPool3DGradGrad计算三维最大池化一次反向传播对其上游梯度的梯度。算子按D→H→W顺序扫描每个池化窗口中的有效输入位置；存在匹配时，从首个满足`orig_x`等于`orig_y`目标值的位置读取`grads`，否则输出FP16零。

### 计算公式

$$
p^*(q)=\operatorname{first}\{p\in W_{valid}(q)\mid orig\_x(p)=orig\_y(q)\}
$$

$$
y(q)=
\begin{cases}
grads(p^*(q)), & p^*(q)\text{存在} \\
0_{FP16}, & p^*(q)\text{不存在}
\end{cases}
$$

其中$q=(n,od,oh,ow,c)$，$W_{valid}$只包含落在`orig_x`范围内的有效输入位置；SAME padding位置不参与相等比较。

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
      <td>orig_x</td>
      <td>输入</td>
      <td>前向MaxPool3D的输入，shape为`(N,D,H,W,C)`。</td>
      <td>FLOAT16</td>
      <td>ND（逻辑NDHWC）</td>
    </tr>
    <tr>
      <td>orig_y</td>
      <td>输入</td>
      <td>由orig_x和池化属性推导的前向输出，shape为`(N,Do,Ho,Wo,C)`。</td>
      <td>FLOAT16</td>
      <td>ND（逻辑NDHWC）</td>
    </tr>
    <tr>
      <td>grads</td>
      <td>输入</td>
      <td>二阶上游输入，shape与orig_x完全相同。</td>
      <td>FLOAT16</td>
      <td>ND（逻辑NDHWC）</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>每个池化窗口的梯度结果，shape与orig_y完全相同。</td>
      <td>FLOAT16</td>
      <td>ND（逻辑NDHWC）</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>必选属性，表示D/H/W方向的池化窗口大小。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>必选属性，表示D/H/W方向的池化步长。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>必选属性，长度为6的非负padding配置；全0表示VALID，任一非0表示SAME。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>可选属性，逻辑数据格式，默认值为NDHWC。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- `orig_x`、`orig_y`、`grads`和`y`仅支持FLOAT16、ND物理格式和逻辑NDHWC。
- 所有输入输出均为非空rank=5张量，且各维必须大于0；不支持标量、1D、空张量或8D张量。
- `orig_x`与`grads`的shape必须完全相同；`orig_y`与`y`的shape必须完全相同。
- `orig_x`与`orig_y`的N、C维必须相同，`orig_y`的D、H、W维必须与`ksize`、`strides`及`pads`推导的池化输出一致。
- `ksize`和`strides`长度仅支持1、3或5，空间值必须大于0；长度为5时首尾元素必须为1，禁止在N、C轴池化。
- `pads`长度必须为6且元素非负；全0按VALID处理，任一非0按SAME处理。
- `data_format`仅支持NDHWC。
- 多个`orig_x`值与目标相等时，固定选择D→H→W扫描顺序中的首个位置。无匹配或目标为NaN时输出FP16零；K1路径同样先比较`orig_x`与`orig_y`，不匹配时输出零。
- SAME边界只扫描有效输入位置；padding不构造可见哨兵，也不参与`orig_x == orig_y`匹配。

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
    <td><a href="./examples/test_geir_max_pool_3d_grad_grad.cpp">test_geir_max_pool_3d_grad_grad</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
