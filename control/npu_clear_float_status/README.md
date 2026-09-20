# NpuClearFloatStatus

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：清除NPU每个AI core的浮点溢出状态寄存器，输出固定为8个float32零值。

- 计算公式：

$$
data = zeros(8, dtype=float32)
$$

## 参数说明

<table><thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>addr</td>
    <td>输入</td>
    <td>地址占位符，shape为(8,)，数据内容不参与计算。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>data</td>
    <td>输出</td>
    <td>固定输出8个float32零值，shape为(8,)。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- addr数据类型必须为float32。
- 输出data固定为8个float32零值，与输入数据内容无关。
- addr仅作为算子输入接口占位符，其数据内容不参与计算。

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
    <td><a href="./examples/test_geir_npu_clear_float_status.cpp">test_geir_npu_clear_float_status</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
