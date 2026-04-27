# DataFormatDimMap

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：根据源数据格式和目标数据格式的维度映射关系，将输入的维度索引转换为目标格式下的对应维度索引。
- 计算示例：输入维度索引 `[0, 1, 2, 3]`，源格式 `NHWC`，目标格式 `NCHW`，输出 `[0, 2, 3, 1]`。即 NHWC 的第 1 维 (H) 对应 NCHW 的第 2 维，第 2 维 (W) 对应第 3 维，第 3 维 (C) 对应第 1 维。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>维度索引张量，取值范围为 [-(格式长度), 格式长度)。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>src_format</td>
    <td>可选属性</td>
    <td><ul><li>源数据格式字符串，如 "NHWC"。</li><li>默认值为 "NHWC"。</li></ul></td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dst_format</td>
    <td>可选属性</td>
    <td><ul><li>目标数据格式字符串，如 "NCHW"。</li><li>默认值为 "NCHW"。</li></ul></td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>映射后的维度索引张量，shape 与输入 x 相同。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- src_format 和 dst_format 必须包含相同的字符集（即相同的维度字母，顺序可不同）。
- 格式字符串长度支持 1~5。
- 输入 x 的 shape 维度数不超过 8。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="examples/arch35/test_aclnn_data_format_dim_map.cpp">test_aclnn_data_format_dim_map</a></td>
    <td>参见<a href="doc/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody></table>
