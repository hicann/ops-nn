# AscendRequant

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

AscendRequant 是 INT8 静态量化推理链路中的反量化再量化算子，将上游 Conv/MatMul 在 INT8 权重下产生的 INT32 累加结果，按反量化缩放因子重新量化回 INT8，作为下一层 INT8 计算的输入。`req_scale` 为 UINT64 类型的硬件亲和反量化缩放因子（DeqScale 编码），按 numpy 广播规则作用到 `x`；计算结果按 round-to-nearest-even 取整并饱和到 INT8 范围 `[-128, 127]`。

计算公式：

$$
y = \text{SaturateInt8}(\text{RoundToNearestEven}(x \times scale))
$$

其中 `scale` 由 `req_scale`（UINT64 DeqScale 编码）解码得到，`SaturateInt8` 将结果饱和到 `[-128, 127]`。当 `relu_flag=true` 时再执行 `y = max(y, 0)`，对负值截零。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
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
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>输入</td>
<td>待再量化的 INT32 累加结果，对应公式中x。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>req_scale</td>
<td>输入</td>
<td>UINT64 硬件亲和反量化缩放因子（DeqScale 编码），对应公式中scale的来源。</td>
<td>UINT64</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>再量化 INT8 结果，shape与x相同，对应公式中y。</td>
<td>INT8</td>
<td>ND</td>
</tr>
<tr>
<td>relu_flag</td>
<td>可选属性</td>
<td>是否对再量化结果融合ReLU，默认false，true时对负值截零。</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- 数据类型固定组合：`x` 为 INT32、`req_scale` 为 UINT64、`y` 为 INT8，不支持其他数据类型组合。
- 数据格式仅支持 ND。
- `x` 的维度范围为 1-8；`req_scale` 的维度范围为 0-8，且 `rank(req_scale)` 需小于等于 `rank(x)`。
- `req_scale` 的 shape 需可按 numpy 广播规则广播到 `x` 的 shape，广播在 Kernel 内处理；输出 shape 等于 `x` 的 shape。
- 输入需为连续 Tensor，不支持非连续 Tensor。

## 调用说明

<table style="table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 600px">
</colgroup>
<thead>
<tr>
<th>调用方式</th>
<th>样例代码</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td>GE图模式</td>
<td>-</td>
<td>通过 <a href="op_graph/ascend_requant_proto.h">AscendRequant算子IR</a> 注册图引擎算子原型。</td>
</tr>
</tbody>
</table>
