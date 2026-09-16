# UpdateTensorDesc

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

* 算子功能：更新输出Tensor的描述信息（tensor_desc）。算子接收一个占位输入`x`（kernel不读取其数据），按必填属性`shape`对输出`y`进行RMW（读-改-写）：将目标维度个数及各维大小写入`y`对应的128×int64描述缓冲区的固定槽位，其余槽位保留原值。常用于动态shape场景下输出TensorDesc的更新。

- 计算公式：

  设 $N = len(shape)$，输出`y`为128×int64描述缓冲区：

  $$
  y[3] = N,\quad y[4+i] = shape[i],\ i = 0,\dots,N-1
  $$

  其中`y`的其余槽位（下标0~2与4+N~127）保持原值不变。

- 示例：

  ```text
  输入x（占位输入，数据不参与计算）：
  tensor([1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.float32)
  属性shape：
  [4, 8, 4]
  输出y（RMW后，仅展示前8个元素）：
  tensor([1, 1, 1, 3, 4, 8, 4, 1], dtype=torch.int64)
  ```

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
      <td>x</td>
      <td>输入</td>
      <td>占位输入Tensor，数据不参与计算，kernel不读取其数据。</td>
      <td>bool/float16/float/float64/int8/int16/int32/int64/uint8/uint16/uint32/uint64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>描述缓冲区Tensor，dtype恒为int64，shape由属性`shape`推导得出。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>属性</td>
      <td>目标shape，必填，ListInt类型。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 该算子有一个输入`x`、一个输出`y`、一个必填属性`shape`。
- 输入`x`：rank ∈ [0, 8]，dtype支持bool/float16/float/float64/int8/int16/int32/int64/uint8/uint16/uint32/uint64共12种，format仅支持ND。
- 输出`y`：dtype恒为int64，format仅支持ND，shape由属性`shape`推导得出。
- 属性`shape`：rank ∈ [1, 124]，每个元素为非负整数，且各元素乘积（numel） ≥ 128。
- 输出`y`的元素总个数（numel） ≥ 128：kernel固定按128×int64整块RMW，numel大于128时仅前128个元素被读写，可观察效果等价于仅覆盖下标[3, 3+N)区间。
- 输出`y`的非写入槽位（下标0~2与4+N~127）保留执行前的原值透传。

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
    <td><a href="./examples/test_geir_update_tensor_desc.cpp">test_geir_update_tensor_desc</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
