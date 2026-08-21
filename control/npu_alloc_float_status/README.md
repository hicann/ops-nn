# NPUAllocFloatStatus

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

* 算子功能：分配8个float32零值的Tensor，用于NPU溢出状态检测。该算子无输入，输出为固定shape(8,)的全零float32 Tensor，常用于混合精度训练中的loss scaling溢出状态分配场景。

- 计算公式：

  $$
  \text{output} = \text{zeros}(8, \text{dtype}=\text{float32})
  $$

- 示例：

  ```text
  输出data：
  tensor([0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float32)
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
      <td>data</td>
      <td>输出</td>
      <td>NPU浮点状态Tensor，固定shape为(8,)，包含8个float32零值，用于溢出状态检测。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 该算子无输入、无属性，仅有一个输出`data`。
- 输出`data`的shape固定为(8,)，不可变更。8个元素对应NPU硬件的溢出状态标志位（共32字节）。
- 输出`data`的dtype仅支持float32，不支持其他数据类型。
- 输出`data`的format仅支持ND。
- 输出值始终为全零（8个float32零值），不依赖任何输入。
- 该算子为确定性算子，输出固定全零，无并行归约，天然确定性。

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
    <td><a href="./examples/test_geir_npu_alloc_float_status.cpp">test_geir_npu_alloc_float_status</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
