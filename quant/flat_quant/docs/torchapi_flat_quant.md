# flat_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

  为矩阵x依次进行两次克罗内克积小矩阵乘法，然后针对矩阵乘的结果进行量化处理。底层封装 `aclnnFlatQuantV3`。

- 计算公式：

  $$
  \text{intermediate} = x \cdot \text{kronecker\_p2}
  $$

  $$
  \text{result} = \text{kronecker\_p1} \cdot \text{intermediate}
  $$

  $$
  \text{scale} = \text{compute\_quant\_scale}(\text{result}, \text{clip\_ratio}, \text{dst\_type\_max})
  $$

  $$
  \text{out} = \text{quantize}(\text{result}, \text{scale})
  $$

## 函数原型

```python
cann_ops_nn.flat_quant(x, kronecker_p1, kronecker_p2, clip_ratio=1.0, dst_dtype=torch.quint4x2, dst_type_max=0.0, group_list=None, group_list_type=0)
    -> (Tensor out, Tensor quant_scale)
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>输入矩阵，进行克罗内克积乘法和量化处理的数据张量。N2必须是8的整数倍（dst_dtype = torch.quint4x2时）或偶数（dst_dtype = torch_npu.float4_e2m1fn_x2时）。</td>
        <td>float16、bfloat16</td>
        <td>(M, N1, N2)</td>
    </tr>
    <tr>
        <td>kronecker_p1</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>第一个克罗内克积矩阵，N1维与x中N1维一致。</td>
        <td>float16、bfloat16</td>
        <td>(N1, N1)</td>
    </tr>
    <tr>
        <td>kronecker_p2</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>第二个克罗内克积矩阵，N2维与x中N2维一致。</td>
        <td>float16、bfloat16</td>
        <td>(N2, N2)</td>
    </tr>
    <tr>
        <td>clip_ratio</td>
        <td>float</td>
        <td>可选</td>
        <td>量化裁剪比例，用于控制量化的裁剪比例，范围(0, 1]。默认值1.0。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>指定输出数据类型。支持torch.quint4x2（默认）、torch_npu.float4_e2m1fn_x2。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type_max</td>
        <td>float</td>
        <td>可选</td>
        <td>目标数据类型的最大值，输入值只能是0或[6, 12]范围内的数。默认值0.0。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group_list</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>分组列表，用于分组量化。支持1D或2D输入。默认值None。</td>
        <td>int64</td>
        <td>(G,) 或 (G, 2)</td>
    </tr>
    <tr>
        <td>group_list_type</td>
        <td>int</td>
        <td>可选</td>
        <td>group_list输入的分组方式，取值范围0-2。默认值0。</td>
        <td>-</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>输出名</th>
    <th>输出类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>out</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>量化输出。dst_dtype = torch.quint4x2时数据类型为int32，shape为(M, N1, N2//8)，dst_dtype = torch_npu.float4_e2m1fn_x2时数据类型为uint8，shape为(M, N1 * N2//2)。</td>
        <td>int32、uint8</td>
        <td>(M, N1, N2//8) 或 (M, N1 * N2//2)</td>
    </tr>
    <tr>
        <td>quant_scale</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>量化缩放因子。dst_dtype = torch.quint4x2时数据类型为float32，shape为(M,)，dst_dtype = torch_npu.float4_e2m1fn_x2时数据类型为uint8，shape为(M, ceildiv(N1*N2,64), 2)。</td>
        <td>float32、uint8</td>
        <td>(M,) 或 (M, ceildiv(N1*N2,64), 2)</td>
    </tr>
</tbody>
</table>

## 约束说明

- 输入x的N2维度必须满足以下条件：
  - dst_dtype = torch.quint4x2时，N2必须是8的整数倍。
  - dst_dtype = torch_npu.float4_e2m1fn_x2时，N2必须是偶数。
- 输入kronecker_p1和kronecker_p2的数据类型必须与x一致。
- clip_ratio范围为(0, 1]。
- dst_dtype支持torch.quint4x2（默认）、torch_npu.float4_e2m1fn_x2，输出说明如下：
  - 如果dtype为torch.quint4x2时，输出out类型为int32，由8个int4拼接，查看具体值需自行解包，输出quant_scale类型为float32。
  - 如果dtype为torch_npu.float4_e2m1fn_x2时，输出out类型为uint8，由2个float4_e2m1fn_x2拼接，查看具体值需自行解包，输出quant_scale类型为uint8，查看实际值需自行转换成float8_e8m0fnu。
- dst_type_max只能为0或[6, 12]范围内的数。
- group_list_type取值范围为0-2，当group_list_type为0或1时，group_list的shape为(G,)，当group_list_type为2时，group_list的shape为(G, 2)，G表示分组数，G需要小于等于1024。
- group_list需要满足以下条件：
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - group_list仅支持None输入。
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - group_list不为None时，group_list输入的数值需要满足以下条件，否则无法保证输出是否符合预期：
      - 当group_list_type为0时，group_list必须为非负单调非递减数列，表示分组后每组大小的cumsum结果（累计和），最后一个值应小于等于x中tensor的第一维。
      - 当group_list_type为1时，group_list必须为非负数列，表示分组后每组大小，数值的总和应小于等于x中tensor的第一维。
      - 当group_list_type为2时，group_list必须为非负数列，数据排布为[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]，其中groupSize为分组后每组大小，第二列数值的总和应小于等于x中tensor的第一维。
  <!-- end id8 -->
- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用（eager）

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, N1, N2 = 128, 16, 16
  x = torch.randn(M, N1, N2, dtype=torch.float16).npu()
  kronecker_p1 = torch.randn(N1, N1, dtype=torch.float16).npu()
  kronecker_p2 = torch.randn(N2, N2, dtype=torch.float16).npu()

  # int32 输出（默认）
  out, quant_scale = cann_ops_nn.flat_quant(x, kronecker_p1, kronecker_p2)
  print("out: ", out)
  print("quant_scale: ", quant_scale)
  ```

  带可选输入：

  ```python
  group_list = torch.tensor([0, 16, 32, 48, 64], dtype=torch.int64).npu()
  out, quant_scale = cann_ops_nn.flat_quant(
      x, kronecker_p1, kronecker_p2,
      group_list=group_list,
      clip_ratio=0.95,
      group_list_type=0
  )
  ```

- 图模式（torchair）调用

  ```python
  import torch, torch_npu, cann_ops_nn

  @torch.compile(backend="npu")
  def func(x, p1, p2):
      return cann_ops_nn.flat_quant(x, p1, p2)

  M, N1, N2 = 128, 16, 16
  x = torch.randn(M, N1, N2, dtype=torch.float16).npu()
  kronecker_p1 = torch.randn(N1, N1, dtype=torch.float16).npu()
  kronecker_p2 = torch.randn(N2, N2, dtype=torch.float16).npu()
  out, quant_scale = func(x, kronecker_p1, kronecker_p2)
  print("out: ", out)
  print("quant_scale: ", quant_scale)
  ```
