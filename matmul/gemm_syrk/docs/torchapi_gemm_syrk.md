# gemm_syrk

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

  实现对称秩k更新（syrk，参考 cublas `syrk`）计算。原地更新对称矩阵 C：底层封装
  `aclnnGemmSyrk`，将 c 同时绑定为算子的输入与输出（同一 device buffer），计算结果
  完整写回整个对称矩阵（上三角和下三角均写出）。transpose_x 为 True 时 a 以转置的
  (k, m) 布局存储，对应 cublas syrk 的 OP_T 语义。

- 计算公式（transpose_x = False，a 为 (…, m, k)，2-6 维，前面为 batch 轴）：

  <div>
  C = α × (A @ A<sup>T</sup>) + β × C
  </div>

  transpose_x = True 时，a 为转置的 (…, k, m) 存储：

  <div>
  C = α × (A<sup>T</sup> @ A) + β × C
  </div>

## 函数原型

```python
cann_ops_nn.gemm_syrk(a, c, *, alpha=None, beta=None, transpose_x=False, fill_mode="full")
    -> Tensor
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
        <td>a</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>输入矩阵A，对应公式中的A。transpose_x为True时为转置的(k, m)存储，m为a的最后一维。数据类型必须与c一致。支持非连续Tensor。</td>
        <td>float16、bfloat16</td>
        <td>(…, m, k)，2-6 维，前面为 batch 轴；transpose_x=True时为(…, k, m)</td>
    </tr>
    <tr>
        <td>c</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>对称矩阵C，对应公式中的C。原地更新：计算结果直接写回该tensor的内存。最后两维相等且等于a的m轴，batch轴与a一致（不支持广播）。支持非连续Tensor。</td>
        <td>float16、bfloat16</td>
        <td>(…, m, m)（与 a 的 batch 轴一致）</td>
    </tr>
    <tr>
        <td>alpha</td>
        <td>Scalar</td>
        <td>可选</td>
        <td>矩阵乘结果的缩放系数，对应公式中的α。为None时默认1.0。</td>
        <td>float</td>
        <td>-</td>
    </tr>
    <tr>
        <td>beta</td>
        <td>Scalar</td>
        <td>可选</td>
        <td>C的缩放系数，对应公式中的β。为None时默认1.0。</td>
        <td>float</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transpose_x</td>
        <td>bool</td>
        <td>可选</td>
        <td>是否按转置布局解读a。为True时a为(k, m)存储，计算C = alpha * (A^T @ A) + beta * C。默认False。</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>fill_mode</td>
        <td>str</td>
        <td>可选</td>
        <td>输出区域模式："full"（完整对称矩阵）/"up"（上三角）/"low"（下三角）。当前仅支持"full"。默认"full"。</td>
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
        <td>c</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>原地更新后的对称矩阵C（即输入的c本身，同一tensor对象）。计算结果完整写回整个对称矩阵。</td>
        <td>float16、bfloat16</td>
        <td>(…, m, m)（与 a 的 batch 轴一致）</td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持推理场景下使用。
- a与c的数据类型必须一致（float16或bfloat16），数据格式仅支持ND，维度为2~6维。
- c必须为方阵且m轴与a一致（transpose_x为True时m为a的最后一维），batch轴与a一致（原地更新不支持广播）。
- k轴为0或alpha为0时，接口自动路由为逐元素计算C = beta * C，不进入matmul计算路径。
- fill_mode当前仅支持"full"，"up"/"low"为预留值尚未实现。
- 建议输入c为对称矩阵：beta不为0时输出矩阵的对称性依赖输入c对称。
- 支持非连续Tensor，无需额外做contiguous。
- m轴或batch轴乘积为0时，接口直接返回成功，不启动kernel。
- m、k、n各维度及多维batch乘积的取值范围为(0, 2147483647)。

## 确定性计算

默认支持确定性计算（每个输出tile由单条Mmad链按固定顺序累加，无原子操作与切K归约）。

## 调用示例

- 单算子模式调用（eager）

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  m, k = 128, 64
  a = torch.randn(m, k, dtype=torch.float16).npu()
  c = torch.randn(m, m, dtype=torch.float16).npu()
  c = 0.5 * (c + c.t())  # 对称化输入

  alpha, beta = 1.0, 1.0
  # 原地更新：返回值即输入的c本身
  c = cann_ops_nn.gemm_syrk(a, c, alpha=alpha, beta=beta)
  print("result: ", c)
  ```

- 转置输入调用（transpose_x）

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  m, k = 128, 64
  a_t = torch.randn(k, m, dtype=torch.float16).npu()  # 转置的(k, m)存储
  c = torch.randn(m, m, dtype=torch.float16).npu()
  c = 0.5 * (c + c.t())

  # C = alpha * (A^T @ A) + beta * C，A为a_t所代表的转置存储
  c = cann_ops_nn.gemm_syrk(a_t, c, transpose_x=True)
  print("result: ", c)
  ```
