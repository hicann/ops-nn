# aclnnFusedPatchMlp

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 接口功能：

  将 `TimeSeriesEmbedding` 的多层 Patch Embedding MLP（多个 `Linear` + `GELU` 重复，最后一层无激活）融合为单个算子，一次下发完成多层前向。

- 计算公式：

  给定输入 `x`（形状 `[..., patch_size]`，支持 2D~8D，前置维度展平为 `N` 行），权重 `weights`、偏置 `biases` 按层展平，层数为 `num_layers`，令 $h_0 = x$：

  对第 `l` 层（`l = 0, 1, ..., num\_layers-1`）：

  $$
  z_l = h_l \cdot W_l^\top + b_l
  $$

  除最后一层外应用 GELU（tanh 近似）：

  $$
  h_{l+1} = 0.5 \cdot z_l \cdot \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \cdot \left( z_l + 0.044715 z_l^3 \right) \right) \right)
  $$

  最后一层不做激活，输出 $y = z_{num\_layers-1}$，形状 `[N, hidden_size]`。

## 函数原型

每个算子分为两段式接口(详见'../../../docs/zh/context/两段式接口.md')，必须先调用“aclnnFusedPatchMlpGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnFusedPatchMlp”接口执行计算。

```Cpp
aclnnStatus aclnnFusedPatchMlpGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weights,
  const aclTensor *biases,
  int64_t          numLayers,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFusedPatchMlp(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnFusedPatchMlpGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1460px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 280px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
    </tr></thead>
   <tbody>
      <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>输入的张量，公式中的 x。</td>
      <td>形状为 [..., patch_size]，最后一维为 patch_size，前置维度展平为 patch 行数 N。</td>
      <td>FLOAT16 / FLOAT / BF16</td>
      <td>ND</td>
      <td>2-8</td>
    </tr>
      <tr>
      <td>weights（aclTensor*）</td>
      <td>输入</td>
      <td>各层权重（转置后）展平拼接。</td>
      <td>首层 [hidden_size, patch_size] 转置，其余层 [hidden_size, hidden_size] 转置。</td>
      <td>FLOAT16 / FLOAT / BF16</td>
      <td>ND</td>
      <td>1</td>
    </tr>
      <tr>
      <td>biases（aclTensor*）</td>
      <td>输入</td>
      <td>各层偏置展平拼接。</td>
      <td>形状为 [num_layers * hidden_size]。当 x 为 BF16 时 biases 需为 FLOAT。</td>
      <td>FLOAT16 / FLOAT / FLOAT</td>
      <td>ND</td>
      <td>1</td>
    </tr>
      <tr>
      <td>numLayers（int64_t）</td>
      <td>输入</td>
      <td>MLP 的 Linear 层数。</td>
      <td>需大于 0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>输出的张量，公式中的 y。</td>
      <td>前置维度与 x 一致，最后一维为 hidden_size，数据类型与输入一致。</td>
      <td>FLOAT16 / FLOAT / BF16</td>
      <td>ND</td>
      <td>2-8</td>
    </tr>
       <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见aclnn返回码(详见'../../../docs/zh/context/aclnn返回码.md')。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的 x、weights、biases 或 out 是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>x、weights、biases 的数据类型不在支持的范围之内。</td>
    </tr>
  </tbody></table>

## aclnnFusedPatchMlp

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedPatchMlpGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见aclnn返回码(详见'../../../docs/zh/context/aclnn返回码.md')。

## 约束说明

- 支持 FLOAT16 / FLOAT(fp32) / BF16 三种数据类型（x/weights/y 同 dtype）；当 x 为 BF16 时 biases 需为 FLOAT（bf16 matmul 的 bias 表为 fp32）。
- `hidden_size = biases_length / num_layers`，`num_layers` 需大于 0。
- `hidden_size` 需使 bias 张量 ≥ 64 字节（FLOAT16/BF16 下 ≥ 32，FLOAT 下 ≥ 16）。
- `patch_size` 需 ≥ 16（满足 Cube 单元 K 方向的分形基本块 `C0=16`）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例(详见'../../../docs/zh/context/编译与运行样例.md')。

完整示例见 test_aclnn_fused_patch_mlp.cpp(详见'../examples/test_aclnn_fused_patch_mlp.cpp')。
