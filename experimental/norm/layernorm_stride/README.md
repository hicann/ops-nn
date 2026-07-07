# 算子名称：LayerNormStride

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A2 训练系列产品                                         | 是       |

## 功能说明

- 算子功能：MLA中带跨步的layernorm。
- 计算公式：
  $$
  y = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} + \beta
  $$
- 在基本layernorm的基础上，追加了跨步功能, 输入是shape为[Batch, Seqlen, Stride]的张量，Stride为跨步大小，计算Stride的前kv_lora_rank维的归一化。

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>blockDim</td>
      <td>输入</td>
      <td>AI CORE的数量，比如：Ascend910B是40。</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  <tr>
      <td>gbW</td>
      <td>输入</td>
      <td>MLA中的kv_lora_rank</td>
      <td>int64_t</td>
      <td>-</td>
    </tr>
  <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>layernorm计算的eps</td>
      <td>float</td>
      <td>-</td>
    </tr>
  <tr>
      <td>in</td>
      <td>输入</td>
      <td>带跨步layernorm计算的输入，shape为 (Batch, Seqlen, Stride)，Stride大小为kv_lora_rank + qk_rope_head_dim</td>
      <td>BFLOAT16/HALF</td>
      <td>ND</td>
    </tr>
  <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>layernorm中的weight</td>
      <td>BFLOAT16/HALF</td>
      <td>ND</td>
    </tr>
  <tr>
      <td>beta</td>
      <td>输入</td>
      <td>layernorm中的bias</td>
      <td>BFLOAT16/HALF</td>
      <td>ND</td>
    </tr>
  <tr>
      <td>out</td>
      <td>输出</td>
      <td>等同输入</td>
      <td>BFLOAT16/HALF</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入输出仅支持BFLOAT16/HAFL类型。

## 调用说明

```text
torch.ops.ascend_ops.layernorm_stride(blockDim, gbW, epsilon, in, gamma, beta, out)
```
