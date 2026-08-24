# SwigluGroupGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- **算子功能**：完成ClampedSwiglu激活函数的反向梯度计算。从上游梯度`grad_y`和前向输入`x`重算clamp mask与sigmoid，输出`grad_x`与可选`grad_weight`。
- **计算公式**：

  前向分解：x按hidden维劈半得到gate(g)和up(u)；可选clamp产生g̃ = min(c, g)、ũ = clip(u, −c, c)；SiLU(g̃) = g̃·σ(g̃)；y = SiLU(g̃)·ũ·w_t。

  $$silu'(g̃) = s + f − f·s$$

  $$dg = grad\_y \cdot silu'(g̃) \cdot ũ \cdot w_t \cdot I(g < c) \cdot m_r$$

  $$du = grad\_y \cdot f \cdot w_t \cdot I(−c < u < c) \cdot m_r$$

  $$grad\_weight = \Sigma(grad\_y \cdot y\_origin) \text{ along hidden dim}$$

  其中I为开区间指示函数（边界值时mask=0），m_r为group_index mask，w_t为weight的broadcast。

  **约束**：weight和y_origin必须同时提供或同时为空；成对提供时计算grad_weight。

  grad_x拼回：gradX[..., :H] = dg，gradX[..., H:] = du。

- **关键特性**：支持MoE场景的group_index动态分组和weight权重梯度计算。

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>grad_y</td>
      <td>输入</td>
      <td>上游梯度，shape (..., H)，常用 shape 为 (T, H) 或 (B, S, H)。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>前向输入，shape (..., 2H)，常用 shape 为 (T, 2H) 或 (B, S, 2H)，包含 gate 和 up 分支。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td>MoE top-k 路由权重，元素个数等于 grad_y 非尾轴元素个数，dtype FP32。缺省视作全1。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_origin</td>
      <td>可选输入</td>
      <td>前向输出 y，最后一维为 H，非尾轴元素个数等于 grad_y 非尾轴元素个数，dtype 同 grad_y；weight 存在时 y 已乘该权重。weight 提供时必须同时提供。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td>各分组 token/batch 数量索引，shape (G,)，G > 0，dtype INT64。缺省视作全部行有效。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>属性</td>
      <td>截断门限标量 c；取值范围为 -1.0 或 >0.0，-1.0 表示不启用 clamp，启用时必须 >0.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>输出</td>
      <td>x 的梯度，shape 与 x 相同，dtype 同 grad_y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_weight</td>
      <td>可选输出</td>
      <td>weight 的梯度，shape 与 weight 相同，dtype FP32。仅 weight 和 y_origin 同时提供时计算。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- H > 0
- x.shape[-1] = 2 × H（grad_y.shape[-1]）
- grad_y 与 x 的尾轴满足 x.shape[-1] == 2 * grad_y.shape[-1]，非尾轴元素总数相等
- weight 和 y_origin 必须同时提供才能计算 grad_weight
- clamp_limit 默认值 -1.0，传入 -1.0 表示不进行截断，启用截断时必须传入大于 0 的值
- group_index 非空时必须是一维非空 Tensor（G > 0）
- group_index 缺省时所有前导维度展平后的行均为有效行

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_swiglu_group_grad](./examples/test_aclnn_swiglu_group_grad.cpp) | 通过[aclnnSwigluGroupGrad](./docs/aclnnSwigluGroupGrad.md)调用SwigluGroupGrad算子 |
| 图模式 | [test_geir_swiglu_group_grad](./examples/test_geir_swiglu_group_grad.cpp) | 通过[算子IR](./op_graph/swiglu_group_grad_proto.h)调用SwigluGroupGrad算子 |
| torch接口 | [test_torch_swiglu_group_grad](./examples/test_torch_swiglu_group_grad.py) | 通过`torch.ops.cann_ops_nn.swiglu_group_backward`调用SwigluGroupGrad算子 |
