# aclnnApplyGradientDescent

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    ×     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term>    |    √     |

## 功能说明

- 接口功能：梯度下降单步参数更新，对 `var` 做原地（inplace）更新。
- 计算公式：

$$
var = var - alpha \times delta
$$

  中间计算在 fp32 精度下完成（fp16/bf16 先 Cast 到 fp32 再 Cast 回原类型）。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用
“aclnnApplyGradientDescentGetWorkspaceSize”接口获取计算所需 workspace 大小以及包含了算子计算流程的
执行器，再调用“aclnnApplyGradientDescent”接口执行计算。

```Cpp
aclnnStatus aclnnApplyGradientDescentGetWorkspaceSize(
  aclTensor*       var,
  const aclTensor* alpha,
  const aclTensor* delta,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnApplyGradientDescent(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnApplyGradientDescentGetWorkspaceSize

- **参数说明：**

  <table>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>var（aclTensor*）</td>
      <td>输入&输出</td>
      <td>待更新的参数，公式中的 var；inplace 语义，既是输入也是输出。</td>
      <td>shape 需要与 delta 一致；维度 1~8。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
    </tr>
    <tr>
      <td>alpha（aclTensor*）</td>
      <td>输入</td>
      <td>标量学习率张量，公式中的 alpha。</td>
      <td>必须是 1 个元素的标量 Tensor；dtype 与 var 一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>delta（aclTensor*）</td>
      <td>输入</td>
      <td>梯度，公式中的 delta。</td>
      <td>shape 与 dtype 均需与 var 一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table>
  <thead>
    <tr><th>返回码</th><th>错误码</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的 var、alpha、delta 是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>var 的数据类型不在支持的范围之内。</td>
    </tr>
    <tr><td>alpha、delta 的数据类型与 var 不一致。</td></tr>
    <tr><td>var 与 delta 的 shape 不一致。</td></tr>
    <tr><td>alpha 不是 1 个元素的标量 Tensor。</td></tr>
  </tbody>
  </table>

## aclnnApplyGradientDescent

- **参数说明：**

  <table>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在 Device 侧申请的 workspace 内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在 Device 侧申请的 workspace 大小，由第一段接口获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op 执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的 Stream。</td></tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- `var.shape == delta.shape`，`alpha` 为 1 元素标量 Tensor。
- `alpha.dtype == var.dtype == delta.dtype`，且仅支持 BFLOAT16/FLOAT16/FLOAT。

## 性能说明

本算子为逐元素、访存受限（MTE2 GM 读）型算子：每元素读 `var`+`delta`、写 `var`。在
<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term>（`ascend910b`）上，大 shape 的
float32 已接近 HBM 读带宽上限（读流量不可约减），性能主要由访存带宽决定；fp16/bf16 及中小
shape 因实现中采用 ≥512B 对齐的读突发与 `Axpy` 融合计算而略有收益。

## 调用示例

示例代码见 [`../examples/test_aclnn_apply_gradient_descent.cpp`](../examples/test_aclnn_apply_gradient_descent.cpp)。
