# ApplyAdagradD

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | |
| <term>Atlas 推理系列产品</term> | |
| <term>Atlas 训练系列产品</term> | |

## 功能说明

- 算子功能：ApplyAdagradD 是 Adagrad 优化器的参数更新算子，用于根据梯度、学习率和累加平方梯度更新参数。
- 计算公式：

  $$
  accum = accum + grad * grad
  $$

  $$
  var = var - lr * grad * (1 / sqrt(accum))
  $$

- 当 `update_slots` 为 `false` 时，`accum` 不累加 `grad * grad`，仅使用输入 `accum` 更新 `var`。

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
      <td>var</td>
      <td>输入</td>
      <td>待更新的参数张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入</td>
      <td>累加的平方梯度张量。与"var"具有相同类型和shape。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率张量。shape为{1}，与"var"具有相同类型。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>梯度张量。与"var"具有相同类型和shape。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>更新后的参数张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输出</td>
      <td>更新后的累加平方梯度张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update_slots</td>
      <td>属性</td>
      <td>是否更新"accum"，默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>是否使用锁，当前无特殊并发控制实现，默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|:--------|:---------|:-----|
| 图模式 | [test_geir_apply_adagrad_d.cpp](examples/test_geir_apply_adagrad_d.cpp) | 通过[算子IR](op_graph/apply_adagrad_d_proto.h)构图调用ApplyAdagradD算子。 |
| aclnn | [test_aclnn_apply_adagrad_d.cpp](examples/test_aclnn_apply_adagrad_d.cpp) | 通过aclnn接口调用ApplyAdagradD算子。 |

### aclnn接口

```cpp
aclnnStatus aclnnApplyAdagradDGetWorkspaceSize(
    const aclTensor* var,
    const aclTensor* accum,
    const aclTensor* lr,
    const aclTensor* grad,
    bool updateSlots,
    bool useLocking,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

aclnnStatus aclnnApplyAdagradD(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream);
```

## 约束与限制

- `var`、`accum`、`grad` 的shape和数据类型必须一致。
- `lr` 必须为标量张量，shape为{1}。
- 仅支持ND数据格式。
- 空tensor支持no-op执行。
- `use_locking` 当前无特殊并发控制实现。
- FLOAT16、BFLOAT16输入会提升至FLOAT计算后再转回原类型。

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|:------|:------|:--------|:--------|:--------|
| Tream | 个人开发者 | ApplyAdagradD | 2026/05/29 | ApplyAdagradD算子适配开源仓 |
