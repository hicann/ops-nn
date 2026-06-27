# aclnnMishBackward

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|√|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|√|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|√|
|<term>Atlas 训练系列产品</term>|√|

## 功能说明

- 接口功能：计算 Mish 的反向梯度。

- 目录 `experimental/activation/mish_grad_v2` 仅保留实现区分用的 `v2` 后缀，对外 ACLNN 接口名称仍为 `aclnnMishBackward`。

- 计算公式：

$$
xGrad = grad * \left(tanhx + x * \frac{1 - tanhx^2}{1 + e^{-x}}\right)
$$

其中：

$$
tanhx = tanh(softplus(x)),\ softplus(x) = relu(x) + log(1 + e^{-|x|})
$$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMishBackwardGetWorkspaceSize”接口获取执行器和 workspace 大小，再调用“aclnnMishBackward”接口执行计算。

```Cpp
aclnnStatus aclnnMishBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclTensor* self,
  aclTensor* gradInput,
  uint64_t* workspaceSize,
  aclOpExecutor** executor)
```

```Cpp
aclnnStatus aclnnMishBackward(
  void* workspace,
  uint64_t workspaceSize,
  aclOpExecutor* executor,
  aclrtStream stream)
```

## aclnnMishBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1497px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 247px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
  <col style="width: 145px">
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
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gradOutput（aclTensor*）</td>
      <td>输入</td>
      <td>上游反向传播输入梯度。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 self 和 gradInput 完全一致。</li><li>数据类型必须与 self 和 gradInput 完全一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>Mish 正向输入。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 gradOutput 和 gradInput 完全一致。</li><li>数据类型必须与 gradOutput 和 gradInput 完全一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInput（aclTensor*）</td>
      <td>输出</td>
      <td>Mish 反向梯度计算结果。</td>
      <td><ul><li>支持空Tensor。</li><li>shape 必须与 gradOutput 和 self 完全一致。</li><li>数据类型必须与 gradOutput 和 self 完全一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas 训练系列产品</term>：数据类型支持 FLOAT16、FLOAT32。
  - <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：额外支持 BFLOAT16。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

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
      <td>传入的 grad、x 或 xGrad 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>grad、x 或 xGrad 的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>grad、x 和 xGrad 的数据类型不一致。</td>
    </tr>
    <tr>
      <td>grad、x 和 xGrad 的 shape 不一致。</td>
    </tr>
    <tr>
      <td>grad、x 或 xGrad 的维度大于 8。</td>
    </tr>
    <tr>
      <td>当前平台不支持 BFLOAT16。</td>
    </tr>
  </tbody></table>

## aclnnMishBackward

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnMishBackwardGetWorkspaceSize 获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 默认确定性实现。
- 不支持 broadcast。
- 不支持隐式类型提升。
- 支持非连续 Tensor，内部会执行连续化和 `ViewCopy`。

## 调用示例

示例代码请参考 [test_aclnn_mish_grad_v2.cpp](../examples/test_aclnn_mish_grad_v2.cpp)。
