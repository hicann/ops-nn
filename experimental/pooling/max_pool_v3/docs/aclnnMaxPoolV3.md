# aclnnMaxPoolV3

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|√|

## 功能说明

- 接口功能：对输入张量进行 2D 最大池化（Max Pooling 2D）操作。在输入张量的每个空间窗口内取最大值作为输出。

- 目录 `experimental/pooling/max_pool_v3` 的 `v3` 后缀为实现区分标识，对外 ACLNN 接口名称为 `aclnnMaxPoolV3`。

- 计算公式（NCHW 格式，h_dim=2, w_dim=3）：

$$
y[n, c, h_o, w_o] = \max_{i=0}^{kH-1} \max_{j=0}^{kW-1} x[n, c, h_o \cdot sH + i - padT, w_o \cdot sW + j - padL]
$$

其中窗口超出输入边界的元素视为 $-\infty$，不参与最大值计算。

- 输出尺寸（CALCULATED padding 模式）：

$$
H_{out} = \left\lfloor \frac{H_{in} + padT + padB - kH + (ceil\_mode\; ?\; sH - 1 : 0)}{sH} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W_{in} + padL + padR - kW + (ceil\_mode\; ?\; sW - 1 : 0)}{sW} \right\rfloor + 1
$$

当 `ceil_mode = true` 时，如果最后一个窗口起始位置超出 `H_in + padT`，会额外减少一个输出。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMaxPoolV3GetWorkspaceSize”接口获取执行器和 workspace 大小，再调用“aclnnMaxPoolV3”接口执行计算。

```Cpp
aclnnStatus aclnnMaxPoolV3GetWorkspaceSize(
  const aclTensor* x,
  const aclIntArray* ksize,
  const aclIntArray* strides,
  const aclIntArray* pads,
  const aclScalar* ceilMode,
  aclTensor* out,
  uint64_t* workspaceSize,
  aclOpExecutor** executor)
```

```Cpp
aclnnStatus aclnnMaxPoolV3(
  void* workspace,
  uint64_t workspaceSize,
  aclOpExecutor* executor,
  aclrtStream stream)
```

## aclnnMaxPoolV3GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1497px"><colgroup>
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 230px">
  <col style="width: 350px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 120px">
  <col style="width: 100px">
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>4 维输入张量（NCHW 格式）。</td>
      <td><ul><li>支持空Tensor。</li><li>不支持 broadcast。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ksize（aclIntArray*）</td>
      <td>输入</td>
      <td>池化窗口大小，长度为 4，格式 [N, C, H, W]。</td>
      <td><ul><li>不可为空。</li><li>H、W 维度值必须大于 0。</li></ul></td>
      <td>int64_t[]</td>
      <td>-</td>
      <td>[4]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides（aclIntArray*）</td>
      <td>输入</td>
      <td>池化步长，长度为 4，格式 [N, C, H, W]。</td>
      <td><ul><li>不可为空。</li><li>H、W 维度值必须大于 0。</li></ul></td>
      <td>int64_t[]</td>
      <td>-</td>
      <td>[4]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads（aclIntArray*）</td>
      <td>输入</td>
      <td>填充大小，长度为 4，格式 [pad_top, pad_bottom, pad_left, pad_right]。</td>
      <td><ul><li>允许为空（等同于全 0 填充）。</li><li>默认值为 {0, 0, 0, 0}。</li></ul></td>
      <td>int64_t[]</td>
      <td>-</td>
      <td>[4]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode（aclScalar*）</td>
      <td>输入</td>
      <td>ceil 模式开关：非 0 表示使用 ceil 模式计算输出尺寸。</td>
      <td><ul><li>允许为空（等同于 false）。</li><li>默认值为 false(0)。</li></ul></td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>池化后的输出张量。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型必须与 x 一致。</li></ul></td>
      <td>BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>4</td>
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

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持 FLOAT16、FLOAT32、BFLOAT16。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 600px">
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
      <td>传入的 x、ksize、strides 或 out 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>x 或 out 的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>x 和 out 的数据类型不一致。</td>
    </tr>
    <tr>
      <td>x 或 out 的维度大于 8。</td>
    </tr>
  </tbody></table>

## aclnnMaxPoolV3

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
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnMaxPoolV3GetWorkspaceSize 获取。</td>
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
- 输入张量必须为 4 维（NCHW 格式）。
- `ksize` 和 `strides` 的 H、W 维度值必须大于 0。
- BFLOAT16 仅在 Ascend910B 及后续同代 SoC 上支持。
- 支持非连续 Tensor，内部会执行连续化和 `ViewCopy`。
