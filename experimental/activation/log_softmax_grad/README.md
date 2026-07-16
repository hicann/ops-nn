# LogSoftmaxGrad

## 产品支持情况

| 产品                         | 是否支持 |
| ---------------------------- | -------- |
| Atlas A2 训练系列产品       | √        |
| Atlas 800I A2 推理产品     | √        |
| Atlas 200I/500 A2 推理产品 | ✗        |

## 功能说明

- **接口功能**：完成 LogSoftmax 的反向传播，计算 LogSoftmax 正向输入的梯度（gradInput）。
- **计算公式**：

  其中 $output$ 为 LogSoftmax 正向输出（即 $\text{log\_softmax}(x)$），$gradOutput$ 为上游梯度：

  $$
  gradInput = gradOutput - \exp(output) \cdot \text{reduce\_sum}(gradOutput, dim)
  $$

  由于 $\exp(output)$ 即为 softmax 概率分布，上述计算等价于：

  $$
  gradInput = gradOutput - softmax(x) \cdot \text{reduce\_sum}(gradOutput, dim)
  $$

- **支持的输入/输出数据类型**：FLOAT（float32）、FLOAT16（float16）、BFLOAT16（bfloat16），且 gradOutput、output 与 out 三者的数据类型与 shape 必须一致。

## 参数说明

| 参数名     | 输入/输出 | 数据类型                  | 数据格式 | 维度(shape) | 描述                                                                   |
| ---------- | --------- | ------------------------- | -------- | ----------- | ---------------------------------------------------------------------- |
| gradOutput | 输入      | FLOAT、FLOAT16、BFLOAT16 | ND       | 0-8         | LogSoftmax 正向输出的梯度，公式中的 gradOutput，支持空 Tensor。         |
| output     | 输入      | FLOAT、FLOAT16、BFLOAT16 | ND       | 0-8         | LogSoftmax 正向输出，公式中的 output，支持空 Tensor。                   |
| dim        | 输入      | INT64                     | -        | -           | 指定进行 reduce 的维度，取值范围为 $[-dimNum, dimNum-1]$。             |
| out        | 输出      | FLOAT、FLOAT16、BFLOAT16 | ND       | 0-8         | 反向计算的输出，即 LogSoftmax 正向输入的梯度 gradInput，支持空 Tensor。 |

## 调用说明

本算子提供两段式 aclnn 接口 `aclnnLogSoftmaxBackwardGetWorkspaceSize` 与 `aclnnLogSoftmaxBackward`，完整可编译的调用示例见 [test_aclnn_log_softmax_grad.cpp](examples/test_aclnn_log_softmax_grad.cpp)。

```cpp
// 第一段：获取 workspace 大小与执行器
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus ret = aclnnLogSoftmaxBackwardGetWorkspaceSize(gradOutput, output, dim, out,
                                                          &workspaceSize, &executor);
// 申请 workspace 并执行
void* workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
ret = aclnnLogSoftmaxBackward(workspace, workspaceSize, executor, stream);
```

## 约束说明

- 输入 gradOutput、output 与输出 out 的 shape 必须一致，不支持广播。
- 指定维度 dim 需在 gradOutput 的维度范围内，范围 $[-dimNum, dimNum-1]$。
- 支持的数据类型为 FLOAT、FLOAT16、BFLOAT16；当硬件不支持 bfloat16 向 float32 的 Cast 时，使用 BFLOAT16 会报错。
- 输入 tensor 的维度数不能超过 8；当 dim 为多个轴时，reduce 轴必须连续（如 [2,3,4] 合法，[2,4] 不合法）。

## 详细设计

### host 侧设计

**tiling 策略：**

在 tiling 函数中会做参数校验，保证输入输出 shape 完全相同，并且 axis 里面的 reduce 轴都是连续的。然后根据输入属性 axis 对当前 shape 进行合轴操作，这样任意 shape 合轴完成后，就只剩三个轴，形状如下：

`[mergedDim0, mergedDim1, mergedDim2]`

- 如果 axis 中指定的**最小** reduce 轴是 0，说明 reduce 轴之前没有维度，那么 `mergedDim0 = 1`；
- 如果 axis 中指定的**最大** reduce 轴是最后一个轴（即 `dimNum-1`），说明 reduce 轴之后没有维度，那么 `mergedDim2 = 1`。

通过平台信息动态获取 UbSize、CoreNum 与 UB block 大小（不再硬编码 BLOCK_SIZE），以保证不同芯片平台的兼容性。

**分核策略：**

1. 根据 mergedDim0 进行分核，优先使用满核原则；如果核间能均分，可视作无大小核区分，大核小核数据块一致；如果核间不能均分，需要将余出的数据块分配到前几个核上；如果 `mergedDim0 < CoreNum`，则设置实际使用的核数为 `mergedDim0`。
2. 数据分块和内存优化策略：充分使用 UB 空间。算子总共是两个输入和一个输出，input_dy 需要进行累加和 reduce 操作，因此给它开启 double buffer，分配 2 块 UB 空间，给 reduce 操作留一块空间，input_x 和 output_z 各 1 块空间，总共 5 块，将 UB 均分为 5 份，并对齐 BLOCK_SIZE，计算出每个 UB 块的数据数量。对于 bfloat16 和 half 类型，需要在计算之前做 cast 操作，这里将搬入的 UB 空间和 Cast 结果的 UB 空间复用：搬入时将数据放到 UB 块的后半部分，再 cast 到整个 UB 块。
3. tilingkey 规划策略：
   - 如果 `mergedDim1` 值为 1，即不需要进行 reduce 操作，tilingkey 为 0；
   - 其他情况，tilingkey 为 1。

**数据检测：**

对不支持 `AscendC::Cast()` bfloat16 向 float32 转换的硬件（即非 Atlas A2 系列），在参数校验阶段直接失败，并打印报错提示。

### kernel 侧设计

进行 Init 和 Process 两个阶段，其中 Process 包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. 根据不同的 tilingkey 实例化不同的核函数对象。
2. 根据 golden 实现，所有数据都需要 cast 为 float32 计算；如果当前数据类型不是 float32，则搬入数据后就进行 cast 操作，并且复用空间。
3. 根据 mergedDim0 计算当前核需要处理哪些数据，计算出 start 和 end，开启第一层循环。
4. 根据 mergedDim1 和 dim2Tile 确定第二层循环的循环次数。
5. tilingKey0（不需要 reduce）的流程如下，公式化简得：`output_z = (input_dy - 1) * exp(input_x)`。
6. tilingKey1（需要 reduce）的流程按 REDUCE_TAIL / REDUCE_MID 两种模式分别处理。

## 贡献说明

| 贡献者      | 贡献方     | 贡献算子     | 贡献时间 | 贡献内容                 |
| ----------- | ---------- | ------------ | -------- | ------------------------ |
| DreamOfAnts | 个人开发者 | LogSoftmaxGrad | 2026     | LogSoftmaxGrad 算子适配开源仓 |
