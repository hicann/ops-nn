# ActsUlq

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：ActsUlq（Activations Universal Linear Quantization）实现均匀线性量化模拟（Fake Quantization），用于量化感知训练（QAT）场景。该算子模拟INT8量化过程的前向传播，输出量化后的近似值以及量化损失信息，使模型在训练阶段学习适应量化误差。

- 计算公式：

  **1. 计算量化范围**

  $$
  ori\_clip\_min = \begin{cases} 0 & \text{if } fixedMin \\ \min(clampMin, 0) & \text{otherwise} \end{cases}
  $$

  $$
  ori\_clip\_max = \max(clampMax, 255 \times 1.192092896 \times 10^{-7})
  $$

  **2. 计算scale和offset**

  $$
  scale = \frac{ori\_clip\_max - ori\_clip\_min}{255}
  $$

  $$
  offset = round\left(\frac{ori\_clip\_min}{scale}\right)
  $$

  **3. 计算fake quant的clip边界**

  $$
  clip\_min = scale \times offset
  $$

  $$
  clip\_max = scale \times (offset + 255)
  $$

  **4. 对x做clamp**

  $$
  clamped\_x = \min(\max(x, clip\_min), clip\_max)
  $$

  **5. 生成mask**

  $$
  clampMinMask = \begin{cases} 1.0 & \text{if } x \geq clip\_min \\ 0.0 & \text{otherwise} \end{cases}
  $$

  $$
  clampMaxMask = \begin{cases} 1.0 & \text{if } x \leq clip\_max \\ 0.0 & \text{otherwise} \end{cases}
  $$

  **6. 量化**

  $$
  raw\_x = \frac{clamped\_x}{scale}
  $$

  $$
  round\_x = round(raw\_x)
  $$

  **7. 计算clamped_loss**

  $$
  clamped\_loss = \frac{round\_x - raw\_x}{255}
  $$

  **8. 计算loss_m**

  $$
  raw\_m = \frac{ori\_clip\_min}{scale}
  $$

  $$
  round\_m = round(raw\_m)
  $$

  $$
  loss\_m = \frac{round\_m - raw\_m}{255}
  $$

  **9. 根据mask选择loss**

  $$
  xClampedLoss = \begin{cases} clamped\_loss & \text{if } clampMinMask = 1.0 \\ loss\_m & \text{otherwise} \end{cases}
  $$

  $$
  xClampedLoss = \begin{cases} xClampedLoss & \text{if } clampMaxMask = 1.0 \\ loss\_m & \text{otherwise} \end{cases}
  $$

  **10. 计算y**

  $$
  y = round\_x \times scale
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| x | 输入 | 表示输入数据张量，对应公式中x；支持空Tensor；shape需要与clamp_min、clamp_max满足broadcast关系。 | FLOAT、FLOAT16 | ND |
| clamp_min | 输入 | 表示量化下界张量，对应公式中clampMin；支持空Tensor；数据类型必须与x一致；shape需要与x、clamp_max满足broadcast关系。 | FLOAT、FLOAT16 | ND |
| clamp_max | 输入 | 表示量化上界张量，对应公式中clampMax；支持空Tensor；数据类型必须与x一致；shape需要与x、clamp_min满足broadcast关系。 | FLOAT、FLOAT16 | ND |
| fixed_min | 可选属性 | 是否固定下界为0；true时ori_clip_min=0，false时ori_clip_min=min(clampMin, 0)；缺省值为false。 | BOOL | - |
| num_bits | 可选属性 | 量化位宽；当前仅支持8；缺省值为8。 | INT | - |
| y | 输出 | 量化输出张量，对应公式中y；不支持空Tensor；数据类型必须与x一致；shape必须是x、clamp_min、clamp_max broadcast后的结果shape。 | FLOAT、FLOAT16 | ND |
| clamp_min_mask | 输出 | 下界掩码张量，值为1.0（x >= clip_min）或0.0；shape必须与y一致。 | FLOAT、FLOAT16 | ND |
| clamp_max_mask | 输出 | 上界掩码张量，值为1.0（x <= clip_max）或0.0；shape必须与y一致。 | FLOAT、FLOAT16 | ND |
| x_clamped_loss | 输出 | 量化损失张量，对应公式中xClampedLoss；shape必须与y一致。 | FLOAT、FLOAT16 | ND |

## 约束说明

- x、clamp_min、clamp_max的数据类型必须一致，仅支持FLOAT和FLOAT16。
- x、clamp_min、clamp_max的shape需要满足broadcast关系，shape维度支持0-8维。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| 图模式 | [test_geir_acts_ulq](./examples/test_geir_acts_ulq.cpp) | 通过[算子IR](./op_graph/acts_ulq_proto.h)构图方式调用ActsULQ算子。 |
