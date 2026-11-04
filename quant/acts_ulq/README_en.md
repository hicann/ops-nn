# ActsUlq

English | [简体中文](./README.md)


## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function

- Operator function: ActsUlq (Activations Universal Linear Quantization) implements uniform linear quantization simulation (Fake Quantization) for quantization-aware training (QAT) scenarios. The operator simulates the forward pass of the INT8 quantization process and outputs the quantized approximate values and quantization loss information, so that the model learns to adapt to quantization errors during training.

- Formula:

  **1. Compute the quantization range**

  $$
  ori\_clip\_min = \begin{cases} 0 & \text{if } fixedMin \\ \min(clampMin, 0) & \text{otherwise} \end{cases}
  $$

  $$
  ori\_clip\_max = \max(clampMax, 255 \times 1.192092896 \times 10^{-7})
  $$

  **2. Compute scale and offset**

  $$
  scale = \frac{ori\_clip\_max - ori\_clip\_min}{255}
  $$

  $$
  offset = round\left(\frac{ori\_clip\_min}{scale}\right)
  $$

  **3. Compute the clip boundaries of fake quant**

  $$
  clip\_min = scale \times offset
  $$

  $$
  clip\_max = scale \times (offset + 255)
  $$

  **4. Clamp x**

  $$
  clamped\_x = \min(\max(x, clip\_min), clip\_max)
  $$

  **5. Generate the mask**

  $$
  clampMinMask = \begin{cases} 1.0 & \text{if } x \geq clip\_min \\ 0.0 & \text{otherwise} \end{cases}
  $$

  $$
  clampMaxMask = \begin{cases} 1.0 & \text{if } x \leq clip\_max \\ 0.0 & \text{otherwise} \end{cases}
  $$

  **6. Quantize**

  $$
  raw\_x = \frac{clamped\_x}{scale}
  $$

  $$
  round\_x = round(raw\_x)
  $$

  **7. Compute clamped_loss**

  $$
  clamped\_loss = \frac{round\_x - raw\_x}{255}
  $$

  **8. Compute loss_m**

  $$
  raw\_m = \frac{ori\_clip\_min}{scale}
  $$

  $$
  round\_m = round(raw\_m)
  $$

  $$
  loss\_m = \frac{round\_m - raw\_m}{255}
  $$

  **9. Select the loss according to the mask**

  $$
  xClampedLoss = \begin{cases} clamped\_loss & \text{if } clampMinMask = 1.0 \\ loss\_m & \text{otherwise} \end{cases}
  $$

  $$
  xClampedLoss = \begin{cases} xClampedLoss & \text{if } clampMaxMask = 1.0 \\ loss\_m & \text{otherwise} \end{cases}
  $$

  **10. Compute y**

  $$
  y = round\_x \times scale
  $$

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|--------|---------------|------|----------|----------|
| x | Input | Denotes the input data tensor, corresponding to x in the formula; empty tensors are supported; the shape supports 0 to 8 dimensions. | FLOAT, FLOAT16 | ND |
| clamp_min | Input | Denotes the quantization lower-bound tensor, corresponding to clampMin in the formula; the data type must be consistent with x; the shape must be 1. | FLOAT, FLOAT16 | ND |
| clamp_max | Input | Denotes the quantization upper-bound tensor, corresponding to clampMax in the formula; the data type must be consistent with x; the shape must be 1. | FLOAT, FLOAT16 | ND |
| fixed_min | Optional attribute | Whether to fix the lower bound to 0; when true, ori_clip_min=0, and when false, ori_clip_min=min(clampMin, 0); the default value is false. | BOOL | - |
| num_bits | Optional attribute | Quantization bit width; currently only 8 is supported; the default value is 8. | INT | - |
| y | Output | Quantized output tensor, corresponding to y in the formula; empty tensors are supported; the data type must be consistent with x; the shape is consistent with x. | FLOAT, FLOAT16 | ND |
| clamp_min_mask | Output | Lower-bound mask tensor, with the value 1.0 (x >= clip_min) or 0.0; the shape must be consistent with y. | FLOAT, FLOAT16 | ND |
| clamp_max_mask | Output | Upper-bound mask tensor, with the value 1.0 (x <= clip_max) or 0.0; the shape must be consistent with y. | FLOAT, FLOAT16 | ND |
| x_clamped_loss | Output | Quantization loss tensor, corresponding to xClampedLoss in the formula; the shape must be consistent with y. | FLOAT, FLOAT16 | ND |

## Constraints

- The data types of x, clamp_min, and clamp_max must be consistent; only FLOAT and FLOAT16 are supported.
- The shapes of clamp_min and clamp_max must be 1 (scalar), and the shape of x supports 0 to 8 dimensions.

## Invocation Description

| Invocation method | Sample code | Description |
|----------|----------|------|
| Graph mode | [test_geir_acts_ulq](./examples/test_geir_acts_ulq.cpp) | Calls the ActsULQ operator by building a graph through [operator IR](./op_graph/acts_ulq_proto.h). |
