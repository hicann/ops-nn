<!--
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 CANN Open Software License Agreement Version 2.0 (the "License").
 Please refer to the License for details. You may not use this file except in compliance with the License.
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 See LICENSE in the root of the software repository for the full text of the License.

 NOTE: Portions of this content were AI-generated and have been
 technically reviewed for functional accuracy and security
-->

# ApplyRmsProp

> TensorFlow `ApplyRMSProp`（RMSProp 优化器，非 centered 版本）算子在 Ascend 950 上的自定义实现。

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> 本次交付仅适配 <term>Ascend 950PR/Ascend 950DT</term>（DAV_3510 / `__NPU_ARCH__ == 3510`）。其他产品后续按规划扩展。

## 功能说明

- **算子功能**：实现 TensorFlow `ApplyRMSProp`（非 centered 版本）优化器语义，对参数张量 `var`、梯度平方移动平均 `ms`、动量累积 `mom` 进行 **inplace** 更新，用于深度学习模型训练（尤其是 RNN/Transformer 类模型）的参数更新阶段。

- **计算公式**（对每个元素 $i$）：

  $$
  ms[i] = ms[i] + (1 - \rho) \cdot (g[i]^2 - ms[i])
  $$

  $$
  mom[i] = \mu \cdot mom[i] + \alpha \cdot g[i] \cdot (\epsilon + ms[i])^{-\frac{1}{2}}
  $$

  $$
  var[i] = var[i] - mom[i]
  $$

  其中 $\alpha=$ `lr`（学习率），$\rho=$ `rho`（衰减率，通常 0.9），$\mu=$ `momentum`（动量系数），$\epsilon=$ `epsilon`（数值稳定项），$g=$ `grad`（当前步梯度）。

- **使用场景**：RMSProp 优化器的 step 阶段；与 TensorFlow `tf.raw_ops.ApplyRMSProp` 语义对齐，`var`/`ms`/`mom` 在函数返回后被原地覆盖。

- **版本差异**：v1.0 仅支持非 centered 版本；如需 `ApplyCenteredRMSProp`，另走独立接口。

## 参数说明

| 参数名 | 输入/输出 | 数据类型 | 数据格式 | 维度 rank |
|--------|-----------|----------|----------|-----------|
| var | 输入/输出（inplace）| FLOAT、FLOAT16、BFLOAT16 | ND | 1-8 |
| ms  | 输入/输出（inplace）| 与 var 一致 | ND | 1-8 |
| mom | 输入/输出（inplace）| 与 var 一致 | ND | 1-8 |
| grad | 输入 | 与 var 一致 | ND | 1-8 |
| lr  | Attr (float) | float | - | 标量 |
| rho | Attr (float) | float | - | 标量 |
| momentum | Attr (float) | float | - | 标量 |
| epsilon  | Attr (float) | float | - | 标量 |
| var_out | 输出 | 与 var 一致 | ND | 1-8 |
| ms_out  | 输出 | 与 var 一致 | ND | 1-8 |
| mom_out | 输出 | 与 var 一致 | ND | 1-8 |

## 约束说明

- **dtype 严格一致**：`var`、`ms`、`mom`、`grad` 的数据类型必须完全一致（均为 FLOAT、FLOAT16 或 BFLOAT16 之一），算子内部不做隐式 Cast。
- **shape 约束**：`var`、`ms`、`mom`、`grad` 的 shape 必须完全相同，rank ∈ [1, 8]。
- **空 tensor**：当 `var`/`ms`/`mom`/`grad` 元素数为 0 时，接口直接返回成功，不执行任何计算。
- **数值稳定性**：建议 `epsilon > 0`（如 `1e-7`）以保证 $\sqrt{\epsilon + ms}$ 分母非零。
- **inplace 语义**：`var_out` / `ms_out` / `mom_out` 在 aclnn 接口中通常与 `var` / `ms` / `mom` 指向同一内存。
- **fp16 / bf16 精度**：Kernel 内部将低精度数据 Cast 到 fp32 做中间计算，最终再 Cast 回原 dtype 写回 GM。

## 调用说明

| 调用方式 | 调用样例 |
|----------|----------|
| aclnn 调用 | [examples/test_aclnn_apply_rms_prop.cpp](./examples/test_aclnn_apply_rms_prop.cpp) |

## 目录结构

```
apply_rms_prop/
├── README.md                         # 本文件
├── CMakeLists.txt                    # 顶层 CMake 配置
├── op_host/                          # Host 侧 (InferShape / Tiling / OpDef)
│   ├── CMakeLists.txt
│   ├── apply_rms_prop_def.cpp
│   ├── apply_rms_prop_infershape.cpp
│   └── apply_rms_prop_tiling.cpp
├── op_kernel/                        # Device 侧 Kernel 实现
│   ├── apply_rms_prop.cpp
│   ├── apply_rms_prop.h
│   ├── apply_rms_prop_tiling_data.h
│   └── apply_rms_prop_tiling_key.h
├── examples/                         # aclnn 调用示例
│   └── test_aclnn_apply_rms_prop.cpp
└── tests/                            # 测试占位
```
