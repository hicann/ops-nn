# BnInferGrad 自定义算子

## 功能简介

BnInferGrad 是基于 Ascend C 开发的 BatchNorm 推理阶段反向梯度计算算子，用于计算 BatchNorm 在推理模式（training=False）下输入数据关于损失函数的梯度。

核心公式：

```
x_backprop = grads * scale / sqrt(batch_variance + epsilon)
```

等价语义：PyTorch `torch.nn.functional.batch_norm` 推理模式下反向传播中的输入梯度计算。

## 支持平台与数据类型

| 平台 | 支持数据类型 | 输出数据类型 |
|------|-------------|-------------|
| Ascend950 (arch35) | fp32, fp16, bf16 | 与输入一致 |

说明：fp16/bf16 输入在 Kernel 内部提升为 fp32 计算后再转回原类型，以保证精度。scale 和 batch_variance 始终为 fp32 类型。

## 支持数据格式

| 格式 | TilingKey | 说明 |
|------|-----------|------|
| NCHW | CONTIGUOUS (0) | 通道维度在第 2 维，与 NHWC 共用同一 TilingKey |
| NHWC | CONTIGUOUS (0) | 通道维度在最后一维，展平为连续数据处理 |
| NC1HWC0 | NC1HWC0 (1) | 五维格式，通道分布在 C1 和 C0 两个维度，需特殊处理 |

## ACLNN API 接口

```cpp
// 第一段：计算 workspace 大小
aclnnStatus aclnnBnInferGradGetWorkspaceSize(
    const aclTensor* grads,
    const aclTensor* scale,
    const aclTensor* batchVariance,
    float epsilon,
    aclTensor* xBackprop,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

// 第二段：执行计算
aclnnStatus aclnnBnInferGrad(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

### 输入参数

| 参数名 | 类型 | Shape | 说明 |
|-------|------|-------|------|
| grads | const aclTensor* | (N, C, H, W) 4D | 上游梯度 |
| scale | const aclTensor* | (C,) 1D | BatchNorm 缩放参数 gamma |
| batch_variance | const aclTensor* | (C,) 1D | BatchNorm 推理阶段方差 |
| epsilon | float | 标量 | 防除零常数，默认 0.0001 |

### 输出参数

| 参数名 | 类型 | Shape | 说明 |
|-------|------|-------|------|
| x_backprop | aclTensor* | 与 grads 相同 | 输入梯度 |

### 约束与限制

- grads 的 dtype 须为 float16、float32 或 bfloat16
- scale 和 batchVariance 的 dtype 须为 float32
- xBackprop 的 dtype 须与 grads 相同
- grads 为 4D 或 5D 张量
- scale 和 batchVariance 为 1D 张量，长度等于 grads 通道维度 C
- 当 format 为 NC1HWC0 时，scale 和 batchVariance 的 shape 需调整为 (C1, 1, 1, C0)

## 精度标准

| grads dtype | atol | rtol |
|------------|------|------|
| fp32 | 1e-4 | 1e-4 |
| fp16 | 1e-3 | 1e-3 |
| bf16 | 4e-3 | 4e-3 |

## 目录结构

```
bn_infer_grad/
├── CMakeLists.txt              # 顶层构建配置
├── build.sh                    # 编译和测试入口脚本
├── op_host/                    # Host 端代码
│   ├── bn_infer_grad_def.cpp           # 算子定义和注册
│   ├── bn_infer_grad_infershape.cpp    # 输出 shape 推导
│   ├── arch35/
│   │   └── bn_infer_grad_tiling.cpp    # Tiling 计算逻辑
│   └── CMakeLists.txt
├── op_kernel/                  # Kernel 端代码
│   ├── bn_infer_grad_arch35.cpp        # Kernel 入口
│   ├── common/
│   │   ├── bn_infer_grad.h             # Kernel 主类定义
│   │   ├── bn_infer_grad_tiling_data.h # Tiling 数据结构
│   │   ├── bn_infer_grad_tiling_key.h  # TilingKey 枚举
│   │   ├── bn_infer_grad_contiguous.h  # CONTIGUOUS 分支实现 (NCHW/NHWC)
│   │   └── bn_infer_grad_nc1hwc0.h     # NC1HWC0 分支实现
│   └── CMakeLists.txt
├── tests/
│   ├── ut/                     # 单元测试（InferShape + Tiling 逻辑验证）
│   │   ├── op_host/
│   │   │   ├── test_bn_infer_grad_infershape.cpp
│   │   │   └── test_bn_infer_grad_tiling.cpp
│   │   ├── CMakeLists.txt
│   │   └── common/
│   └── st/                     # 系统测试（端到端验证）
│       ├── test_aclnn_bn_infer_grad.cpp
│       ├── testcases/          # CSV 格式测试用例
│       ├── run.sh
│       └── CMakeLists.txt
└── docs/                       # 文档
    ├── DEVELOPMENT_LOG.md      # 开发日志
    ├── REQUIREMENT_ANALYSIS.md # 需求分析
    ├── DETAILED_DESIGN.md      # 详细设计
    ├── TEST_DESIGN.md          # 测试设计
    └── CODE_REVIEW_REPORT.md   # 代码检视报告
```

## 编译方法

### 前置条件

- CANN 9.0.0 已安装，`ASCEND_HOME_PATH` 环境变量已设置
- 目标平台为 Ascend950 (arch35)

### 编译算子包

```bash
# 编译并生成算子安装包
bash build.sh --pkg --soc=ascend950 -j8

# 安装算子包
./build/custom_opp_ubuntu_aarch64.run

# 清理构建产物
bash build.sh --make_clean
```

### 仅编译（不生成安装包）

```bash
bash build.sh --soc=ascend950 -j8
```

## 测试方法

### 运行 UT

```bash
cd tests/ut && bash run.sh
```

UT 测试覆盖 InferShape 和 Tiling 计算逻辑，验证各 TilingKey 分支的参数计算正确性。共 66 个用例，覆盖：

- 3 种数据类型（fp32, fp16, bf16）
- 3 种数据格式（NCHW, NHWC, NC1HWC0）
- 多核切分、跨 dtype 等价性、空 tensor 边界、单通道边界

### 运行 ST

```bash
# Mock 模式（CPU Golden 比对，无需 NPU）
cd tests/st && bash run.sh --mock

# Real 模式（真实 NPU 执行，需先安装算子包）
cd tests/st && bash run.sh --real
```

ST 测试为端到端测试，支持 Mock 模式（CPU Golden 自验证）和 Real 模式（真实 NPU 执行）。共 63 个用例，覆盖：

- 3 种数据类型（fp32, fp16, bf16）
- 3 种数据格式（NCHW, NHWC, NC1HWC0）
- 9 个 dtype x format 全交叉组合
- 边界条件（最小 shape、单通道、大 batch、非对齐维度、大通道、spatial 退化）
- epsilon 变体

## 已知限制

1. **真实 NPU 验证待补充**：算子为 ascend950 平台编译，当前开发环境为 910B3，真实 NPU 测试因平台不兼容未执行。Mock 模式下 63/63 用例全部通过。待 ascend950 硬件环境可用后需补充 Real NPU 验证。
2. **性能验收未执行**：无明确性能指标要求，待真实 NPU 可用后可补充性能测试。

## 开发信息

- 开发周期：2026-03-27 ~ 2026-03-28
- 详细开发过程见 `docs/DEVELOPMENT_LOG.md`
