# HardShrinkGrad

Hardshrink 激活函数的反向梯度算子，用于神经网络训练过程中的梯度反向传播。基于 Ascend C 实现，运行在昇腾 NPU 上。

## 算子功能

HardShrinkGrad 根据前向输入 `self` 的绝对值与阈值 `lambd` 的比较结果，决定是否将上游梯度 `grad_output` 传递到下游：

```bash
output_i = grad_output_i,   if |self_i| > lambd
           0,                otherwise
```

等价表达：`output = grad_output * (|self| > lambd ? 1 : 0)`

**对标框架**：PyTorch `hardshrink_backward`

## 算子规格

| 项目 | 规格 |
|------|------|
| 算子名称 | HardShrinkGrad |
| 算子类别 | Elementwise（逐元素反向梯度算子） |
| 调用方式 | ACLNN (`aclnnHardShrinkGrad`) |
| 目标芯片 | Ascend910B (910B3) |
| 目标架构 | arch32 (DAV_2201) |

### 输入

| 参数名 | 含义 | Shape | Dtype |
|--------|------|-------|-------|
| grad_output | 上游反向传播的梯度 | 任意 shape，最高 8 维 | float16, float32, bfloat16 |
| self | Hardshrink 前向的原始输入 | 与 grad_output 相同 | 与 grad_output 相同 |

### 输出

| 参数名 | 含义 | Shape | Dtype |
|--------|------|-------|-------|
| output | 反向梯度结果 | 与输入相同 | 与输入相同 |

### 属性

| 属性名 | 类型 | 默认值 | 约束 |
|--------|------|--------|------|
| lambd | float | 0.5 | >= 0 |

### 约束

- grad_output 和 self 的 dtype 和 shape 必须完全一致（不支持广播）
- 最大支持 8 维张量
- 空 tensor（元素数量为 0）：直接返回空 tensor

## 使用方法

### ACLNN 调用

```cpp
#include "aclnnop/aclnn_hard_shrink_grad.h"

// 第一段：计算 workspace 大小
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus ret = aclnnHardShrinkGradGetWorkspaceSize(
    gradOutput, self, lambd, output,
    &workspaceSize, &executor);

// 分配 workspace
void* workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 第二段：执行计算
ret = aclnnHardShrinkGrad(workspace, workspaceSize, executor, stream);
```

### 编译与安装

```bash
# 编译算子包
bash build.sh

# 安装到 CANN OPP 目录
# 编译产物在 build/ 目录下，按自定义算子安装流程部署
```

### 运行 UT 测试

```bash
cd tests/ut
bash run.sh
```

### 运行 ST 测试

```bash
cd tests/st
# 编译并运行（需要 NPU 设备）
bash run.sh
```

## 精度标准

| 数据类型 | 精度标准 | rtol | atol |
|---------|---------|------|------|
| float16 | 双千分之一 | 1e-3 | 1e-3 |
| float32 | 双万分之一 | 1e-4 | 1e-4 |
| bfloat16 | 双千分之一 | 1e-3 | 1e-3 |

精度比对方法：逐元素 allclose，判定公式 `|golden - actual| <= atol + rtol * |golden|`

HardShrinkGrad 为条件赋值算子，输出为精确的 0 或精确的 grad_output 值，理论上不引入额外计算误差。

## 目录结构

```text
hard_shrink_grad/
├── CMakeLists.txt                              # 顶层构建配置
├── build.sh                                    # 编译脚本
├── README.md                                   # 本文件
├── op_host/                                    # Host 侧实现
│   ├── CMakeLists.txt                          # Host 构建配置
│   ├── hard_shrink_grad_def.cpp                # 算子原型注册
│   ├── hard_shrink_grad_infershape.cpp         # Shape 推导（output shape = input shape）
│   └── arch32/                                 # Ascend910B (DAV_2201)
│       └── hard_shrink_grad_tiling.cpp         # Tiling 实现（多核切分、UB 切分、TilingKey 选择）
├── op_kernel/                                  # Kernel 侧实现
│   ├── CMakeLists.txt                          # Kernel 构建配置
│   ├── hard_shrink_grad_arch32.cpp             # Kernel 入口（模板实例化，双类分发）
│   └── arch32/                                 # Ascend910B (DAV_2201)
│       ├── hard_shrink_grad.h                  # Kernel 类定义与实现
│       ├── hard_shrink_grad_tiling_data.h      # TilingData 结构体
│       └── hard_shrink_grad_tiling_key.h       # TilingKey 定义（6 种模板组合）
├── docs/                                       # 文档
│   ├── LOG.md                                  # 开发日志
│   ├── REQUIREMENTS.md                         # 需求分析文档
│   ├── DESIGN.md                               # 详细设计文档
│   ├── TEST.md                                 # 测试设计文档
│   ├── PLAN.md                                 # 迭代执行计划
│   ├── precision-report.md                     # 精度验收报告（20/20 = 100%）
│   └── review-report.md                        # 代码检视报告
├── tests/                                      # 测试代码
│   ├── ut/                                     # 单元测试（54 条：48 Tiling + 6 InferShape）
│   │   ├── CMakeLists.txt
│   │   ├── run.sh
│   │   ├── test-report.md
│   │   └── op_host/                            # UT 源码
│   ├── st/                                     # 系统测试（20 条 L0 用例）
│   │   ├── CMakeLists.txt
│   │   ├── run.sh
│   │   ├── test_aclnn_hard_shrink_grad.cpp     # ST 测试主程序
│   │   └── testcases/                          # 测试用例定义
│   └── reports/                                # 迭代报告
│       ├── iter1-integration-report.md
│       ├── iter1-acceptance-report.md
│       ├── iter2-integration-report.md
│       ├── iter2-acceptance-report.md
│       ├── iter3-integration-report.md
│       └── iter3-acceptance-report.md
└── probe/                                      # 穿刺验证工程（6 个）
    ├── PROBE_SUMMARY.md                        # 穿刺汇总报告
    ├── probe_fp16_single_buf/                  # 迭代一：fp16 单缓冲
    ├── probe_fp32_double_buf/                  # 迭代一：fp32 双缓冲
    ├── probe_fp32_single_buf/                  # 迭代一：fp32 单缓冲
    ├── probe_bf16_support/                     # 迭代二：bf16 支持
    ├── probe_empty_tensor/                     # 迭代二：空 tensor
    └── probe_lambd_zero/                       # 迭代二：lambd=0 边界
```

## 实现要点

### 计算方案

- **fp32**：直接使用 Abs -> Compare(GT, tensor-tensor) -> Select 计算
- **fp16/bf16**：先 Cast 到 fp32，再执行 Abs -> Compare -> Select，结果 Cast 回原 dtype

### 关键技术决策

1. **arch32 上 Compare API 对 fp16/bf16 不可靠**：统一 Cast 到 fp32 计算，确保精度达标
2. **bf16 不支持 Abs/Compare/Select API**：复用 fp16 的 CastFp32 路径
3. **使用 Compare(tensor-tensor) 替代 CompareScalar**：lambd 通过 Duplicate 广播到 tensor，更通用可靠

### TilingKey 组合

| TilingKey | Dtype | BufferMode | 适用场景 |
|-----------|-------|------------|---------|
| 1 | float16 | 单缓冲 | fp16 小数据量 (<=1024) |
| 2 | float16 | 双缓冲 | fp16 大数据量 (>1024) |
| 3 | float32 | 单缓冲 | fp32 小数据量 (<=1024) |
| 4 | float32 | 双缓冲 | fp32 大数据量 (>1024) |
| 5 | bfloat16 | 单缓冲 | bf16 小数据量 (<=1024) |
| 6 | bfloat16 | 双缓冲 | bf16 大数据量 (>1024) |

## 测试结果

| 测试类型 | 用例数 | 通过数 | 通过率 |
|---------|--------|--------|--------|
| UT (Tiling + InferShape) | 54 | 54 | 100% |
| ST (NPU 精度验证) | 20 | 20 | 100% |
| 穿刺验证 | 6 | 6 | 100% |

## 开发环境

| 项目 | 版本 |
|------|------|
| CANN Toolkit | 8.5.0 |
| 芯片型号 | Ascend 910B3 |
| OS | Linux 5.10.0 (aarch64) |
| npu-smi | 25.5.0 |
