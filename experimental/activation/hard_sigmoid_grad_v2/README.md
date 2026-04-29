# HardSigmoidGradV2 算子

HardSigmoid 激活函数的反向梯度算子，基于 Ascend C 实现。

## 算子功能

计算 HardSigmoid 的梯度：

```text
grad_input = grad_output * (1/6)    当 -3 < self < 3
grad_input = 0                      其他情况
```

**输入**：

- `grad_output`：上游梯度张量
- `self`：前向传播的输入张量

**输出**：

- `grad_input`：计算得到的梯度张量（shape 与输入相同）

## 支持的数据类型

| 数据类型 | 支持 | 精度标准 |
|----------|------|----------|
| float16 (fp16) | 是 | rtol=0.001, atol=0.001 |
| float32 (fp32) | 是 | rtol=0.0001, atol=0.0001 |
| bfloat16 (bf16) | 是 | rtol=0.004, atol=0.004 |

## 目标芯片

- **芯片型号**：Ascend910B
- **架构**：arch32
- **CANN 版本**：9.0.0

## 编译与安装

### 前置条件

- 已安装 CANN Toolkit（9.0.0 或兼容版本）
- 已配置 `ASCEND_HOME_PATH` 环境变量

### 编译算子包

```bash
cd /path/to/hard_sigmoid_grad_v2
mkdir -p build && cd build
cmake .. -DASCEND_COMPUTE_UNIT=ascend910b
make -j$(nproc)
make binary
make package
```

或使用构建脚本（推荐）：

```bash
bash build.sh --soc=ascend910b
```

### 安装算子包

编译完成后在 `build/` 目录下生成 `custom_opp_ubuntu_aarch64.run` 安装包：

```bash
./build/custom_opp_ubuntu_aarch64.run
```

安装后算子包位于 `$ASCEND_OPP_PATH/vendors/` 目录下。

## 使用方法

### ACLNN 接口

```cpp
#include "aclnn_hard_sigmoid_grad_v2.h"

// 获取 workspace 大小
aclnnStatus ret = aclnnHardSigmoidGradV2GetWorkspaceSize(
    gradOutput, self, gradInput, &workspaceSize, &executor);

// 执行算子
ret = aclnnHardSigmoidGradV2(workspace, workspaceSize, executor, stream);
```

## 运行测试

### 单元测试（UT）

```bash
cd tests/ut
bash run.sh
```

UT 包含 22 个测试用例，覆盖 Tiling 分支和 InferShape 逻辑。

### 系统测试（ST）

```bash
cd tests/st
bash run.sh --mode=real --device=0
```

ST 包含 24 个测试用例，在真实 NPU 上验证精度：

- fp16：12 个用例
- fp32：4 个用例
- bf16：8 个用例

## 目录结构

```text
hard_sigmoid_grad_v2/
├── CMakeLists.txt                          # 顶层 CMake 配置
├── README.md                               # 本文件
├── op_kernel/
│   ├── CMakeLists.txt                      # Kernel 编译配置
│   ├── hard_sigmoid_grad_v2_arch32.cpp        # Kernel 入口（模板实例化）
│   └── arch32/
│       ├── hard_sigmoid_grad_v2.h             # Kernel 核心实现
│       ├── hard_sigmoid_grad_v2_tiling_data.h # Tiling 数据结构定义
│       └── hard_sigmoid_grad_v2_tiling_key.h  # TilingKey 定义（6 个 Key）
├── op_host/
│   ├── CMakeLists.txt                      # Host 编译配置
│   ├── hard_sigmoid_grad_v2_def.cpp           # 算子注册与 dtype/format 定义
│   ├── hard_sigmoid_grad_v2_infershape.cpp    # Shape 推导（elementwise）
│   └── arch32/
│       └── hard_sigmoid_grad_v2_tiling.cpp    # Host 端 Tiling 策略实现
├── docs/
│   ├── DEVELOPMENT_LOG.md                  # 开发日志
│   ├── REQUIREMENT_ANALYSIS.md             # 需求分析文档
│   ├── hard_sigmoid_grad_v2_DETAILED_DESIGN.md # 详细设计文档
│   ├── hard_sigmoid_grad_v2_TEST_DESIGN.md    # 测试设计文档
│   ├── hard_sigmoid_grad_v2_ITERATION_PLAN.md # 迭代计划
│   ├── final_precision_report.md           # 最终精度验收报告
│   └── INTEGRATION_REPORT.md              # 代码集成报告
├── tests/
│   ├── ut/                                 # 单元测试（22 用例）
│   │   ├── CMakeLists.txt
│   │   ├── run.sh
│   │   └── op_host/                        # Tiling/InferShape 测试
│   ├── st/                                 # 系统测试（24 用例）
│   │   ├── CMakeLists.txt
│   │   ├── run.sh
│   │   ├── test_aclnn_hard_sigmoid_grad_v2.cpp
│   │   └── testcases/                      # L0/L1 测试用例表
│   └── reports/                            # 各迭代测试报告
└── probe/                                  # 穿刺验证工程
```

## 实现说明

### TilingKey 策略

算子使用 6 个 TilingKey，由数据类型和 Buffer 模式组合而成：

| TilingKey | 数据类型 | Buffer 模式 |
|-----------|----------|-------------|
| 0 | fp16 | 单 buffer |
| 1 | fp16 | 双 buffer |
| 2 | fp32 | 单 buffer |
| 3 | fp32 | 双 buffer |
| 4 | bf16 | 单 buffer |
| 5 | bf16 | 双 buffer |

### 关键实现细节

- **标量比较**：使用 Compares API（Level 2）进行 self 与边界值（-3, 3）的比较
- **bf16 处理**：bf16 不支持直接 Compares，通过 Cast 到 float 进行计算后再 Cast 回 bf16
- **对齐要求**：Compares Level 2 要求 count*sizeof(T) 按 256 字节对齐，ubFactor 对齐到 256/sizeof(ComputeT)
