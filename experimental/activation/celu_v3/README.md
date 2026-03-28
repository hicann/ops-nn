# CeluV3 算子

基于 Ascend C 框架实现的 CELU（Continuously Differentiable Exponential Linear Unit）激活函数算子。

## 算子功能

CeluV3 实现逐元素的 CELU 激活函数，计算公式为：

$$CELU(x) = \max(0, x) + \min(0, \alpha \cdot (\exp(x / \alpha) - 1))$$

其中 `alpha` 为标量系数（默认值 1.0，不能为 0）。当 `alpha = 1.0` 时退化为标准 ELU 激活函数。

对标 PyTorch 接口：`torch.nn.functional.celu(input, alpha=1.0)`

### ACLNN 接口

```cpp
// 第一段接口：计算 workspace 大小
aclnnStatus aclnnCeluV3GetWorkspaceSize(
    const aclTensor* self, const aclScalar* alpha,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

// 第二段接口：执行计算
aclnnStatus aclnnCeluV3(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, const aclrtStream stream);
```

### 输入输出

| 参数 | 类型 | 格式 | 说明 |
|------|------|------|------|
| self | Tensor (输入) | ND | 输入张量，支持任意 shape |
| alpha | Scalar (输入) | - | 激活系数，不能为 0 |
| out | Tensor (输出) | ND | 输出张量，shape 和 dtype 与 self 一致 |

## 支持的数据类型

| 数据类型 | 支持 | 计算路径 |
|---------|------|---------|
| float32 | 是 | float32 直接计算 |
| float16 | 是 | 提升至 float32 计算后转回 |
| bfloat16 | 是 | 提升至 float32 计算后转回 |

## 支持的芯片

| 芯片型号 | 架构 | 编译宏 |
|---------|------|--------|
| Ascend 910B | A2 (arch32) | DAV_2201 |

## 精度标准

| 数据类型 | 相对误差 (rtol) | 绝对误差 (atol) | 说明 |
|---------|----------------|----------------|------|
| float32 | 1e-4 | 1e-4 | 双万分之一 |
| float16 | 1e-3 | 1e-3 | 双千分之一 |
| bfloat16 | 1e-3 | 1e-3 | 双千分之一 |

精度验收结果：200/200 CSV用例全部通过（L0:40 + L1:160），另有14条CPU Golden自测全部通过，覆盖 fp32/fp16/bf16 全部数据类型。

## 性能概况

在 Ascend 910B3 上测试（CANN 9.0.0）：

| Shape | dtype | 吞吐量 (GOPS) | 场景 |
|-------|-------|---------------|------|
| [1048576] | float32 | 214.87 | 满核，计算密集 |
| [1048576] | float16 | 223.37 | 满核 + Cast 路径 |
| [1048576] | bfloat16 | 217.63 | 满核 + Cast 路径 |

大 shape 场景下向量计算单元利用率约 52%，属于计算密集型特征，多核扩展性良好。

## 工程结构

```
ops/celu_v3/
├── CMakeLists.txt                          # 顶层 CMake 构建文件
├── build.sh                                # 编译脚本
├── op_host/
│   ├── CMakeLists.txt                      # Host 侧构建
│   ├── celu_v3_def.cpp                     # 算子定义（IR、Attr、AICore 配置）
│   ├── celu_v3_infershape.cpp              # Shape 推导
│   └── arch32/
│       └── celu_v3_tiling.cpp              # Tiling 计算（多核切分 + UB 切分）
├── op_kernel/
│   ├── CMakeLists.txt                      # Kernel 侧构建
│   ├── celu_v3_arch32.cpp                  # Kernel 入口
│   └── arch32/
│       ├── celu_v3.h                       # Kernel 实现（CopyIn/Compute/CopyOut）
│       ├── celu_v3_tiling_data.h           # TilingData 结构定义
│       └── celu_v3_tiling_key.h            # TilingKey 模板参数定义
├── docs/                                   # 文档
│   ├── DEVELOPMENT_LOG.md                  # 开发日志
│   ├── REQUIREMENT_ANALYSIS.md             # 需求分析文档
│   ├── celu_v3_DETAILED_DESIGN.md          # 详细设计文档
│   ├── celu_v3_TEST_DESIGN.md              # 测试设计文档
│   ├── final_precision_report.md           # 精度验收报告
│   ├── final_performance_report.md         # 性能验收报告
│   └── final_code_review_report.md         # 代码检视报告
└── tests/
    ├── ut/                                 # 单元测试（Tiling 逻辑、InferShape）
    │   ├── CMakeLists.txt
    │   └── run.sh
    └── st/                                 # 系统测试（NPU 精度验证）
        ├── CMakeLists.txt
        ├── run.sh
        ├── test_aclnn_celu_v3.cpp          # CPU Golden 自测（14 条，CMake target: test_golden_selftest）
        ├── test_csv_runner.cpp             # CSV 驱动测试用例
        ├── test_perf_benchmark.cpp         # 性能基准测试
        └── testcases/
            ├── L0_test_cases.csv           # L0 门槛用例（40 条，单因子覆盖）
            └── L1_test_cases.csv           # L1 功能用例（160 条，两两组合覆盖）
```

## 编译方法

### 前置条件

- CANN Toolkit 已安装（cann-9.0.0 或以上）
- 环境变量已配置（`source /path/to/cann/set_env.sh`）
- Ascend 910B NPU 可用

### 编译算子包

```bash
cd ops/celu_v3

# 编译并打包（生成 .run 安装包）
bash build.sh --pkg --soc=ascend910b

# 仅编译（不打包）
bash build.sh --soc=ascend910b

# 清理构建产物
bash build.sh --make_clean
```

编译时使用 `--ops CeluV3` 指定算子名称（已内置于 CMakeLists.txt），避免影响其他算子。

### 安装算子包

```bash
# 安装编译产物
cd build
./custom_opp_ubuntu_aarch64.run
```

### 部署前清理

部署前需先删除以下目录（如果存在）：

```bash
rm -rf build/ third_party/ build_out/
```

## 测试方法

### 单元测试 (UT)

验证 Host 侧 Tiling 逻辑、InferShape 正确性：

```bash
# 方式一：通过 build.sh
bash build.sh --soc=ascend910b -u

# 方式二：直接运行
cd tests/ut
./run.sh
```

### 系统测试 (ST)

验证 NPU 上的端到端精度：

```bash
# 方式一：通过 build.sh
bash build.sh --soc=ascend910b -s

# 方式二：运行 CSV 测试用例（NPU 精度验证）
cd tests/st
./run.sh --csv testcases/L0_test_cases.csv                              # L0 门槛用例（40 条）
./run.sh --csv testcases/L0_test_cases.csv testcases/L1_test_cases.csv  # L0 + L1 全量（200 条）

# 方式三：运行 CPU Golden 自测（14 条，不依赖 NPU）
./run.sh --golden

# Mock 模式（CPU golden 验证，无需 NPU）
./run.sh --mock

# 运行全部测试（UT + ST）
cd ops/celu_v3
bash build.sh --soc=ascend910b -a
```

## 约束与限制

- alpha 不能为 0（除零保护）
- 输出 dtype 和 shape 必须与输入一致
- 仅支持 ND 数据格式
- 空 tensor（shape 含 0 维度）：正常接受，执行 no-op 操作
- 特殊值遵循 IEEE 754 浮点运算规则（NaN 输入产生 NaN 输出，+Inf/-Inf 按公式计算）
