# ForeachDivScalar 算子

基于 Ascend C 框架开发的自定义算子，实现张量列表（TensorList）中每个张量逐元素除以同一标量值。

## 算子功能

$$y_i = \frac{x_i}{\text{scalar}} \quad (i = 0, 1, \ldots, n-1)$$

- **输入 x**：TensorList，支持 FLOAT32 / FLOAT16 / BFLOAT16，ND 格式，1~8 维
- **输入 scalar**：单元素张量，支持 FLOAT32 / FLOAT16 / DOUBLE
- **输出 y**：TensorList，dtype 和 shape 与 x 一致

## 支持平台

- 芯片：Ascend910B（910B1/910B2/910B3/910B4）
- 架构：arch32 / DAV_2201
- CANN：9.0.0

## ACLNN 接口

```cpp
aclnnStatus aclnnForeachDivScalarGetWorkspaceSize(
    const aclTensorList* x, const aclTensor* scalar,
    aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnForeachDivScalar(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

## 目录结构

```
ops/foreach_div_scalar/
├── op_kernel/                      # Kernel 实现
│   ├── foreach_div_scalar_arch32.cpp   # 核函数入口
│   └── arch32/
│       ├── foreach_div_scalar.h            # 核心算子逻辑
│       ├── foreach_div_scalar_tiling_data.h # TilingData 结构
│       └── foreach_div_scalar_tiling_key.h  # TilingKey 定义
├── op_host/                        # Host 侧逻辑
│   ├── foreach_div_scalar_def.cpp          # 算子定义
│   ├── foreach_div_scalar_infershape.cpp   # 形状推导
│   └── arch32/
│       └── foreach_div_scalar_tiling.cpp   # Tiling 实现
├── tests/
│   ├── ut/                         # 单元测试（27 用例）
│   └── st/                         # 系统测试（49 用例）
├── docs/                           # 文档
│   ├── DEVELOPMENT_LOG.md          # 开发日志
│   ├── REQUIREMENT_ANALYSIS.md     # 需求分析
│   ├── DETAILED_DESIGN.md          # 详细设计
│   ├── foreach_div_scalar_TEST_DESIGN.md  # 测试设计
│   ├── STAGE3_ACCEPTANCE_REPORT.md # 精度验收报告
│   ├── PERFORMANCE_REPORT.md       # 性能验收报告
│   └── CODE_REVIEW_REPORT.md       # 代码检视报告
└── build.sh                        # 编译脚本
```

## 编译与安装

```bash
# 编译算子包
bash build.sh --pkg --soc=ascend910b -j8

# 安装算子包
bash build/custom_opp_ubuntu_aarch64.run --quiet
```

## 测试

```bash
# UT 测试
bash tests/ut/run.sh

# ST 测试（真实 NPU）
bash tests/st/run.sh --real

# ST 测试（Mock 模式，无需 NPU）
bash tests/st/run.sh --mock
```

## 精度标准

| 数据类型 | 相对误差 | 绝对误差 |
|---------|---------|---------|
| FLOAT32 | < 1e-4 | < 1e-4 |
| FLOAT16 | < 1e-3 | < 1e-3 |
| BFLOAT16 | < 4e-3 | < 4e-3 |

## 关键设计

- **多核并行**：按 tensor 粒度分核（避免非对齐 DMA），核数动态调整
- **UB 分块**：三段式流水线（CopyIn/Compute/CopyOut），充分利用 UB 缓存
- **除法优化**：使用 `Muls(x, 1/scalar)` 替代逐元素除法
- **bf16 精度保护**：Cast→fp32 计算→Cast 回 bf16
- **scalar 读取**：Kernel 端 DMA 直接读取，支持 float/half/double 三种标量类型
