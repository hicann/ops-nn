# ApplyProximalGradientDescent

## 1. 算子简介

`ApplyProximalGradientDescent` 是一个带 L1/L2 正则的近端梯度下降（Proximal Gradient Descent）一步更新算子，是 SGD 在带稀疏 (L1) / 权重衰减 (L2) 正则场景下的推广，常用于 FOBOS、在线学习等训练优化场景。

- **对标接口**：TensorFlow `tensorflow.python.training.gen_training_ops.apply_proximal_gradient_descent`（原始 kernel 见 `tensorflow/core/kernels/training_ops.cc`）。
- **接口形式**：非 Inplace，结果写入独立输出张量 `varOut`；用户侧可让 `var` 与 `varOut` 指向同一块设备内存以实现 inplace。
- **目标平台**：Atlas A3 / Atlas 950 系列（Ascend950，`__NPU_ARCH__=3510`）。

## 2. 计算公式

$$
prox\_v = var - alpha \times delta
$$

$$
varOut = \dfrac{\operatorname{sign}(prox\_v)}{1 + alpha \times l2} \times \max\bigl(|prox\_v| - alpha \times l1,\ 0\bigr)
$$

其中 `sign(0) = 0`，与 TensorFlow 语义一致。

退化情形：

| 条件            | 等价公式                                       |
| --------------- | ---------------------------------------------- |
| `l1 = 0, l2 = 0` | `varOut = var - alpha * delta`（标准 SGD）      |
| `l1 = 0`         | `varOut = (var - alpha * delta) / (1 + alpha * l2)` |

## 3. 接口规格

### 3.1 函数原型

```cpp
aclnnStatus aclnnApplyProximalGradientDescentGetWorkspaceSize(
    const aclTensor *var,
    const aclTensor *alpha,
    const aclTensor *l1,
    const aclTensor *l2,
    const aclTensor *delta,
    aclTensor       *varOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);

aclnnStatus aclnnApplyProximalGradientDescent(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

### 3.2 参数说明

| 参数     | 输入/输出 | 描述                                                       |
| -------- | --------- | ---------------------------------------------------------- |
| `var`    | 输入      | 待更新权重张量，shape 任意 (1-8 维)                        |
| `alpha`  | 输入      | 学习率标量（0-D 或 shape=[1]，非负）                       |
| `l1`     | 输入      | L1 正则系数标量（0-D 或 shape=[1]，非负）                  |
| `l2`     | 输入      | L2 正则系数标量（0-D 或 shape=[1]，非负）                  |
| `delta`  | 输入      | 梯度张量，shape 与 `var` 完全相同                          |
| `varOut` | 输出      | 输出张量，shape/dtype/format 与 `var` 一致                 |

详细字段（非连续张量/错误码等）参见 [aclnnApplyProximalGradientDescent.md](docs/aclnnApplyProximalGradientDescent.md)。

## 4. 数据类型与 shape 约束

- **数据类型**：`var / alpha / l1 / l2 / delta / varOut` 必须一致，仅支持 `FLOAT`、`FLOAT16`。
- **Format**：`ND`。
- **Shape**：
  - `var` 维度范围 [1, 8]；
  - `delta.shape == varOut.shape == var.shape`（不广播）；
  - `alpha / l1 / l2` 必须为 0-D 或 shape=[1] 的标量张量。
- **值域建议**：`alpha / l1 / l2` 建议非负；传入负值时结果未定义。
- **内存别名**：`varOut` 允许与 `var` 指向同一块设备内存（实现 inplace），但 `varOut` 与 `delta` 不能别名。
- **精度标准**：
  - FP32：MERE < 2⁻¹³ (≈1.22e-4)，MARE < 10×2⁻¹³ (≈1.22e-3)；
  - FP16：MERE < 2⁻¹⁰ (≈9.77e-4)，MARE < 10×2⁻¹⁰ (≈9.77e-3)；
  - FP16 Kernel 内部会提升至 FP32 计算，输出 Cast 回 FP16。

## 5. 编译 / 安装 / 运行

### 5.1 构建并安装自定义算子包

```bash
# 1. 加载 CANN 环境
source /home/cjl/Ascend/ascend-toolkit/set_env.sh

# 2. 在算子根目录执行构建
cd ops/apply_proximal_gradient_descent
bash build.sh

# 构建脚本会自动将 run 包安装到
#   ${ASCEND_HOME_PATH}/opp/vendors/apply_proximal_gradient_descent_custom
```

### 5.2 运行 ST 测试

```bash
cd ops/apply_proximal_gradient_descent/tests/st
bash run.sh --mock               # CPU Mock + Golden 自测（无需 NPU）
bash run.sh                      # NPU 真机执行 L0+L1 全量用例
bash run.sh --suite=L0           # 仅跑 L0 用例
```

### 5.3 运行 aclnn 调用示例

```bash
cd ops/apply_proximal_gradient_descent/examples
bash run.sh
```

示例会构造一个 `[2, 3]` 的 FP32 `var/delta`，以 `alpha=0.01, l1=0.001, l2=0.01` 调用一次 `aclnnApplyProximalGradientDescent`，并把 NPU 输出与 CPU Golden 打印对比。

### 5.4 PyTorch ST

参见 [tests/st/torch/README.md](tests/st/torch/) 下的 PyTorch 适配层与 L0+L1 精度用例。

## 6. 目录结构

```text
ops/apply_proximal_gradient_descent/
├── README.md                     # 本文件
├── build.sh                      # 一键构建 + 安装脚本
├── CMakeLists.txt                # 算子工程 CMake
├── docs/                         # 需求/设计/接口文档
│   ├── REQUIREMENTS.md
│   ├── DESIGN.md
│   ├── TEST_DESIGN.md
│   ├── TEST_CASES.md
│   ├── PLAN.md
│   ├── LOG.md
│   ├── precision-report.md
│   └── aclnnApplyProximalGradientDescent.md
├── op_host/                      # Host 侧 (InferShape / Tiling / OpDef)
├── op_kernel/                    # Device 侧 Kernel 实现
├── op_api/                       # aclnn 两段式封装
├── examples/                     # aclnn 调用示例（本阶段产出）
│   ├── test_aclnn_apply_proximal_gradient_descent.cpp
│   ├── CMakeLists.txt
│   └── run.sh
├── tests/
│   ├── ut/                       # UT 测试 (op_host / op_api / op_kernel)
│   └── st/                       # ST 测试
│       ├── test_aclnn_apply_proximal_gradient_descent.cpp  # C++ L0+L1
│       ├── CMakeLists.txt
│       ├── run.sh
│       ├── torch/                # PyTorch L0+L1 精度用例
│       └── testcases/
├── probe/                        # Kernel 直调穿刺工程
├── tools/                        # 辅助脚本
└── issues/                       # 问题记录
```

## 7. 相关文档

- [需求文档](docs/REQUIREMENTS.md)
- [详细设计](docs/DESIGN.md)
- [测试设计](docs/TEST_DESIGN.md)
- [测试用例](docs/TEST_CASES.md)
- [aclnn 接口文档](docs/aclnnApplyProximalGradientDescent.md)
- [最终精度验收报告](docs/precision-report.md)
- [开发日志](docs/LOG.md)
