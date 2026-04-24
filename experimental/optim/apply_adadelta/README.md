# ApplyAdadelta

## 算子说明

执行 Adadelta 优化器的单步参数更新。根据当前梯度、梯度平方累积和更新量平方累积，计算参数更新量并**原地**更新权重参数（inplace 语义）。对标 TensorFlow `tf.raw_ops.ApplyAdadelta` 接口和 PyTorch `torch.optim.Adadelta` 的计算语义。

### 计算公式

$$
\begin{aligned}
accum_{t} &= \rho \cdot accum_{t-1} + (1 - \rho) \cdot grad^2 \\
update &= \frac{\sqrt{accum\_update_{t-1} + \epsilon}}{\sqrt{accum_{t} + \epsilon}} \cdot grad \\
var_{t} &= var_{t-1} - lr \cdot update \\
accum\_update_{t} &= \rho \cdot accum\_update_{t-1} + (1 - \rho) \cdot update^2
\end{aligned}
$$

其中：
- `var`：待更新的权重参数
- `accum`：梯度平方的指数移动平均（$E[g^2]$）
- `accum_update`：参数更新量平方的指数移动平均（$E[\Delta x^2]$）
- `lr`：学习率
- `rho`：衰减系数，取值范围 [0, 1)
- `epsilon`：数值稳定常数，必须 > 0
- `grad`：当前梯度

## 产品支持情况

| 产品 | 是否支持 |
|------|:------:|
| Ascend 950PR / Ascend 950DT（Atlas A5 推理系列，DAV_3510） | √ |
| Atlas A3 训练系列 / Atlas A3 推理系列 | × |
| Atlas A2 训练系列 / Atlas A2 推理系列 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列 | × |
| Atlas 训练系列 | × |

## 调用方式支持

| 调用方式 | 是否支持 |
|---------|:--------:|
| ACLNN 单算子调用 | √ |
| torch_npu 单算子调用 | × |
| torch.compile 入图 | × |
| GE 图模式（静态 shape） | × |
| GE 图模式（动态 shape） | × |

> **说明**：本算子仅支持 ACLNN 单算子调用，暂不支持 GE 图模式、torch_npu 单算子及 torch.compile 入图。因此仅提供 ACLNN 调用示例 `test_aclnn_apply_adadelta.cpp`，**不提供 GE IR 示例**。

## 数据类型

| 参数 | 支持 dtype |
|-----|-----------|
| var / accum / accumUpdate / grad | FLOAT16、FLOAT |
| lr / rho / epsilon (aclScalar) | FLOAT16、FLOAT（需与 Tensor dtype 一致） |

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|-------|----------|------|
| var | 输入+输出 (inplace) | 权重参数 Tensor，dtype ∈ {FLOAT16, FLOAT}，1-8 维 ND 格式 |
| accum | 输入+输出 (inplace) | 梯度平方累积 Tensor，shape/dtype 必须与 var 一致 |
| accumUpdate | 输入+输出 (inplace) | 更新量平方累积 Tensor，shape/dtype 必须与 var 一致 |
| lr | 输入 | 学习率标量，dtype 必须与 Tensor 一致 |
| rho | 输入 | 衰减系数标量，取值范围 [0, 1) |
| epsilon | 输入 | 数值稳定常数标量，必须 > 0 |
| grad | 输入 | 梯度 Tensor，shape/dtype 必须与 var 一致 |
| varOut | 输出 | 与 var 共享 Device 内存 |
| accumOut | 输出 | 与 accum 共享 Device 内存 |
| accumUpdateOut | 输出 | 与 accumUpdate 共享 Device 内存 |

详细接口参数与返回码定义请参见 [aclnnApplyAdadelta 接口文档](docs/aclnnApplyAdadelta.md)。

## 约束说明

- 本算子仅支持 Ascend 950PR / Ascend 950DT 产品（Atlas A5 推理系列，DAV_3510 架构）。
- `var`、`accum`、`accumUpdate`、`grad` 的 shape 必须**完全相同**，不支持广播。
- `var`、`accum`、`accumUpdate`、`grad` 的 dtype 必须**完全一致**，`lr`、`rho`、`epsilon` 的 dtype 也需与之一致。
- 不支持 0 维标量 Tensor 作为 var / accum / accumUpdate / grad。
- 参数值域：`epsilon > 0`，`rho ∈ [0, 1)`。
- **Inplace 语义**：`varOut / accumOut / accumUpdateOut` 分别与 `var / accum / accumUpdate` 共享 GM 地址，输出直接写回输入内存。调用方需自行保存调用前的原始值（如需）。
- 空 Tensor 处理：元素数为 0 时直接返回成功，不执行计算。
- FP16 精度路径：Kernel 内部提升为 FP32 计算，输出时转回 FP16，以保证精度。
- 确定性：本算子为逐元素运算，无 Reduce，相同输入产出相同输出。

## 目录结构

```
apply_adadelta/
├── CMakeLists.txt
├── README.md
├── build.sh
├── docs/
│   ├── aclnnApplyAdadelta.md        # aclnn 接口说明
│   ├── REQUIREMENTS.md              # 需求文档
│   ├── DESIGN.md                    # 详细设计
│   ├── TEST.md                      # 测试设计
│   ├── precision-report.md          # 最终精度验收报告
│   └── performance-report.md        # 性能验收报告
├── examples/
│   ├── CMakeLists.txt
│   └── test_aclnn_apply_adadelta.cpp  # aclnn 调用示例
├── op_host/
├── op_kernel/
├── op_api/
└── tests/
    ├── ut/
    └── st/
```

## 调用示例

本算子提供 ACLNN 调用示例，样例代码位于 [`examples/test_aclnn_apply_adadelta.cpp`](examples/test_aclnn_apply_adadelta.cpp)。

### 编译运行

前置条件：已安装 CANN Toolkit，并已编译安装本算子自定义包（生成 `opp/vendors/apply_adadelta_custom/`）。

```bash
# 1. 设置环境
source ${ASCEND_HOME_PATH}/bin/setenv.bash

# 2. 编译示例
cd examples
mkdir -p build && cd build
cmake ..
make

# 3. 运行示例
./test_aclnn_apply_adadelta
```

### 示例输出

示例以 shape=[4,2] 的 fp32 张量、`lr=0.01, rho=0.9, epsilon=1e-6` 执行一次 Adadelta 更新，典型输出：

```
var_out[0]=0.998526  accum_out[0]=0.115000  accum_update_out[0]=0.011174
var_out[1]=2.000976  accum_out[1]=0.189000  accum_update_out[1]=0.018952
...
```

> **不支持 GE 图模式**：如前述调用方式支持矩阵所述，本算子**不支持 GE 图模式**，因此不提供 `test_geir_apply_adadelta.cpp` 示例。

具体编译和运行过程可参考 [编译与运行样例](../../../docs/context/编译与运行样例.md)。
