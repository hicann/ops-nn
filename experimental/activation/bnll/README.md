# BNLL

## 产品支持情况

| 产品型号 | 是否支持 |
| :--- | :---: |
| Ascend 950PR / Ascend 950DT | √ |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

BNLL (Binomial Normal Log Likelihood) 算子实现数值稳定版本的 softplus 激活函数。

计算公式:

$$
y_i = \begin{cases} x_i + \ln(1 + e^{-x_i}) & \text{if } x_i > 0 \\ \ln(1 + e^{x_i}) & \text{otherwise} \end{cases}
$$

上述分段公式等价于 $y = \ln(1 + e^x)$，采用分段计算以保证数值稳定性，避免指数运算溢出。

## 参数说明

### 输入

| 参数名 | 数据类型 | 数据格式 | 维度范围 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| x | FLOAT, FLOAT16, BFLOAT16 | ND | 0-8 | 输入 Tensor，支持空 Tensor 和非连续 Tensor |

### 输出

| 参数名 | 数据类型 | 数据格式 | 维度范围 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| y | FLOAT, FLOAT16, BFLOAT16 | ND | 0-8 | 输出 Tensor，shape 和 dtype 与输入 x 一致 |

## 约束说明

- x 和 y 的数据类型必须一致，仅支持 FLOAT、FLOAT16、BFLOAT16。
- y 的 shape 必须与 x 完全一致（逐元素算子，无广播语义）。
- x 支持空 Tensor（0 元素），此时 y 也为空 Tensor，直接返回成功。
- 输入维度范围 0-8（支持 0 维标量 Tensor）。
- 仅支持 ND 数据格式。
- 确定性计算：BNLL 为逐元素算子，相同输入必然产生相同输出。
- 目标平台仅为 Ascend950 (arch35)。

## 调用说明

### ACLNN 调用（两段式接口）

BNLL 算子通过 aclnn 两段式接口调用：

```cpp
#include "aclnn_bnll.h"

// 第一段：获取 workspace 大小和执行器
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnBnllGetWorkspaceSize(x, y, &workspaceSize, &executor);

// 分配 workspace（如需要）
void* workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 第二段：执行计算
aclnnBnll(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

完整调用示例请参考 [examples/test_aclnn_bnll.cpp](examples/test_aclnn_bnll.cpp)。

### GE 图模式调用（单算子模型执行）

通过 GE IR 的 `aclopExecuteV2` 接口执行，需先使用 ATC 工具将算子描述文件编译为 om 模型文件。

算子原型定义（注册名为 "Bnll"）：

```text
REG_OP(BNLL)
    .INPUT(x, TensorType::FloatingDataType())
    .OUTPUT(y, TensorType::FloatingDataType())
    .OP_END_FACTORY_REG(BNLL)
```

算子描述文件（JSON）示例：

```json
[
    {
        "op": "Bnll",
        "input_desc": [
            { "name": "x", "param_type": "required", "format": "ND", "shape": [4, 256], "type": "float32" }
        ],
        "output_desc": [
            { "name": "y", "param_type": "required", "format": "ND", "shape": [4, 256], "type": "float32" }
        ]
    }
]
```

编译 om 模型：

```bash
atc --singleop=bnll.json --output=. --soc_version=Ascend950DT_9576
```

完整调用示例请参考 [test_geir_bnll.cpp](./examples/test_aclnn_bnll.cpp)。

## 工程结构

```text
bnll/
├── CMakeLists.txt              # 构建配置
├── README.md                   # 本文档
├── build.sh                    # 构建脚本
├── examples/                   # 调用示例
│   ├── CMakeLists.txt          # 示例构建配置
│   ├── run.sh                  # 示例构建运行脚本
│   ├── bnll.json               # GE IR 算子描述文件
│   ├── test_aclnn_bnll.cpp     # aclnn 调用示例
│   └── test_geir_bnll.cpp      # GE IR 调用示例
├── op_host/                    # Host 侧实现
│   ├── bnll_def.cpp            # 算子原型注册
│   ├── bnll_infershape.cpp     # Shape 推导
│   └── arch35/
│       └── bnll_tiling.cpp     # Tiling 实现
├── op_kernel/                  # Kernel 侧实现
│   ├── bnll_arch35.cpp         # Kernel 入口
│   └── arch35/
│       ├── bnll.h              # Kernel 算子类
│       ├── bnll_tiling_key.h   # TilingKey 模板参数
│       └── bnll_tiling_data.h  # TilingData 结构体
├── op_api/                     # ACLNN 接口
│   ├── aclnn_bnll.h            # L2 API 头文件
│   ├── aclnn_bnll.cpp          # L2 API 实现
│   ├── bnll.h                  # L0 API 头文件
│   └── bnll.cpp                # L0 API 实现
└── tests/                      # 测试代码
    ├── ut/                     # 单元测试
    └── st/                     # 系统测试
```

## 构建与运行

### 环境准备

```bash
# 设置 CANN 环境变量
source ${INSTALL_DIR}/set_env.sh
```

### 构建算子包

```bash
bash build.sh --soc=ascend950
```

### 运行示例

```bash
# 运行 aclnn 示例（需要 NPU 设备）
bash build.sh -e

# 或者手动运行
cd examples
./run.sh --eager    # aclnn 调用示例
./run.sh --geir     # GE IR 调用示例
./run.sh --all      # 运行全部示例
```

### 运行测试

```bash
bash build.sh -u    # 运行单元测试
bash build.sh -s    # 运行系统测试（需要 NPU 设备）
bash build.sh -a    # 运行全部测试
```
