# cann_ops_nn

面向 Ascend NPU 的高性能算子扩展库，通过 JIT 编译将 PyTorch 接口与 ACLNN 算子库桥接。

## 构建与安装

### 前置条件

- 操作系统：Linux
- Python：3.8+
- 编译器：GCC 9.4.0+
- 框架：PyTorch>=2.6.0、torch_npu（需匹配 PyTorch 版本）
- 工具包：Ascend CANN Toolkit

### 构建 Wheel 包

支持两种构建方式：通过 `build.sh` 构建和直接使用 `python3 -m build` 构建。

#### 方式一：通过 build.sh 构建（推荐）

```sh
# 构建整包（包含所有算子）
bash build.sh --torch_extension

# 构建单算子包（仅包含指定算子）
bash build.sh --torch_extension --ops=swiglu_group --vendor_name=custom

# 构建多算子包
bash build.sh --torch_extension --ops=swiglu_group,swiglu_group_quant --vendor_name=custom
```

构建完成后，wheel 包会自动复制到 `build_out/` 目录。

**参数说明：**

| 参数 | 说明 |
| --- | --- |
| `--torch_extension` | 仅构建 torch_extension wheel 包，不执行 cmake 编译 |
| `--ops=op1,op2,...` | 指定编译的算子名（逗号分隔），不指定则编译所有算子 |
| `--vendor_name=name` | 指定子包名，用于子包命名和隔离 |

**包命名规则：**

| 场景 | 包名 | 安装目录 |
| --- | --- | --- |
| 整包 | `cann_ops_nn` | `cann_ops_nn/` |
| 单算子/多算子包 | `cann_ops_nn_<vendor>` | `cann_ops_nn_<vendor>/` |

#### 方式二：直接使用 python3 -m build 构建

```sh
# 安装依赖
python3 -m pip install -r requirements.txt

# 构建整包
python3 -m build --wheel -n

# 构建单算子包
TORCH_EXTENSION_OPS=swiglu_group TORCH_EXTENSION_VENDOR=custom python3 -m build --wheel -n
```

### 安装

```sh
# 安装整包
python3 -m pip install dist/cann_ops_nn-1.0.0-*.whl --no-deps

# 安装单算子包
python3 -m pip install dist/cann_ops_nn_myvendor-1.0.0-*.whl --no-deps
```

### 整包与子包共存机制

整包和单算子包可以同时安装，互不冲突：

- **整包**安装到 `cann_ops_nn/` 目录，包含所有算子。
- **单算子包**安装到 `cann_ops_nn_<vendor>/` 目录，与整包物理隔离。
- 单算子包通过 **entry point** 机制注册算子，优先级高于整包。用户调用 `cann_ops_nn.swiglu_group(x)` 时，若子包已安装则使用子包的算子实现。
- 卸载单算子包后，整包的同名算子自动接管。

```sh
# 安装整包
pip install cann_ops_nn-1.0.0-*.whl --no-deps

# 安装单算子包（覆盖整包中的同名算子）
pip install cann_ops_nn_myvendor-1.0.0-*.whl --no-deps

# 卸载单算子包（整包算子自动恢复）
pip uninstall cann-ops-nn-myvendor
```

## 目录结构

```
├── torch_extension
│   ├── build.sh                              # 构建脚本（支持 --torch_extension 参数）
│   ├── setup.py                              # wheel 打包配置（支持单算子编包）
│   ├── requirements.txt
│   ├── cann_ops_nn                           # 整包安装目录
│   │   ├── __init__.py                       # 包入口，from .ops import *
│   │   ├── op_builder/
│   │   │   └── builder.py                    # OpBuilder 基类，JIT 编译管理
│   │   ├── common/
│   │   │   └── aclnn_common.h                # ACLNN_CMD 宏等公共能力
│   │   ├── csrc/
│   │   │   └── <category>/<op>.cpp           # C++ kernel wrapper
│   │   ├── ops/
│   │   │   ├── __init__.py                   # 自动发现算子 + entry point 加载
│   │   │   └── <category>/<op>/<op>.py       # Python 前端
│   │   └── docs/
│   │       └── torch_extension_guidelines.md # 开发规范
│   └── dist/                                 # 构建产物输出目录
├── <category>/<op>/torch_extension/           # 分布式算子（仓库根目录下）
│   ├── <op>.py
│   ├── __init__.py
│   └── csrc/<op>.cpp
└── build_out/                                # build.sh 输出目录
```

## 快速入门

```python
import torch
import torch_npu
import cann_ops_nn

# 初始化 NPU 张量
x = torch.randn(16, 32, dtype=torch.float16).npu()

# 调用算子（两种方式等价）
result = cann_ops_nn.swiglu_group(x)
result = cann_ops_nn.ops.swiglu_group(x)

print(result.shape)  # torch.Size([16, 16])
```



## 开发者指南：新增算子

以添加新算子 `new_operator` 为例，需提供 C++ 绑定和 Python 构建器。

### 1. C++ 后端 (`cann_ops_nn/csrc/<op_category>/new_operator.cpp`)

该文件将 PyTorch 张量桥接到 ACLNN C-API。

```cpp
#include <torch/extension.h>
#include "aclnnop/aclnn_new_operator.h"
#include "../common/aclnn_common.h"

namespace cann_ops_nn {
namespace <op_category> {

at::Tensor new_operator(
    const at::Tensor& input1,
    const at::Tensor& input2,
    int64_t param1,
    const std::string& param2)
{
    // 设备检查
    TORCH_CHECK(input1.device().type() == at::kPrivateUse1, "input1 must be on NPU device");
    TORCH_CHECK(input2.device().type() == at::kPrivateUse1, "input2 must be on NPU device");

    // 输出创建
    at::Tensor out = at::empty(output_shape,
        at::TensorOptions().dtype(output_dtype).device(at::kPrivateUse1));

    // ACLNN 调用（宏内部自动处理类型转换和 workspace）
    ACLNN_CMD(aclnnNewOperator, input1, input2, out);

    return out;
}

}  // namespace <op_category>
}  // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("new_operator", &cann_ops_nn::<op_category>::new_operator,
          "NewOperator on NPU");
}
```

### 2. Python 前端 (`cann_ops_nn/ops/<op_category>/new_operator.py`)

该文件管理 JIT 编译逻辑并将算子注册到 PyTorch Dispatcher。

```python
import torch
import torch_npu
from torch.library import impl
from cann_ops_nn.op_builder import OpBuilder, get_as_library

class NewOperatorOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("new_operator")

    def sources(self):
        """C++ 源码路径。"""
        return ['csrc/<op_category>/new_operator.cpp']

    def schema(self) -> str:
        """PyTorch 算子签名。"""
        return "new_operator(Tensor input1, Tensor input2, int param1=0, str param2=\"\") -> Tensor"

    def register_meta(self):
        """
        注册 Meta 实现（形状/类型推导）。
        对 Autograd 和 FakeTensor 支持至关重要。
        """
        @impl(get_as_library(), self.name, "Meta")
        def new_operator_meta(input1, input2, param1=0, param2=""):
            return torch.empty_like(input1)

# 实例化构建器
builder = NewOperatorOpBuilder()

@impl(get_as_library(), builder.name, "PrivateUse1")
def new_operator(input1, input2, param1=0, param2=""):
    """
    Dispatcher 的 NPU 实现。
    'PrivateUse1' 是自定义 NPU 后端的分发键。
    """
    op_module = builder.load()  # 编译/加载 .so 文件
    return op_module.new_operator(input1, input2, param1, param2)
```

### 技术说明

| 组件 | 职责 |
| --- | --- |
| **OpBuilder** | 使用 `ninja` 处理 C++ 源码的 JIT 编译 |
| **Meta 分发** | 允许 PyTorch 在不运行 NPU 代码的情况下推导输出形状/类型 |
| **PrivateUse1** | PyTorch 路由 NPU 特定操作使用的后端分发键 |
| **ACLNN_CMD 宏** | 自动处理 `at::Tensor` → `aclTensor*` 类型转换、workspace 申请/释放、流调度 |
