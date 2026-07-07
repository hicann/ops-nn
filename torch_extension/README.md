# cann_ops_nn

面向 Ascend NPU 的高性能算子扩展库，通过 JIT 编译将 PyTorch 接口与 ACLNN 算子库桥接。

## 构建与安装

### 前置条件

- 操作系统：Linux
- Python：3.8+
- 编译器：GCC 9.4.0+
- 框架：PyTorch>=2.6.0、torch_npu（需匹配 PyTorch 版本）
- 工具包：Ascend CANN Toolkit

### 安装步骤

1. 安装依赖：

    ```sh
    python3 -m pip install -r requirements.txt
    ```

2. 构建 Wheel 包：

    ```sh
    # -n: non-isolated build（使用当前环境已有依赖）
    python3 -m build --wheel -n
    ```

3. 安装包：

    ```sh
    python3 -m pip install dist/*.whl --force-reinstall --no-deps
    ```

## 快速入门

```python
import torch
import torch_npu
import cann_ops_nn

# 初始化 NPU 张量
x1 = torch.randn(16, 32, dtype=torch.float16).npu()
x2 = torch.randn(32, 16, dtype=torch.float16).npu()

# 调用自定义 NPU 算子
result = torch.ops.cann_ops_nn.{ops}(x1, x2)

# 验证结果
cpu_result = torch.{ops}(x1.cpu(), x2.cpu())
assert torch.allclose(result.cpu(), cpu_result, rtol=1e-3)
print("验证成功！")
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
