# 使用Ascend C和PyTorch Extension能力开发自定义NPU算子指南

## 概述

本文档介绍如何使用Ascend C和[PyTorch Extension](https://docs.pytorch.org/tutorials/extension.html)能力开发自定义NPU算子。通过PyTorch Extension机制，可以将Ascend C开发的算子封装成PyTorch可调用的扩展包，实现与PyTorch原生操作一致的使用体验。

**核心优势:**

- **单交付件：** 一个文件完成算子开发和PyTorch框架适配。
- **高效调用：** 使用`<<<>>>`语法启动核函数，流程简单高效。
- **PyTorch集成：** 安装后可像调用PyTorch操作一样调用NPU算子。

## 环境要求

- 请先参考[环境部署](../install/quick_install.md)完成基础环境搭建，注意以下基础库依赖的版本。
- python >= 3.8.0
- gcc >= 9.4.0
- PyTorch >= 2.6.0
- 对应版本的[torch_npu](https://gitcode.com/Ascend/pytorch/releases)

## 算子开发流程

### 算子目录结构

在`ops-nn/experimental`目录下创建算子，目录结构如下：

```
experimental/
└── ${op_class}/                # 算子类型目录
    └── ${op_name}/             # 算子名称目录
        ├── tests/              # 测试文件目录（可选）
        ├── ${dirs}/            # 自定义目录（可选）
        ├── ${op_name}.cpp      # 算子实现文件
        ├── ${op_name}.py       # Python接口文件（可选）
        └── CMakeLists.txt      # 编译配置文件
```

### 算子实现文件

算子实现文件需要包含以下四个部分：

1. **算子SchemaSchema注册**
2. **算子Meta Function实现（InferShape + InferDtype）**
3. **算子Kernel实现（Ascend C）**
4. **算子NPU调用实现**

示例代码如下：

```cpp
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <type_traits>

namespace ascend_ops {
namespace Add {

/**
 * 算子Schema注册
 * 告诉PyTorch框架有这样一个算子
 */
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("add(Tensor x, Tensor y) -> Tensor");
}

/**
 * 算子Meta函数实现
 * 根据输入推导输出的shape和dtype
 */
torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
{
    TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
    auto z = torch::empty_like(x);
    return z;
}

/**
 * 注册Meta函数实现
 */
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("add", add_meta);
}

/**
 * NPU算子Kernel实现
 * 使用Ascend C API编写
 */
template <typename T>
__global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    // Kernel实现代码
}

/**
 * 算子NPU调用实现算子NPU调用实现
 */
torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto z = add_meta(x, y);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t totalLength, numBlocks, blockLength, tileSize;
    totalLength = x.numel();
    std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(totalLength);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto z_ptr = (GM_ADDR)z.data_ptr();
    auto acl_call = [=]() -> int {
        AT_DISPATCH_SWITCH(
            x.scalar_type(), "add_npu",
            AT_DISPATCH_CASE(torch::kFloat32, [&] {
                using scalar_t = float;
                add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
            })
            AT_DISPATCH_CASE(torch::kFloat16, [&] {
                using scalar_t = half;
                add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
            })
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Add", acl_call);
    return z;
}

/**
 * 注册NPU调用实现
 */
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("add", add_npu);
}

}  // namespace Add
}  // namespace ascend_ops
```

### CMakeLists.txt配置

在算子名称目录下创建`CMakeLists.txt`文件：

```cmake
add_sources()
# 若需指定自定义源文件列表，使用 add_sources("file1.cpp;file2.cpp")
```

## 编译与打包

### 安装依赖

```
build
pyyaml
numpy<2
pytest
```

### 编译命令

在`ops-nn`目录下执行以下命令进行编译打包：

```bash
bash build.sh --pkg --experimental --soc=${soc} --ops=${op1,op2}
```

**参数说明：**

- `--pkg`: 打包模式
- `--experimental`: 编译experimental目录下的算子
- `--soc`: 指定NPU架构（可选），默认为`ascend910b`，当前支持`ascend910b、ascend910_93、ascend950`
- `--ops`: 指定要编译的算子列表（可选），多个算子用逗号分隔

**示例：**

```bash
# 编译所有experimental算子，使用默认架构ascend910b
bash build.sh --pkg --experimental

# 编译指定算子
bash build.sh --pkg --experimental --ops=add,mul

# 指定NPU架构
bash build.sh --pkg --experimental --soc=ascend910b
```

### 编译产物

编译成功后，在`build_out`目录下会生成whl包，文件名格式为：

```
ascend_ops_nn-1.0.0-${python_version}-abi3-${arch}.whl
```

其中：
- `${python_version}`: Python版本（如cp38表示Python 3.8）
- `${arch}`: CPU架构

## 安装与使用

### 安装whl包

```bash
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

### 调用算子

安装完成后，可以像调用PyTorch原生操作一样调用NPU算子：

```python
import torch
import torch_npu
import ascend_ops_nn

# 初始化NPU数据
x = torch.randn(10, 32, dtype=torch.float32).npu()
y = torch.randn(10, 32, dtype=torch.float32).npu()

# 调用自定义NPU算子
result = torch.ops.ascend_ops_nn.add(x, y)

# 验证结果
cpu_x = x.cpu()
cpu_y = y.cpu()
cpu_result = cpu_x + cpu_y

assert torch.allclose(cpu_result, result.cpu(), rtol=1e-6)
print("验证成功！")
```

> [!NOTE] 注意
>
> 不同芯片型号的代码需要使用宏进行隔离，否则会导致whl包编译失败