# torch_extension开发规范

本文档约定`cann_ops_nn`（torch_extension）新增/修改算子api时的目录组织、命名、各层实现以及文档编写规范。开发者新增算子前请先通读本规范，并参考已有算子（如`swiglu_group`）作为模板。

`cann_ops_nn`通过PyTorch的JIT（`torch.utils.cpp_extension.load`）在首次调用时即时编译C++ kernel wrapper，把PyTorch的函数接口桥接到CANN的aclnn接口，同时通过GE Converter支持torchair图模式。一个完整的算子api通常由「Python前端、C++后端、torchair图模式Converter、文档」四部分组成。

## 1. 算子api目录组织规范

nn仓的算子采用**分布式目录结构**：每个算子的torch_extension文件放在仓库根目录下对应的算子目录中，`setup.py`在构建时自动收集并staging到whl包内。以算子`${op_api}`（属于`${category}`分类）为例：

```
├── ${category}/
│   └── ${op_api}/
│       └── torch_extension/                    # 算子的 torch_extension 目录
│           ├── __init__.py                     # 对外导出算子接口与Converter
│           ├── ${op_api}.py                    # Python前端实现（OpBuilder、schema、meta、对外函数）
│           ├── graph_convert_${op_api}.py      # torchair图模式（GE Converter）实现
│           └── csrc/
│               └── ${op_api}.cpp               # C++ kernel wrapper，调用aclnn接口
├── torch_extension/
│   ├── cann_ops_nn/                            # whl包安装目录（构建时自动staging）
│   │   ├── __init__.py                         # 包根入口
│   │   ├── op_builder/
│   │   │   └── builder.py                      # OpBuilder基类，JIT编译、schema/meta注册（一般无需修改）
│   │   ├── common/
│   │   │   └── aclnn_common.h                  # ACLNN_CMD宏、类型转换等公共能力（一般无需修改）
│   │   ├── csrc/                               # C++源码staging目录（构建时自动收集）
│   │   │   └── ${category}/
│   │   │       └── ${op_api}.cpp
│   │   └── ops/
│   │       ├── __init__.py                     # 算子自动发现与加载
│   │       └── ${category}/
│   │           ├── __init__.py                 # 分类入口（从源码复制）
│   │           └── ${op_api}/
│   │               ├── __init__.py             # 算子入口（导入接口与Converter）
│   │               ├── ${op_api}.py
│   │               └── graph_convert_${op_api}.py
│   ├── setup.py                                # wheel打包配置（自动收集分布式算子）
│   └── build.sh                                # 构建脚本
└── build_out/                                  # whl产物输出目录
```

新增一个算子api的标准动作清单（以`swiglu_group`为例）：

1. 在`${category}/${op_api}/torch_extension/csrc/${op_api}.cpp`中实现C++ kernel wrapper，调用`ACLNN_CMD`拉起aclnn接口；
2. 在`${category}/${op_api}/torch_extension/${op_api}.py`中编写`OpBuilder`子类（定义`sources`/`schema`/`register_meta`），注册dispatcher实现，并提供对外的Python函数；
3. 在`${category}/${op_api}/torch_extension/graph_convert_${op_api}.py`中编写图模式Converter（若需支持图模式）；
4. 在`${category}/${op_api}/torch_extension/__init__.py`中导出对外接口与Converter；
5. 在`${category}/__init__.py`中导出新增算子（使源码`__init__.py`在构建时被复制到whl中）；
6. 在`docs/zh/${op_api}.md`中补充算子文档（如有）。

>新增文件放在算子目录下的`torch_extension/`中，`setup.py`构建时自动收集。import路径统一以`cann_ops_nn`为根。

## 2. 命名规范

### 2.1 API命名

一个算子从schema注册到对外导出，涉及多个层级的命名，需保持一致且各司其职。**对外api接口及算子名一律不带`npu_`前缀**，直接采用算子语义的小写蛇形名（如`swiglu_group`）：

| 层级 | 命名约定 | 示例 |
| --- | --- | --- |
| Library名（DEF域） | 固定为`cann_ops_nn` | `Library("cann_ops_nn", "DEF")` |
| schema算子名 / aten注册名 | 算子语义的小写蛇形名，不带`npu_`前缀 | `swiglu_group` |
| C++ wrapper函数名 | 与schema算子名一致，置于`namespace cann_ops_nn::category`内 | `cann_ops_nn::activation::swiglu_group` |
| `PYBIND11_MODULE`导出名 | 与schema算子名一致 | `m.def("swiglu_group", &swiglu_group, "SwigluGroup on NPU")` |
| Meta实现函数名 | schema算子名 + `_meta`后缀 | `swiglu_group_meta` |
| PrivateUse1 dispatcher函数名 | 下划线前缀 + schema算子名 | `_swiglu_group` |
| OpBuilder子类名 | 算子名的大驼峰 + `OpBuilder`后缀 | `SwigluGroupOpBuilder` |
| 对外Python接口名 | 用户直接调用的函数名，体现使用语义，不带`npu_`前缀 | `swiglu_group` |
| 图模式GE op函数名 | 与GE算子`op_type`一致的大驼峰 | `SwigluGroup` |
| 图模式Converter函数名 | `convert_` + schema算子名 | `convert_swiglu_group` |

命名要点：

- **不带`npu_`前缀**：对外算子名与api接口统一使用算子语义名（小写蛇形），不加`npu_`等后端前缀；schema名、C++函数名、pybind导出名三者必须与该名字完全一致，否则JIT编译产物无法被正确调用。
- **接口名体现语义**：对外函数名应贴近业务语义。无论是纯透传aclnn接口的算子（如`swiglu_group`），还是封装了结构体构造、参数整理等额外逻辑的接口，均采用语义化命名。
- **aclnn接口名独立**：底层aclnn接口沿用CANN命名（大驼峰，如`aclnnSwigluGroup`），与对外算子名解耦；C++ wrapper内通过`ACLNN_CMD(aclnnSwigluGroup, ...)`调用。
- **版本后缀**：同一算子的不同迭代版本以`_v2`、`_v3`等后缀区分，schema名、文件名、Converter名需同步带上版本后缀（如`rms_norm_v2`、`graph_convert_rms_norm_v2.py`）。
- **辅助/工具接口**：与主算子配套的工具函数采用动宾语义命名，如`get_swiglu_group_workspace_size`。

### 2.2 文件命名

- 统一使用**小写蛇形命名**（snake_case），单词以`_`连接，禁止使用大写、驼峰或连字符。
- 同一算子的各层文件**主名保持一致**，仅靠目录和前缀区分职责：
  - Python前端：`${op_api}.py`，如`swiglu_group.py`；
  - C++后端：`csrc/${op_api}.cpp`，主名与Python前端一致，如`swiglu_group.cpp`；
  - 图模式：`graph_convert_${op_api}.py`，统一加`graph_convert_`前缀，如`graph_convert_swiglu_group.py`；
  - 算子入口：`__init__.py`，导出算子接口与Converter。
- 文件主名应与该文件主要导出的算子语义对应；带版本的算子文件名需带版本后缀（如`rms_norm_v2.cpp`）。
- 公共头文件放在`common/`下，按能力域命名（如`aclnn_common.h`）。

### 2.3 标识符命名

#### Python标识符

- **函数/变量/参数**：小写蛇形（snake_case），如`clamp_limit`、`group_index`、`dst_type`。
- **类名**：大驼峰（PascalCase），如`OpBuilder`、`SwigluGroupOpBuilder`。
- **模块级常量**：全大写蛇形（UPPER_SNAKE_CASE），如`BLOCK_FP8_QUANT_MODE`、`MX_BLOCK_SIZE`。
- **模块内部私有符号**：以单下划线`_`前缀标识，如`_swiglu_group`、`_TORCHAIR_AVAILABLE`。
- **类型注解**：对外接口与关键内部函数应带类型注解（`from typing import Optional, Tuple, List`），可选参数统一用`Optional[...]`，例如：
  ```python
  def swiglu_group(
      x: torch.Tensor,
      *,
      weight: Optional[torch.Tensor] = None,
      group_index: Optional[torch.Tensor] = None,
      clamp_limit: float = -1.0,
  ) -> torch.Tensor:
  ```
- **参数命名一致性**：同一算子在schema、meta、dispatcher、对外函数、Converter中的同义参数应使用相同的名字（如`clamp_limit`、`group_index`），避免在不同层出现不一致写法。

#### C++标识符

- **函数/局部变量/参数**：小写蛇形，如`clamp_limit`、`group_index`、`k_split_factor`。
- **命名空间**：算子实现统一置于`namespace cann_ops_nn::${category}`内（如`namespace cann_ops_nn::activation`）。
- **常量**：`const`/`constexpr`常量使用`k`前缀的大驼峰或全大写蛇形，如`constexpr int64_t kSplitFactor = 2;`。
- **类型别名/结构体**：大驼峰，如`TensorWrapper`、`TensorListWrapper`。
- **入参类型约定**：
  - 必选Tensor用`const at::Tensor&`；
  - 可选Tensor用`const c10::optional<at::Tensor>&`；
  - 整型属性用`int64_t`，可选整型属性用`c10::optional<int64_t>`；浮点属性用`double`；字符串属性用`std::string`。

#### Schema标识符（算子签名）

- 参数名采用小写蛇形，与Python/C++层一致。
- 用`*`分隔位置参数与关键字参数：`*`之前为必选的位置参数，之后为可选的关键字参数（带默认值）。
- 可选参数以`?`标注并给出默认值，如`Tensor? weight=None`、`float clamp_limit=-1.0`。
- 多输出用元组表示，如`-> (Tensor, Tensor)`。

  以`swiglu_group`为例的schema：

  ```
  swiglu_group(Tensor x, *, Tensor? weight=None, Tensor? group_index=None, float clamp_limit=-1.0) -> Tensor
  ```

## 3. 各层实现规范

### 3.1 C++后端（`csrc/${op_api}.cpp`）

负责把PyTorch张量桥接到aclnn C-API，规范要点：

1. 文件头部包含`#include <torch/extension.h>`与`#include "aclnn_common.h"`，实现置于`namespace cann_ops_nn`和`namespace ${category}`内。
2. 函数签名与schema严格对应：必选/可选参数类型按[2.3 C++入参类型约定](#c标识符)选择。
3. **入参校验**：使用`TORCH_CHECK(cond, msg...)`校验shape、dtype、维度、取值范围等，错误信息要可读且包含实际值，例如：
   ```cpp
   TORCH_CHECK(torch_npu::utils::is_npu(tensor), "tensor must be on NPU device");
   TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
   TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
               "x dtype must be float16 or bfloat16, but got ", x.scalar_type());
   ```
4. **输出张量手动申请**：按meta推导的shape/dtype用`at::empty(...)`申请输出，dtype通过`x.options().dtype(...)`指定。
5. **拉起kernel**：使用`ACLNN_CMD(aclnn接口名, 入参..., 出参...)`宏调用aclnn接口（如`ACLNN_CMD(aclnnSwigluGroup, ...)`），入参顺序需与aclnn接口定义一致；该宏自动完成类型转换、workspace申请、stream下发与资源释放。
6. **导出绑定**：通过`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`将C++函数绑定为与schema同名的Python接口。
7. 魔数（如维度数`2`、默认dtype枚举）应以`constexpr`具名常量表达，避免裸写字面量。

   C++ wrapper的典型骨架（参考`swiglu_group.cpp`）：

   ```cpp
   #include <torch/extension.h>
   #include "aclnn_common.h"

   namespace cann_ops_nn {
   namespace activation {
   namespace {
   constexpr int64_t kSplitFactor = 2;

   void CheckNpuTensor(const at::Tensor& tensor, const char* name)
   {
       TORCH_CHECK(tensor.defined(), name, " must be defined");
       TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
   }
   } // namespace

   at::Tensor swiglu_group(const at::Tensor& x, const c10::optional<at::Tensor>& weight,
                           const c10::optional<at::Tensor>& group_index, double clamp_limit)
   {
       CheckNpuTensor(x, "x");
       TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

       c10::SmallVector<int64_t, N> y_shape;
       // ... 推导输出 shape ...

       at::Tensor y = at::empty(y_shape, x.options());

       ACLNN_CMD(aclnnSwigluGroup, x, weight, group_index, clamp_limit, y);
       return y;
   }

   } // namespace activation
   } // namespace cann_ops_nn

   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
   {
       m.def("swiglu_group", &cann_ops_nn::activation::swiglu_group, "SwigluGroup on NPU");
   }
   ```

### 3.2 Python前端（`${op_api}.py`）

负责JIT编译管理、schema/meta注册与对外接口封装：

1. **OpBuilder子类**：继承`OpBuilder`，在`__init__`中以`super().__init__("<schema算子名>")`传入算子名，并实现三个抽象方法：
   - `sources()`：返回相对`cann_ops_nn`包根的C++源文件路径列表，如`["csrc/activation/swiglu_group.cpp"]`；
   - `schema()`：返回算子schema字符串（见[2.3 Schema标识符](#schema标识符算子签名)）；
   - `register_meta()`：用`@impl(get_as_library(), self.name, "Meta")`注册Meta实现，仅做shape/dtype推导，不触碰真实NPU计算（FakeTensor/图模式必需）。
2. **实例化与初始化**：模块加载时实例化builder并调用`_ensure_initialized()`注册schema与meta：
   ```python
   builder = SwigluGroupOpBuilder()
   builder._ensure_initialized()
   ```
3. **PrivateUse1 dispatcher**：用`@impl(get_as_library(), builder.name, "PrivateUse1")`注册NPU后端实现，函数体透传到编译产物`op_module.<算子名>(...)`。`PrivateUse1`是PyTorch为自定义NPU后端预留的dispatch key。
4. **对外接口**：提供面向用户的函数，负责参数整理、默认值处理等，最终调用`torch.ops.cann_ops_nn.<算子名>(...)`通过dispatcher调度。
5. **对外api必须书写注释（docstring）**：每个对外导出的接口都要有docstring，至少覆盖「功能说明、各参数含义/shape/dtype/取值范围、返回值说明」。
6. Meta实现、dispatcher、对外函数三者的参数顺序与默认值必须与schema一致。

   Python前端的典型骨架（参考`swiglu_group.py`）：

   ```python
   import torch
   from torch.library import impl
   from cann_ops_nn.op_builder import OpBuilder, get_as_library


   class SwigluGroupOpBuilder(OpBuilder):
       def __init__(self):
           super().__init__("swiglu_group")

       def sources(self):
           return ["csrc/activation/swiglu_group.cpp"]

       def schema(self):
           return "swiglu_group(Tensor x, *, Tensor? weight=None, Tensor? group_index=None, float clamp_limit=-1.0) -> Tensor"

       def register_meta(self):
           @impl(get_as_library(), self.name, "Meta")
           def swiglu_group_meta(x, *, weight=None, group_index=None, clamp_limit=-1.0):
               last_dim = x.size(x.dim() - 1)
               shape = list(x.shape)
               shape[-1] = last_dim // 2
               return x.new_empty(shape)


   builder = SwigluGroupOpBuilder()
   builder._ensure_initialized()


   @impl(get_as_library(), builder.name, "PrivateUse1")
   def swiglu_group(x, *, weight=None, group_index=None, clamp_limit=-1.0):
       op_module = builder.load()
       return op_module.swiglu_group(x, weight, group_index, clamp_limit)
   ```

### 3.3 图模式Converter（`graph_convert_${op_api}.py`）

负责在torchair图模式（GE）下把aten算子转换为GE节点：

1. **可选依赖保护**：torchair相关import统一包在`try/except ImportError`中，用`_TORCHAIR_AVAILABLE`标志位控制，避免在无torchair环境下导入失败。
2. **GE op函数**：定义与`op_type`同名的大驼峰函数（如`SwigluGroup`），通过docstring写明`REG_OP`的IR定义（INPUT/OPTIONAL_INPUT/OUTPUT/ATTR等），并组织`inputs`/`attrs`/`outputs`后调用`ge_op(...)`，IR通过`IrDef(...)`链式声明。
3. **Converter注册**：用`@register_fx_node_ge_converter(torch.ops.cann_ops_nn.<算子名>.default)`装饰`convert_<算子名>`函数，其参数顺序与schema完全一致，函数体调用上面的GE op函数。
4. 在算子的`__init__.py`中导出Converter（如`convert_swiglu_group`），确保注册逻辑被执行。

### 3.4 对外导出（`__init__.py`）

对外导出分两级：

1. **算子`__init__.py`**：每个新增算子的对外接口与Converter都需在此显式import导出，导入即触发schema/meta/converter注册：
   ```python
   __all__ = ["swiglu_group", "convert_swiglu_group"]

   from .swiglu_group import swiglu_group
   from .graph_convert_swiglu_group import convert_swiglu_group
   ```
2. **分类`__init__.py`**（如`activation/__init__.py`）：在分类目录下导出算子，构建时会被复制到whl中：
   ```python
   __all__ = ["swiglu_group", "swiglu_group_quant"]

   from .swiglu_group import swiglu_group
   from .swiglu_group_quant import swiglu_group_quant
   ```

   `cann_ops_nn/__init__.py`（包根层）通过`from . import ops`触发所有算子的自动发现与加载，用户可通过`cann_ops_nn.<接口名>`直接访问。

## 4. 文档规范（`docs/zh/${op_api}.md`）

每个对外算子api需配套一份中文文档（如有），建议章节顺序与已有算子文档对齐：

1. **标题**：算子名（特殊字符如`_`需转义为`\_`）。
2. **产品支持情况**：表格列出支持的产品形态及是否支持。
3. **功能说明**：API功能概述 + 计算公式（数学表达用LaTeX），并说明各符号与参数的对应关系。
4. **函数原型**：代码块给出完整函数签名，含默认值与`*`分隔。
5. **参数说明**：逐个参数说明「必选/可选、语义、shape、dtype、取值范围/约束」。
6. **输出说明**：逐个输出说明shape、dtype等。
7. **约束说明**：分类列出参数一致性约束、shape/取值范围约束等。
8. **调用示例**：给出单算子模式的完整可运行示例。

## 5. 编码通用约束

- **许可证头**：所有新增源文件（`.py`/`.cpp`/`.h`）必须包含Huawei版权与CANN Open Software License Agreement Version 2.0许可证头，年份填当年。Python/脚本用`#`注释，C++用`//`或`/* */`。
- **接口注释**：对外api接口必须书写docstring（功能、参数、返回值）；C++ wrapper关键逻辑（校验、aclnn调用）也应有简要注释。
- **参数校验前置**：C++侧用`TORCH_CHECK(cond, msg...)`；错误信息需包含变量实际值，便于定位。
- **避免魔数**：维度数、dtype枚举值等以`constexpr`具名常量表达，并在文档/注释中说明枚举含义。
- **公共能力复用**：类型转换、`ACLNN_CMD`等优先复用`common/`下的公共头，不在各算子中重复实现。
- **一致性自检**：提交前确认schema、C++ wrapper、Meta、dispatcher、对外函数、Converter、文档七处的算子名、参数名、参数顺序、默认值保持一致。
