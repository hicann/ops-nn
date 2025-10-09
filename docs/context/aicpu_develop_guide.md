# AI CPU算子开发指南

> 说明：
>
> - 算子开发过程中涉及的基本概念、AI CPU接口等，详细介绍请参考[《TBE&AI CPU算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevWizard)。
> - 若基于社区版CANN包对AI CPU算子源码修改，请使用自定义算子包方式编译执行。

开发指南以`AddExample`算子开发为例，介绍新算子开发流程以及涉及的交付件，流程图如下，完整样例代码请访问项目`examples`目录。

```mermaid
graph LR
	A([前提条件]) --> W([工程创建])
    W --> C([Kernel实现])
    C --> E([图模式适配])
    E --> F([编译部署])
    F --> G([算子验证])
```

1. [前提条件](#前提条件)：

   ① 环境部署：开发算子前，请确保基础环境已安装，例如依赖的驱动、固件、CANN软件包等。

   ② 算子设计：分析实际业务诉求，合理设计算子规格，包括算子输入、输出、属性的数据类型、shape等。

2. [工程创建](#工程创建)：开发算子前，需按要求创建算子目录，方便后续算子的编译和部署。

3. [Kernel实现](#Kernel实现)：实现Device侧算子核函数。

4. [图模式适配](#图模式适配)：AI CPU算子目前仅支持图模式调用，需完成InferShape和InferDataType，后续会支持aclnn接口调用。

5. [编译部署](#编译部署)：通过工程编译脚本完成自定义算子的编译和安装。

6. [算子验证](#算子验证)：通过常见算子调用方式，验证自定义算子功能。

##  前提条件
**1. 环境部署**

开发算子前，请参考[环境准备](./quick_op_invocation.md#环境准备)完成环境搭建。

**2. 算子设计**

确定目标算子的功能和计算逻辑，并将数学计算逻辑转化为可执行的代码逻辑。以自定义`AddExample`算子设计为例，设计步骤如下：

```mermaid
graph LR
	A([设计数学表达式]) -->B([明确输入输出])
    B --> C([设计核函数])
    C -->D([明确所需接口])
```

`AddExample`算子设计规格如下：

<table>
<tr>
<th>算子类型</th>
<td colspan="4" align="center">AddExample</td>
</tr>
<tr>
<th>算子表达式</th>
<td colspan="4" align="center">y[i] = x1[i] + x2[i] </td>
</tr>
<tr>
<th rowspan="3" >算子输入</th>
<th>name</th>
<th>shape</th>
<th>dataType</th>
<th>format</th>
</tr>
<tr>
<td>x1</td>
<td>(32,4,4,4)</td>
<td>float/int32</td>
<td>ND</td>
</tr>
<tr>
<td>x2</td>
<td>(32,4,4,4)</td>
<td>float/int32</td>
<td>ND</td>
</tr>
<tr>
<th>算子输出</th>
<td>y</td>
<td>(32,4,4,4)</td>
<td>float/int32</td>
<td>ND</td>
</tr>
<tr>
<th rowspan="8" align="top">关键接口（AI CPU）</th>
</tr>
<tr>
<td colspan="4" >CpuKernelContext类：包含了Cpu Kernels的Context定义以及方法，可以获取Input，Output以及属性等信息</td>
</tr>
<tr>
<td colspan="4" >TensorShape类：包含了Cpu Kernels的Tensor Shape定义以及方法</td>
</tr>
<tr>
<td colspan="4" >Tensor类：包含了Cpu Kernels的Tensor定义以及方法</td>
</tr>
<tr>
<td colspan="4" >AttrValue类：包含了Cpu Kernels的属性定义以及方法</td>
</tr>
<tr>
<td colspan="4" >NodeDefBuilder类：包含了Cpu kernels的NodeDef定义以及方法</td>
</tr>
<tr>
<td colspan="4" >CpuKernelRegister类：AI CPU算子的Kernel注册类，用于框架获取算子Kernel的Compute函数相关信息，以及执行算子Kernel的Compute函数</td>
</tr>
<tr>
<td colspan="4" >REGISTER_CPU_KERNEL：包含Cpu Kernels的Kernel的基类定义以及注册宏</td>
</tr>
<tr>
<th rowspan="5" >算子实现文件</th>
<td colspan="4" >add_example_aicpu.cpp</td>
</tr>
</table>


## 工程创建

工程创建是算子开发的重要步骤，为后续代码编写、编译构建和调试提供统一的目录结构和文件组织方式。

本项目`build.sh`，支持快速创建算子目录。进入项目根目录，执行以下命令：

```bash
# 创建指定算子目录，如bash build.sh --genop_aicpu=examples/add_example
bash build.sh --genop_aicpu=${op_class}/${op_name}
```
- \$\{op_class\}表示算子类型，如matmul类。
- \$\{op_name\}表示算子名的小写下划线形式，如`AddExample`算子对应为add\_example。

如果命令执行成功，会看到如下提示信息：

```bash
Create the initial directory for ${op_name} under ${op_class} success
```
创建完成后，目录结构如下所示：

```
${op_name}                              # 替换为实际算子名的小写下划线形式
├── op_host                             # Host侧实现
│   ├── ${op_name}_def.cpp              # 算子信息库，定义算子基本信息，如名称、输入输出、数据类型等
│   ├── ${op_name}_infershape.cpp       # InferShape实现，实现算子形状推导，在运行时推导输出shape
│   └── CMakeLists.txt                  # Host侧CMakeLists文件
├── op_graph                            # 图融合相关实现
│   ├── CMakeLists.txt                  # op_graph侧CMakeLists文件
│   ├── ${op_name}_graph_infer.cpp      # InferDataType文件，实现算子类型推导，在运行时推导输出dataType
│   └── ${op_name}_proto.h              # 算子原型定义，用于图优化和融合阶段识别算子
├── op_kernel_aicpu                     # Device侧Kernel实现
│   ├── ${op_name}_aicpu.cpp            # Kernel入口文件，包含主函数和调度逻辑
│   └── ${op_name}_aicpu.h              # Kernel头文件，包含函数声明、结构定义、逻辑实现
└── CMakeLists.txt                      # 算子CMakeLists入口
```
使用上述命令行创建算子工程后，若要手动删除新创建出的算子工程，需要同时删除与算子工程同目录CMakeLists.txt中新添加的add_subdirectory(${op_class})

## Kernel实现

Kernel是算子在NPU执行的核心部分，Kernel实现包括如下步骤：

```mermaid
graph LR
	H([算子类声明]) -->A([Compute函数实现])
	A -->B([注册算子])
```

1. 算子类声明

   Kernel实现的第一步，需在头文件`op_kernel_aicpu/${op_name}_aicpu.h`进行算子类的声明，算子类需继承CpuKernel基类。

2. Compute函数实现

   获取输入/输出Tensor信息并进行合法性校验，然后实现核心计算逻辑（如加法操作），并将计算结果设置到输出Tensor中。

3. 注册算子

   注册AI CPU算子的Kernel实现，用于框架获取算子Kernel的Compute函数。

根据上述步骤，在\$\{op\_name\}\_aicpu.h中定义Kernel头文件，包含算子类声明、结构定义等，示例如下，`AddExample`算子完整代码请参考`examples/add_example_aicpu/op_kernel_aicpu`下[add_example_aicpu.h](../../examples/add_example_aicpu/op_kernel_aicpu/add_example_aicpu.h)。

```CPP
// 1、算子类声明
// 包含AI CPU基础库头文件
#include "cpu_kernel.h"
// 定义命名空间aicpu(固定不允许修改)，并定义算子Compute实现函数
namespace aicpu {
// 算子类继承CpuKernel基类
class AddExampleCpuKernel : public CpuKernel {
 public:
  ~AddExampleCpuKernel() = default;
  // 声明函数Compute（需要重写），形参CpuKernelContext为CPUKernel的上下文，包括算子输入、输出和属性信息
  uint32_t Compute(CpuKernelContext &ctx) override;
};
}  // namespace aicpu
```
编写Kernel入口文件\$\{op\_name\}\_aicpu.cpp ，包含主函数和调度逻辑，示例如下，`AddExample`算子完整代码请参考`examples/add_example_aicpu/op_kernel_aicpu`下[add_example_aicpu.cpp](../../examples/add_example_aicpu/op_kernel_aicpu/add_example_aicpu.cpp)。

```C++
// 2、Compute函数实现
#include "add_example_aicpu.h"

namespace {
// 算子名
const char* const kAddExample = "AddExample";
const uint32_t kParamInvalid = 1;
}  // namespace

// 定义命名空间aicpu
namespace aicpu {
// 实现自定义算子类的Compute函数
uint32_t AddExampleCpuKernel::Compute(CpuKernelContext& ctx) {
  // 从CpuKernelContext中获取input tensor
  Tensor* input0 = ctx.Input(0);
  Tensor* input1 = ctx.Input(1);
  // 从CpuKernelContext中获取output tensor
  Tensor* output = ctx.Output(0);

  // 对tensor进行基本校验, 判断是否为空指针
  if (input0 == nullptr || input1 == nullptr || output == nullptr) {
    return kParamInvalid;
  }

  // 获取input tensor的数据类型
  auto data_type = static_cast<DataType>(input0->GetDataType());
  // 获取input tensor的数据地址，例如输入的数据类型是int32
  auto input0_data = reinterpret_cast<int32_t*>(input0->GetData());
  // 获取tensor的shape
  auto input0_shape = input->GetTensorShape();

  // 获取output tensor的数据地址，例如输出的数据类型是int32
  auto y = reinterpret_cast<int32_t*>(output->GetData());

  // AddCompute函数根据输入类型执行相应计算。
  // 由于C++自身不支持半精度浮点类型，可借助第三方库Eigen（建议使用3.3.9版本）表示。
  switch (data_type) {
    case DT_FLOAT:
      return AddCompute<float>(...);
    case DT_INT32:
      return AddCompute<int32>(...);
    case DT_INT64:
      return AddCompute<int64>(...);
      ....
    default : return PARAM_INVALID;
  }
}

// 3、注册算子Kernel实现，用于框架获取算子Kernel的Compute函数。
REGISTER_CPU_KERNEL(kAddExample, AddExampleCpuKernel);
}  // namespace aicpu
```
## 图模式适配

### Shape与DataType推导

在深度学习中，当一个算子被加入计算图时，为确保图的正确性和后续的编译、优化、执行流程顺利进行，通常需要为该算子实现两个关键的推导函数：
- InferShape：用于推导输出张量的形状（shape）。
- InferDataType：用于推导输出张量的数据类型（dataType）。

操作步骤如下：

**1. 注册InferShape与InferDataType。**

实现两个目标函数之前，需要先进行注册，框架判断算子的shape和dataType推导逻辑由哪两个函数来处理。

**2. InferShape推导实现。**

Infershape函数的作用是根据输入的shape推导输出的shape。

**3. InferDataType推导实现。**

InferDataType函数的作用是根据输入的dataType推导输出的dataType。

根据上述步骤，编写`AddExample`算子的推导实现，示例代码如下：

```C++
// AddExample算子逻辑是两个数相加，因此输出shape与输入shape一致
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    ....
    // 获取输入shape
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    // 获取输出shape
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    // 获取输入DimNum
    auto xShapeSize = xShape->GetDimNum();
    // 设置输出的DimNum
    yShape->SetDimNum(xShapeSize);
    // 依次将输入Dim值设置给输出
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    ....
}

// AddExample算子逻辑是两个数相加，因此输出dataType与输入dataType一致
static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    ....
    // 获取输入的dataType
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    // 将输出dataType设置到输出
    context->SetOutputDataType(IDX_0, sizeDtype);
    ....
}

// 注册InferShape与InferDataType
IMPL_OP_INFERSHAPE(AddExample).
    InferShape(InferShapeAddExample).
    InferDataType(InferDataTypeAddExample);
```

完整代码请参考`examples/add_example_aicpu/op_host`下[add_example_infershape.cpp](../../examples/add_example_aicpu/op_host/add_example_infershape.cpp)。

### 算子原型配置

图模式调用需要将算子原型注册到[Graph Engine](https://www.hiascend.com/cann/graph-engine)（简称GE）中，以便GE能够识别该类型算子的输入、输出及属性信息。注册通过`REG_OP`接口完成，开发者需要定义算子的输入、输出张量类型及数量等基本信息。

示例代码如下，展示了如何注册`AddExample`算子：

  ```CPP
REG_OP(AddExample)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AddExample)
  ```

完整代码请参考`examples/add_example_aicpu/op_graph`下[add_example_proto.h](../../examples/add_example_aicpu/op_graph/add_example_proto.h)。


## 编译部署

算子开发完成后，需对算子工程进行编译，生成自定义算子安装包\*\.run，详细的编译操作如下：

1. **准备工作。**

   参考[前提条件](#前提条件)完成基础环境搭建，同时检查算子开发交付件是否完备，是否在对应算子分类目录下。

2. **编译自定义算子包。**

   以`AddExample`算子为例，假设开发交付件在`examples`目录，完整代码参见[add_example_aicpu](../../examples/add_example_aicpu)目录。

   进入项目根目录，执行如下编译命令（命令介绍参见[build参数说明](./build.md)）：

    ```bash
   # 编译指定算子，如--ops=add_example
   bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list}
    ```

   若提示如下信息，说明编译成功：

    ```bash
   Self-extractable archive "cann-ops-nn-${vendor_name}-linux.${arch}.run" successfully created.
    ```

   若未指定\$\{vendor\_name\}默认使用custom作为包名。编译成功后，生成的自定义算子\*\.run包存放于build_out目录。

   说明：当前自定义算子包\$\{vendor\_name\}和\$\{op\_list\}均为可选，若都不传入编译的是built-in包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。

   注意，构建过程文件在`build`目录，关键文件如下：

    - `libcust_opapi.so`：包含aclnn接口相关实现。
    - `libcust_opmaster_rt2.0.so`：包含Tiling相关实现。

3. **安装自定义算子包。**

   执行以下命令进行安装：

    ```bash
   ./cann-ops-nn-${vendor_name}-linux.${arch}.run
    ```
   自定义算子包安装在`${ASCEND_HOME_PATH}/latest/opp/vendors`路径中，`${ASCEND_HOME_PATH}`表示CANN软件安装目录，可提前在环境变量中配置。自定义算子包不支持卸载。

    自定义算子包的目录结构示例如下：
    ```
    ├── cann-ops-nn-${vendor_name}-linux.${arch}.run             # 包名
    ├── op_api
    │   ├── include
    │   │   ├── aclnn_add_example.h                              # aclnn头文件
    │   └── lib
    │       └── libcust_opapi.so                                 # 算子aclnn接口动态库
    ├── op_impl
    │   └── cpu
    │       └── aicpu_kernel
    │           ├── impl
    │           │   └── libcust_aicpu_kernels.so                 # Kernel实现
    │           └── config
    │               └── cust_aicpu_kernel.json
    ├── op_proto
    │   ├── inc
    │   │   └── add_example_proto.h
    │   └── lib
    │       └── linux
    │           └── ${arch}
    │               └── libcust_opsproto_rt2.0.so
    └── version.info                                             # 包信息
    ```
## 算子验证

开发好的算子完成编译部署后，可通过aclnn方式（推荐）或图模式验证功能，方法请参考[算子调用方式](./op_invocation.md)。