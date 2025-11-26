# AscendOps

**AscendOps** - 一个轻量级，高性能的算子开发工程模板

## 项目简介 | Introduction
AscendOps 是一个轻量级，高性能的算子开发工程模板，它集成了PyTorch、PyBind11和昇腾CANN工具链，提供了从算子内核编写，编译到Python封装的完整工具链。

## 核心特性 | Features
🚀 开箱即用 (Out-of-the-Box): 预置完整的昇腾NPU算子开发环境配置，克隆后即可开始开发。

🧩 极简设计 (Minimalist Design): 代码结构清晰直观，专注于核心算子开发流程。

⚡ 高性能 (High Performance): 基于AscendC编程模型，充分发挥昇腾NPU硬件能力。

📦 一键部署 (One-Click Deployment): 集成setuptools构建系统，支持一键编译和安装。

🔌 PyTorch集成 (PyTorch Integration): 无缝集成PyTorch张量操作，支持自动微分和GPU/NPU统一接口。

## 核心交付件 | Core Deliverables
1. `csrc/xxx/xxx_torch.cpp` 算子Kernel实现
2. `csrc/xxx/CMakeLists.txt` 算子cmake配置
3. `csrc/npu_ops_def.cpp` 注册算子接口

## 环境要求 | Prerequisites
*   Python: 3.8+
*   CANN Ascend Toolkit
*   CANN Ascend Legacy
*   PyTorch: 2.1.0+
*   PyTorchAdapter 7.1.0+

## 环境准备 | Preparation

1. **安装社区版CANN toolkit包和社区版CANN legacy包**

    根据实际环境，安装社区版CANN toolkit包和社区版CANN legacy包，社区包的安装部署参考[算子调用](../../docs/invocation/quick_op_invocation.md)环境准备章节。

2. **配置环境变量**

	根据实际场景，选择合适的命令。

    ```bash
   # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
   source /usr/local/Ascend/set_env.sh
   # 指定路径安装
   # source ${install-path}/set_env.sh
    ```

3. **安装torch与torch_npu包**

   根据实际环境，下载对应 torch 包，常见的 wheel 文件名示例如下：

   - x86_64 Linux 版本：`torch-${torch_version}+cpu-${python_version}-linux_x86_64.whl`
   - ARM Linux（旧版本通常无 `+cpu` 后缀）：`torch-${torch_version}-${python_version}-linux_aarch64.whl`

   说明：上面示例中的版本号及 `+cpu` 后缀仅作为示例。不同操作系统和架构（尤其是 ARM Linux）对应的 wheel 名称可能没有 `+cpu` 后缀，请以 PyTorch 官方下载页面或 `pip` 实际可用的包名为准。

   下载链接为：[官网地址](http://download.pytorch.org/whl/torch)

   安装命令如下（将 `<torch_whl>` 替换为实际下载的文件名）：

    ```sh
    pip3 install <torch_whl>
    ```

   根据实际环境，安装对应 torch-npu 包，例如：`torch_npu-${torch_version}-${python_version}-linux_${arch}.whl`，下载链接：[官网地址](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)

   可以直接使用 pip 命令下载安装（将 `<torch_npu_whl>` 替换为实际下载的文件名），命令如下：

    ```sh
    pip3 install <torch_npu_whl>
    ```

    - ${torch_version}：表示 torch 包版本号。
    - ${python_version}：表示 python 版本号。
    - ${arch}：表示 CPU 架构，如 aarch64、x86_64。

## 安装步骤 | Installation

1. 进入目录，安装依赖
    ```sh
    cd ./examples/fast_kernel_launch_example
    pip3 install -r requirements.txt
    ```

2. 从源码构建.whl包
    ```sh
    python3 -m build --wheel -n
    ```

3. 安装构建好的.whl包
    ```sh
    pip3 install dist/xxx.whl
    ```

    重新安装请使用以下命令覆盖已安装过的版本：
    ```sh
    pip3 install dist/xxx.whl --force-reinstall --no-deps
    ```

4. (可选)再次构建前建议先执行以下命令清理编译缓存
    ```sh
    python3 setup.py clean
    ```

## 开发模式构建 | Developing Mode

此命令实现即时生效的开发环境配置，执行后即可使源码修改生效，省略了构建完整whl包和安装的过程，适用于需要多次修改验证算子的场景：
  ```sh
  pip3 install --no-build-isolation -e .
  ```

## 使用示例 | Usage Example

安装完成后，您可以像使用普通PyTorch操作一样使用NPU算子，以conv3d算子为例，您可以在`ascend_ops/csrc/conv3d_custom/test`目录下找到并执行这个脚本:

```bash
cd ./ascend_ops/csrc/conv3d_custom/test/
python3 test_conv3d_custom.py
```

最终看到如下输出，即为执行成功：
```bash
compare CPU Result vs NPU Result: True
```


## 开发新算子 | Developing New Operators
1. 编写算子调用文件

    在 `ascend_ops/csrc/` 目录下添加新的算子目录 `mykernel`，在 `mykernel` 目录下添加新的算子调用文件 `mykernel_torch.cpp`
    ```c++
    __global__ __aicore__ void mykernel(GM_ADDR input, GM_ADDR output, int64_t num_element) {
        // 您的算子kernel实现
    }

    void mykernel_api(aclrtStream stream, const at::Tensor& x, const at::Tensor& y) {
        // 您的算子入口实现，在该方法中使用<<<>>>的方式调用算子kernel
        mykernel<<<blockDim, nullptr, stream>>>(x, y, num_element);
    }

    torch::Tensor mykernel_npu(torch::Tensor x, torch::Tensor y) {
        // 您的算子wrapper接口，用于向pytorch注册自定义接口
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, x.scalar_type(), "mykernel_npu", [&] { mykernel_api(stream, x, y); });
    }

    // PyTorch提供的宏，用于在特定后端注册算子
    TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
    {
        m.impl("mykernel", mykernel_npu);
    }
    ```

2. 在`mykernel`目录下创建`CMakeLists.txt`

    将如下样例中的mykernel，替换为自己的算子名称
    ```cmake
    message(STATUS "BUILD_TORCH_OPS ON in mykernel")
    # MYKERNEL operation sources
    file(GLOB MYKERNEL_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

    set(MYKERNEL_SOURCES ${MYKERNEL_NPU_SOURCES})
    # Mark .cpp files with special properties
    set_source_files_properties(
        ${MYKERNEL_NPU_SOURCES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=CubeCore --cce-auto-sync -xcce"
    )

    # Create object library
    add_library(mykernel_objects OBJECT ${MYKERNEL_SOURCES})

    target_compile_options(mykernel_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_include_directories(mykernel_objects PRIVATE ${COMMON_INCLUDE_DIRS})
    return()
    ```

3. 在 `ascend_ops/csrc/npu_ops_def.cpp`中添加TORCH_LIBRARY_IMPL定义

    ```c++
    TORCH_LIBRARY(ascend_ops, m) {
        m.def("mykernel(Tensor x) -> Tensor");
    }
    ```

4. (可选)在 `ascend_ops/ops.py`中封装自定义接口
    ```python
    def mykernel(x: Tensor) -> Tensor:
        return torch.ops.ascend_ops.mykernel(x)
    ```

5. 使用开发模式进行编译
    ```bash
    pip install --no-build-isolation -e .
    ```

6. 编写测试脚本并测试新算子
    ```python
    torch.ops.ascend_ops.mykernel(x)
    ```
