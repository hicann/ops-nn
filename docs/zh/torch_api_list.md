# torch_extension接口

## 使用说明

为简化算子调用，项目提供了一套兼容PyTorch原生风格的API。该API通过PyTorch的JIT机制（`torch.utils.cpp_extension.load`），在首次调用时即时编译C++ Kernel Wrapper，将PyTorch函数桥接到CANN的aclnn API，同时通过GE Converter支持TorchAir图模式，便于开发者构建模型与应用。

- **软件包说明**

  调用torch\_extension接口时，请确保已安装CANN Toolkit包、ops-nn包、Ascend for PyTorch包。

- **调用方式**：

  调用torch\_extension接口时，依赖`cann_ops_nn`模块，定义在`${INSTALL_DIR}/python/site-packages/cann_ops_nn`，\$\{INSTALL\_DIR\}表示CANN安装后文件路径。

  ```python
  import torch
  import torch_npu
  import cann_ops_nn
  ```

## 接口列表

> **确定性简介**：因CANN或NPU型号不同等原因，可能无法保证同一个API运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分接口可通过PyTorch中控制算法确定性的全局开关[torch.use_deterministic_algorithms](https://github.com/pytorch/pytorch/blob/main/torch/__init__.py)开启确定性算法，使多次运行结果一致。

|    接口名   |   说明     |  确定性说明（A2/A3）  | 确定性说明（Ascend 950） |
| ----------- | ------------------- | ------------------- | ------------------- |
|[swiglu_group](../../torch_extension/cann_ops_nn/ops/activation/swiglu_group/swiglu_group.md)|SwiGLU分组激活算子，对输入张量按最后一维拆分为两部分，分别进行clamp和sigmoid操作后相乘，支持可选的权重和分组索引。|-|-|
|[swiglu_group_quant](../../torch_extension/cann_ops_nn/ops/activation/swiglu_group_quant/swiglu_group_quant.md)|融合SwiGLU分组激活与量化的算子，在SwiGLU计算基础上支持FP8/MXFP4等多种量化模式输出。|-|-|
|[rms_norm_dynamic_quant](../../torch_extension/cann_ops_nn/ops/norm/rms_norm_dynamic_quant/rms_norm_dynamic_quant.md)|融合RMS Normalization与INT8动态量化，输出量化后的张量及缩放因子。|-|-|
