# torch_extension接口

## 使用说明

为简化算子调用，项目提供了一套兼容PyTorch原生风格的API。该API通过PyTorch的JIT机制（`torch.utils.cpp_extension.load`），在首次调用时即时编译C++ Kernel Wrapper，将PyTorch函数桥接到CANN的aclnn API，同时通过GE Converter支持TorchAir图模式，便于开发者构建模型与应用。

- **软件包说明**

  调用torch\_extension接口时，请确保已安装CANN Toolkit包、ops-nn包、TorchNPU包。

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
|[situ_glu](../../activation/situ_glu/docs/torchapi_situ_glu.md)|SiTU门控线性单元激活函数，对输入张量沿指定维度切分为门控与上路径两半，按SiTU公式计算输出。|默认支持确定性计算|默认支持确定性计算|
|[situ_glu_grad](../../activation/situ_glu_grad/docs/torchapi_situ_glu_grad.md)|SiTU门控线性单元激活函数的反向梯度计算，根据上游梯度和前向输入计算输入梯度。|默认支持确定性计算|默认支持确定性计算|
|[swiglu_group](../../activation/swiglu_group/docs/torchapi_swiglu_group.md)|SwiGLU分组激活算子，对输入张量按最后一维拆分为两部分，分别进行clamp和sigmoid操作后相乘，支持可选的权重和分组索引。|-|默认支持确定性计算|
|[swiglu_group_backward](../../quant/swiglu_group_grad/docs/torchapi_swiglu_group_backward.md)|ClampedSwiglu激活函数的反向梯度算子，从上游梯度grad_output和前向输入x重算clamp mask与sigmoid，输出grad_x与可选grad_weight。|-|-|
|[swiglu_group_quant](../../activation/swiglu_group_quant/docs/torchapi_swiglu_group_quant.md)|融合SwiGLU分组激活与量化的算子，在SwiGLU计算基础上支持FP8/MXFP4等多种量化模式输出。|-|默认支持确定性计算|
|[rms_norm_dynamic_quant](../../norm/rms_norm_dynamic_quant/docs/torchapi_rms_norm_dynamic_quant.md)|融合RMS Normalization与INT8动态量化，输出量化后的张量及缩放因子。|默认支持确定性计算|-|
|[grouped_dynamic_mx_quant_with_dual_axis](../../quant/grouped_dynamic_mx_quant_with_dual_axis/docs/torchapi_grouped_dynamic_mx_quant_with_dual_axis.md)|根据`group_index`描述的行分组，对二维输入`x`同时沿最后一维和倒数第二维进行动态MX量化，输出两个方向的FP8量化结果及对应的FLOAT8_E8M0缩放因子。|-|默认确定性实现|
|[quant_matmul_activation_quant](../../matmul/quant_matmul_activation_quant/docs/torchapi_quant_matmul_activation_quant.md)| 融合量化的矩阵乘、激活以及动态量化计算，weight仅支持NZ格式。 |-|默认支持确定性计算|
|[flat_quant](../../quant/flat_quant/docs/torchapi_flat_quant.md)| 为矩阵x依次进行两次克罗内克积小矩阵乘法，然后针对矩阵乘的结果进行量化处理。 |默认支持确定性计算|默认支持确定性计算|
|[matmul_emu_split_weight](../../matmul/matmul_emu_split_weight/docs/torchapi_matmul_emu_split_weight.md)| 使用双路BF16 GEMM融合模拟FP32精度矩阵乘法。离线将FP32权重拆分为高位BF16与低位残差BF16，推理阶段执行两次BF16 GEMM并做线性组合，激活值全程保持BF16，两路矩阵乘均运行在Cube上。 |-|默认支持确定性计算|
|[mx_to_block_mx_quant](../../quant/mx_to_block_mx_quant/docs/torchapi_mx_to_block_mx_quant.md)| 将调用 npu_dynamic_mx_quant 量化得到的 FLOAT4 的 Tensor 结合 FLOAT8_E8M0 缩放系数，转换为 FLOAT8 分块量化格式，同时输出 -1 轴和 -2 轴方向的量化尺度。|-| 默认确定性实现 |
