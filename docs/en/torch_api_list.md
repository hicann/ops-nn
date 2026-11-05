# torch_extension APIs

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T13:03:37.757Z -->

## Usage Instructions

To simplify operator calls, the project provides a set of APIs compatible with the native PyTorch style. These APIs use the PyTorch JIT mechanism (`torch.utils.cpp_extension.load`) to just-in-time compile the C++ Kernel Wrapper upon the first call, bridging PyTorch functions to the aclnn APIs of CANN. The APIs also support TorchAir graph mode through the GE Converter, facilitating model and application development.

- **Software Package Description**

  Before calling the torch\_extension APIs, ensure that the CANN Toolkit package, ops-nn package, and TorchNPU package have been installed.

- **Calling Method**:

  Calling the torch\_extension APIs requires the `cann_ops_nn` module, which is defined in `${INSTALL_DIR}/python/site-packages/cann_ops_nn`, where \$\{INSTALL\_DIR\} indicates the file path after CANN installation.

  ```python
  import torch
  import torch_npu
  import cann_ops_nn
  ```

## API List

> [!NOTE]
>
> - Operator features: Before calling the APIs, learn the basics of operators, including **deterministic algorithms**, **batch consistency**, and **common quantization modes**. For details, see [Basic Operator Concepts](context/basic_concept.md).
> - Symbol description: In the table, the "-" symbol indicates that the API is not yet supported by the products in the current column.

|    API Name   |   Description     |  Deterministic Computation (A2/A3)  | Deterministic Computation (Ascend 950) |
| ----------- | ------------------- | ------------------- | ------------------- |
|[add_rms_norm_dynamic_quant](../../norm/add_rms_norm_dynamic_mx_quant/docs/torchapi_add_rms_norm_dynamic_quant_en.md)|Fuses Add, RmsNorm, and MX dynamic quantization. Performs dynamic MX quantization along the trailing axis in groups of blocksize=32 and supports three-way residual addition. Outputs FP4/FP8 quantization results, the addition result, quantization scales, and the reciprocal of the standard deviation.|-|Enabled by default|
|[situ_glu](../../activation/situ_glu/docs/torchapi_situ_glu_en.md)|SiTU gated linear unit activation function. Splits the input tensor along the specified dimension into gate and up halves, and computes the output using the SiTU formula.|Enabled by default|Enabled by default|
|[situ_glu_grad](../../activation/situ_glu_grad/docs/torchapi_situ_glu_grad_en.md)|Backward gradient computation of the SiTU gated linear unit activation function. Computes the input gradients based on the upstream gradients and forward inputs.|Enabled by default|Enabled by default|
|[swiglu_group](../../activation/swiglu_group/docs/torchapi_swiglu_group_en.md)|SwiGLU grouped activation operator. Splits the input tensor into two parts along the last dimension, performs clamp and sigmoid operations on each part, and multiplies the results. Optional weights and group indices are supported.|-|Enabled by default|
|[clipped_swiglu](../../activation/clipped_swiglu/docs/torchapi_clipped_swiglu_en.md)|Clipped Swish gated linear unit activation function. Implements the SwiGlu computation of x and outputs the SwiGlu result.|-|Enabled by default|
|[clipped_swiglu_grad](../../activation/clipped_swiglu_grad/docs/torchapi_clipped_swiglu_grad_en.md)|Backward gradient operator of ClippedSwiglu. Recomputes the clamp mask and sigmoid based on the upstream gradient grad_y and forward input x, and outputs grad_x.|-|Enabled by default|
|[swiglu_group_backward](../../activation/swiglu_group_grad/docs/torchapi_swiglu_group_backward_en.md)|Backward gradient operator of the ClampedSwiglu activation function. Recomputes the clamp mask and sigmoid from the upstream gradient grad_output and forward input x, and outputs grad_x and the optional grad_weight.|-|-|
|[swiglu_group_quant](../../activation/swiglu_group_quant/docs/torchapi_swiglu_group_quant_en.md)|Fuses SwiGLU grouped activation with quantization. Supports output in multiple quantization modes such as FP8 based on the SwiGLU computation.|-|Enabled by default|
|[rms_norm_dynamic_quant](../../norm/rms_norm_dynamic_quant/docs/torchapi_rms_norm_dynamic_quant_en.md)|Fuses RMS Normalization with INT8 dynamic quantization and outputs the quantized tensor and scaling factors.|Enabled by default|-|
|[grouped_dynamic_mx_quant_with_dual_axis](../../quant/grouped_dynamic_mx_quant_with_dual_axis/docs/torchapi_grouped_dynamic_mx_quant_with_dual_axis_en.md)|Performs dynamic MX quantization on the 2D input `x` along both the last and second-to-last dimensions based on the row grouping described by `group_index`, and outputs FP8 quantization results in both directions and the corresponding FLOAT8_E8M0 scaling factors.|-|Enabled by default|
|[quant_matmul_activation_quant](../../matmul/quant_matmul_activation_quant/docs/torchapi_quant_matmul_activation_quant_en.md)| Fuses quantized matrix multiplication, GELU/SwiGLU/SiLU activation, and dynamic MX quantization. Weights support ND and NZ formats. |-|Enabled by default|
|[flat_quant](../../matmul/flat_quant/docs/torchapi_flat_quant_en.md)| Performs two Kronecker product-based small matmuls on matrix x in sequence, and then quantizes the matmul results. |Enabled by default|Enabled by default|
|[matmul_emu_split_weight](../../matmul/matmul_emu_split_weight/docs/torchapi_matmul_emu_split_weight_en.md)| Emulates FP32-precision matmul through dual BF16 GEMM fusion. Offline, the FP32 weight is split into high-order BF16 and low-order residual BF16. During inference, two BF16 GEMMs are executed and linearly combined. Activations remain BF16 throughout, and both matmuls run on the Cube. |Enabled by default|Enabled by default|
|[cross_entropy_sum_exp_and_index_logit](../../loss/cross_entropy_sum_exp_and_index_logit/docs/torchapi_cross_entropy_sum_exp_and_index_logit_en.md)| CrossEntropy local computation fusion operator for the vocab parallel (Tensor Parallel) use case.|-| Enabled by default |
|[mx_to_block_mx_quant](../../quant/mx_to_block_mx_quant/docs/torchapi_mx_to_block_mx_quant_en.md)| Converts the FLOAT4 tensor obtained by calling npu_dynamic_mx_quant together with the FLOAT8_E8M0 scaling coefficients into the FLOAT8 block quantization format, and outputs quantization scales along both the -1 axis and -2 axis.|-| Enabled by default |
|[dequant_situ_quant](../../quant/dequant_situ_quant/docs/torchapi_dequant_situ_quant_en.md)|Adds dequant and quant operations before and after the Situ activation function to implement the DequantSituQuant computation of x.|Enabled by default|-|
|[situ_mx_quant](../../quant/situ_mx_quant/docs/torchapi_situ_mx_quant_en.md)|Fuses the Situ activation function with dynamic MX quantization to implement the SituMxQuant computation of x. Supports FP8 quantized output.|-|Enabled by default|
|[fused_matmul](../../matmul/fused_mat_mul/docs/torchapi_fused_matmul_en.md)|Fuses matrix multiplication with general vector computation.|-|Enabled by default|
|[transpose_quant_batch_mat_mul](../../matmul/transpose_quant_batch_mat_mul/docs/torchapi_transpose_quant_batch_mat_mul_en.md)|Performs transposed matmul on MX-quantized (MXFP8/MXFP4) tensors x1 and x2. The input scale is of the FLOAT8_E8M0 type, and the output is FLOAT16/BFLOAT16.|-|Enabled by default|
|[transpose_batch_mat_mul](../../matmul/transpose_batch_mat_mul/docs/torchapi_transpose_batch_mat_mul_en.md)|Performs matrix multiplication on 3D tensors x1 and x2 with perm-based transposes, batch_split_factor splitting, and optional FLOAT16+scale INT8 quantization output.|Enabled by default|Enabled by default|
