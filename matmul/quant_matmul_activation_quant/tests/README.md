# QuantMatmulActivationQuant测试

覆盖GELU（tanh/erf）的MXFP8、MXFP4计算，SwiGLU、SiLU的MXFP8计算，以及sigmoid激活后与`x3`逐元素相乘的MXFP8计算。算子规格见[README](../README.md)。

| 目录 | 测试内容 |
| --- | --- |
| [ut/op_api](ut/op_api) | ND、NZ接口的参数和布局校验 |
| [ut/op_host](ut/op_host) | 形状推导和Tiling |
| [ut/op_kernel](ut/op_kernel) | CSV驱动的Kernel UT |
| [st/arch35](st/arch35) | TTK Kernel、ACLNN、E2E测试 |
| [assets/spec.py](assets/spec.py) | 参考计算（Golden）和精度比较器 |

## 环境准备

- 使用Linux和Ascend 950，配置配套的CANN、ops-tensor/Blaze，并安装当前版本的Host/API及Kernel算子包，参见[编译安装说明](../../../docs/zh/install/build.md)。Kernel UT还需要tikicpulib。
- TTK需要`ml-dtypes`、`en-dtypes>=0.0.4`。ACLNN、E2E和NPU Golden需要`torch`、`torch_npu`；E2E还需要`cann_ops_nn`。
- 测试NZ或FP4时，TTK须支持完整NZ存储的传递，以及FP4自定义NPU Golden。

## ACLNN UT

在ops-nn根目录执行以下命令。

```bash
bash build.sh -u --opapi --ops=quant_matmul_activation_quant --soc=ascend950
```

用例见[ND接口CSV](ut/op_api/test_aclnn_quant_matmul_activation_quant.csv)和[WeightNZ接口CSV](ut/op_api/test_aclnn_quant_matmul_activation_quant_weight_nz.csv)，两者均覆盖V1和V2。

- `x1Stride`、`x2Stride`、`x1ScaleStride`和`x2ScaleStride`描述非连续视图。可选的末尾两列`yStride`和`yScaleStride`用于验证非连续输出返回`ACLNN_ERR_PARAM_INVALID`。步长留空时按连续布局构造。
- 转置用例覆盖左右矩阵及其scale的匹配与不匹配视图、显式转置属性、scale的等长轴和FP4。标准转置视图不匹配时，`GetWorkspaceSize`应返回`ACLNN_ERR_PARAM_INVALID`。
- 单例轴用例覆盖等价形状（如`[2,1,2]`与`[1,2,2]`）、ND和NZ权重、V1和V2、显式转置和视图转置，以及FP4和sigmoid。缺少K分组、batch不匹配或末维不为2的输入应被拒绝。

## Kernel UT

在ops-nn根目录执行：

```bash
bash build.sh -u --opkernel --ops=quant_matmul_activation_quant --soc=ascend950
```

用例见[Kernel UT CSV](ut/op_kernel/test_quant_matmul_activation_quant.csv)。修改输入或计算参数时，须同步更新期望值`expectedY`和`expectedYScale`。

CSV覆盖A/B转置、FP8/FP4的ND与NZ输入、`scaleAlg`、`roundMode`和`dstTypeMax`。FP4的`expectedY`按逻辑元素保存未打包的4位编码，驱动负责GM输入打包和输出解包。

sigmoid用例使用`x3Pattern=signed_coordinate`：BF16输入`x3`按`(3 * row + col) % 6`循环取`{-2, -0.5, 0, 0.5, 1, 2}`，覆盖符号、零值和行列偏移；`expectedY`及`expectedYScale`保存每一行的期望值。其他激活使用`x3Pattern=none`。

## TTK ST

| 测试入口 | 子命令 | 用例文件 | 融合算子的设备耗时列 |
| --- | --- | --- | --- |
| Kernel | `kernel` | [Kernel CSV](st/arch35/ttk_kernel_quant_matmul_activation_quant_st.csv) | `dyn_perf_us`（默认动态编译） |
| ACLNN ND | `aclnn` | [ND CSV](st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv) | `op_perf_us` |
| ACLNN NZ | `aclnn` | [NZ CSV](st/arch35/ttk_aclnn_quant_matmul_activation_quant_weight_nz_st.csv) | `op_perf_us` |
| E2E | `e2e` | [E2E CSV](st/arch35/ttk_e2e_quant_matmul_activation_quant_st.csv) | `eager_device_perf_us` |

以ACLNN ND为例，在ops-test-kit目录运行：

```bash
cd /path/to/ops-test-kit
OP_TESTS=/path/to/ops-nn/matmul/quant_matmul_activation_quant/tests
python3 -m ttk aclnn \
  -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" --seed 42
```

其他入口替换子命令和CSV即可。默认使用CPU Golden进行精度检查，结果CSV的`precision_status`应为`PASS`。

sigmoid + mul使用二维输入，且`x3`的形状须与`y`相同，详见[sigmoid场景](../README.md#sigmoid场景)。

- Kernel用例覆盖完整tile、多tile尾块、单行，以及E4M3FN/BLAS、E5M2/OCP，用于回归`x3`的UB复用和带mask的尾块计算。
- API和Tiling用例验证以下输入会被拒绝：任一矩阵带batch维（包括大小为1的维度）、`x3`带额外维度，或非sigmoid激活传入`x3`。
- sigmoid正向ST使用二维矩阵。其他激活继续覆盖省略`x3`的batch场景。

本算子的E2E交付和验收范围为PyTorch eager，仅验证单算子调用：

```bash
python3 -m ttk e2e \
  -i "$OP_TESTS/st/arch35/ttk_e2e_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" --golden-mode Enable --seed 42 \
  --eager=true -c=false -d=false -o e2e_results.csv
```

检查`eager_precision`有精度结果，且`precision_status`为`PASS`。静态图和动态图不在交付范围内。

### 用例配置

- Kernel NZ：`input_shapes`填写物理形状，`input_ori_shapes`填写逻辑形状，`input_ori_formats`填写`ND`。
- ACLNN NZ：`tensor_view_shapes`填写逻辑形状，`tensor_storage_shapes`填写完整物理形状。例如FP8权重的逻辑形状为`(batch,R,C)`时，物理形状为`(batch,ceil(C/32),ceil(R/16),16,32)`。
- E2E：CSV中的API名称为`cann_ops_nn.ops.quant_matmul_activation_quant`，Tensor列按`x1,x2,x2_scale,x1_scale,bias,x3`排列，标量参数放入`attributes`按名称绑定。Torch接口中`x3`位于`dst_type_max`之后，CSV的Tensor列不需要为标量参数补占位；矩阵的逻辑视图为`(...,M,K)`、`(...,K,N)`。
- E2E旧激活用例省略末尾`x3`，使用前5个Tensor槽位；`silu_nz_optional_defaults_e5m2_e4m3`保留第6槽为`None`，覆盖显式空值。sigmoid用例使用第6槽传入BF16 `x3`。
- 输出类型在Kernel CSV中使用`y_dtype`，ACLNN CSV中使用`yDtype`，E2E CSV中使用`output_dtype`。这些是测试配置字段；ACLNN接口本身通过输出Tensor描述数据类型。枚举值按对应CSV填写。
- E2E FP4使用uint8打包，并设置`x1_dtype=x2_dtype=output_dtype=296`，布局见[PyTorch接口文档](../docs/torchapi_quant_matmul_activation_quant.md#布局与存储)。

## 与独立算子组合的性能对比

使用`--golden-mode NPU`，可测量以下参考组合。QBMM表示量化矩阵乘，括号内为该步骤的输出类型。

| 激活 | 量化类型 | 参考组合 |
| --- | --- | --- |
| SiLU | MXFP8 | QBMM(FP32) → SiLU(FP32) → Cast(BF16) → DynamicMxQuant |
| sigmoid + mul | MXFP8 | QBMM(BF16) → sigmoid(BF16) → mul(BF16输入`x3`，BF16输出) → DynamicMxQuant |
| SwiGLU | MXFP8 | QBMM(BF16) → SwigluMxQuant（沿尾轴等分，激活左半部分，mode=0） |
| GELU（tanh/erf） | MXFP8、MXFP4 | QBMM(FP32) → GELU(FP32) → Cast(BF16) → DynamicMxQuant |

Cast计入参考组合耗时。SwiGLU需要`torch_npu.npu_swiglu_mx_quant`。量化算法和舍入模式按用例参数执行。

例如，测试ND SiLU：

```bash
python3 -m ttk aclnn \
  -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" -t silu_nd_e4m3_n65 \
  --golden-mode NPU --devices 0 --process-count 1 \
  --warmup true --run 1 --seed 42 -o silu_perf.csv
```

测试其他激活时，`-t`可改为`swiglu_s0_y36`、`gelu_tanh_s0_y36`或`gelu_tanh_fp4_nd_rint`。保持预热开启，使NZ权重的格式准备在预热阶段完成。

结果中的`golden_device_perf_us`为参考组合的设备耗时；融合算子的耗时列见上表。加速比按“参考组合耗时 / 融合算子耗时”计算，仅使用`golden_source=npu`且耗时为正的结果。`golden_kernel_details`记录参考组合中各算子的耗时和调用次数。

当前NPU Golden每例采样一次，`--run`不增加其采样次数。需要多次采样时，串行重复运行、分别保存结果并取中位数。

## 精度判定

精度验收使用默认的`--golden-mode Enable`，由CPU Golden计算期望值。NPU模式以独立算子的组合输出为参考；其中SwiGLU在激活前舍入为BF16，sigmoid组合在MatMul、sigmoid和mul后保留BF16输出。融合sigmoid分支在FP32中完成sigmoid及乘法，仅在量化前转BF16，因此NPU模式用于性能对比。

ST分别检查`y`和`y_scale`：形状和dtype一致，至少99.9%的元素误差不超过1 ULP，NaN位置及Inf符号匹配，正负零视为相等。FP4按解包后的元素比较。Kernel UT使用固定输入并严格比较期望结果。
