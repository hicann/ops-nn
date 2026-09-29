# QuantMatmulActivationQuant 测试

本目录同时覆盖 SwiGLU 的 MXFP8 通路，以及 `gelu_tanh` / `gelu_erf` 的 MXFP8、MXFP4 通路。设备用例和 CPU golden 均位于仓内，不依赖桌面脚本或个人目录。

| 目录 | 用途 |
| --- | --- |
| `ut/op_api` | ND 与 WeightNz 接口分别使用独立的 C++/CSV，包含 127 条 ND 和 75 条 WeightNz 用例 |
| `ut/op_host` | InferShape 和 Host tiling 回归 |
| `ut/op_kernel` | CSV 保存用例、tiling 参数和期望输出，公共工具头负责加载、校验及执行；`arch35/test_quant_matmul_activation_quant.cpp` 统一四组 dtype/layout 测试 |
| `assets/spec.py` | 自包含的 NumPy golden、FP8/FP4/E8M0 编解码、比较器及 TTK kernel/ACLNN/PyTorch E2E 适配 |
| `st/arch35` | 54 条 kernel、58 条 ACLNN（49 条 ND、9 条 WeightNZ）、30 条 E2E 回归配置；GELU 覆盖 FP8/FP4 ND 与 WeightNZ，SwiGLU 覆盖 FP8 ND 与 WeightNZ |

## Kernel UT 接入

本仓从算子根 `CMakeLists.txt` 调用 `AddOpTestCase`，沿用 `cmake/ut.cmake` 的源码拷贝、tiling stub 和 `MULTI_KERNEL_TARGET` 路径，并显式声明 `quant_batch_matmul_v3` 依赖。同一个测试源文件编译为四组 dtype/layout 对象，使用独立的 kernel 符号和 GoogleTest suite，共 17 条用例。

`arch35/quant_matmul_activation_quant_cpu_debug_stub.h` 仅在 CPU 调试测试中加载：复用 ops-tensor 的 Blaze 兼容定义，将 GM-to-L1 拷贝转发到模拟器实现，并分别映射 UB、L1、L0A/B/C、bias 和 MX scale buffer 的地址。测试执行复用 ops-tensor 的 `kernel_ut_runner.h` 检查模拟器子进程错误，PV 符号只链接一份。

这里不需要 `tiling_registry_stub.cpp`：NN 的 `AddOpTestCase` 检测到现有 `quant_matmul_activation_quant_tiling_def.h` 后会直接强制包含该文件，跳过注册表生成 tiling 头的流程；`GET_TILING_DATA_WITH_STRUCT` 按指定类型拷贝生产 POD，与入口的 `REGISTER_NONE_TILING` 配合使用，无须另外维护同一结构的注册字段。

算子使用独立的 tiling POD：batch 结构为 104 字节，非 batch 结构为 56 字节。Host 在 x1、x2 的 batch 总数均为 1 时选择 `TPL_WITHOUT_BATCH`，包括二维矩阵以及前导 batch 维全为 1 的输入。Kernel UT 按 CSV 的 batch 维度选择对应入口及结构大小，现有用例中 16 条走非 batch、1 条走 batch；Host UT 另外检查序列化大小、batch 分发及前导维全为 1 时的 bias/广播形状。

用例由 `ut/op_kernel/test_quant_matmul_activation_quant.csv` 驱动，组织方式参考 `block_attn_res_update`：公共 `test_*_utils.h` 返回用例列表和加载错误，统一测试入口按 `socVersion` / `kernelUtTarget` 选择用例并注册参数化测试。缺文件、错误表头、错列、重复名称、非法参数或没有匹配用例都会使 CSV 加载测试失败。

| CSV 字段 | 含义 |
| --- | --- |
| `kernelUtTarget` | `QMMAQ_E4M3` / `QMMAQ_E5M2` / `QMMAQ_E5M2_GELU` / `QMMAQ_E4M3_WEIGHT_NZ`，分别对应 9 / 2 / 4 / 2 条用例及其编译 dtype |
| `m,n,k,batchA,batchB` | 矩阵尺寸和四维 batch；batch 维度用空格分隔 |
| `activationType,scaleAlg,x1Dtype,x2Dtype,yDtype,transposeX2,fullLoad,biasMode,numBlocks` | 激活、量化、dtype 和执行参数；dtype 使用 GE 编码 35/36，biasMode 为 0 无 bias、1 `[N]`、2 `[B,1,N]` |
| `baseM,baseN,baseK,kL1,scaleKL1,nBufferNum,dbL0C,mTailTile,nTailTile` | 构造 tiling 的显式字段；A/B 共用 kL1，尾轮 tile 直接由 CSV 控制，避免在 C++ 测试中克隆并篡改 case，也避免将整个结构体的 ABI 字节串固化到表格 |
| `dataPattern` | `batch_constant` 保留原确定性数据；`coordinate` 随 batch、M/N/K 坐标及 K 组变化；`subnormal_bias` 用正常 FP32 bias 构造非零 BF16 subnormal 激活值；`gate_subnormal` 检查 FP32 subnormal gate 在 SiLU 的 FTZ Div 中清零 |
| `expectedY,expectedYScale` | 原始字节用空格分隔；`coordinate` 按 batch/M 顺序保存全部行，其他模式每个 batch 保存一行，检查时覆盖全部 M 行 |

输入由工具头中的固定规则生成，期望字节按 `assets/spec.py` 的 `reference` 计算。修改尺寸、输入规则或计算属性时须同步更新期望值。无 CPU 模拟器宏时仍可编译 CSV 加载部分，kernel 执行测试明确标记为跳过。

在具备 CANN、Ascend950 tikicpulib、匹配 ops-tensor/Blaze 的 Linux 构建环境运行：

```bash
bash build.sh -u --opkernel --ops=quant_matmul_activation_quant --soc=ascend950
```

17 条 Kernel UT 保留 GELU 基础回归，以及 SwiGLU 的 N=128、WeightNZ FF/FT 和 subnormal gate 覆盖。原始 N 不满足 64 对齐的 SwiGLU 直调用例已移除；接口、图推导和 tiling CSV 将这些尺寸作为拒绝用例。设备 ST 的 SwiGLU 用例使用 64 对齐的原始 N，继续覆盖输出 32 列尾块、五/六维广播、K 尾块、WeightNZ、bias、两种 FP8 输出和全载/非全载路径。

## TTK ST

`assets/spec.py` 顶部通过 `__spec__` 注册四个入口，注册名分别与 CSV 的 `op_name` / `api_name` 完全一致：

| 通路 | 注册名 | TestSpec |
| --- | --- | --- |
| kernel | `quant_matmul_activation_quant` | `QuantMatmulActivationQuantTestSpec` |
| ACLNN | `aclnnQuantMatmulActivationQuant` | `QuantMatmulActivationQuantAclnnTestSpec` |
| ACLNN WeightNZ | `aclnnQuantMatmulActivationQuantWeightNz` | `QuantMatmulActivationQuantAclnnTestSpec` |
| E2E | `cann_ops_nn.ops.quant_matmul_activation_quant` | `QuantMatmulActivationQuantE2ETestSpec` |

E2E 使用 `cann_ops_nn.ops` 导出的函数。不要写成 `cann_ops_nn.ops.matmul.quant_matmul_activation_quant.quant_matmul_activation_quant`：`matmul` 包已将同名属性导出为函数，TTK 执行器逐级访问属性时会报 `'function' object has no attribute 'quant_matmul_activation_quant'`。

完整测试需要 Linux、CANN、Ascend950，以及包含本次修改的算子包和对应 ops-tensor；E2E 还需匹配的 `torch`、`torch_npu`、`cann_ops_nn` 扩展。TTK 的 FP8/FP4/E8M0 输入和设备输出转换需要 `ml-dtypes`、`en-dtypes>=0.0.4`。先加载目标 CANN 环境，再从 ops-test-kit 源码目录执行；将以下两个路径替换为 Linux 环境中的实际路径：

```bash
cd /path/to/ops-test-kit
OP_TESTS=/path/to/ops-nn/matmul/quant_matmul_activation_quant/tests
python3 -m ttk info
python3 -m ttk kernel -i "$OP_TESTS/st/arch35/ttk_kernel_quant_matmul_activation_quant_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
python3 -m ttk aclnn -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
python3 -m ttk aclnn -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_weight_nz_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
python3 -m ttk e2e -i "$OP_TESTS/st/arch35/ttk_e2e_quant_matmul_activation_quant_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
```

本地检查的 ops-test-kit 支持 keyword-only tensor：E2E CSV 的 tensor 顺序为 `x1,x2,x2_scale,x1_scale,bias`，后两者按 Python API 签名装配为关键字参数。E2E 显式传入 x1_scale，绝不使用隐式单位 scale；FT 和 WeightNZ 的物理存储场景由 kernel/ACLNN 用例覆盖。三条通路的 y dtype 来源不同：kernel 用 attrs 的 `y_dtype`（GE 35/36/40），ACLNN 用 attrs 的 `yDtype`（GE 35/36/40，必须显式携带——TTK 会把 yOut 提升为 float32 占位张量，golden 无法从张量本身取回 MX 类型），E2E 的 FP8 用 attrs 的 `output_dtype`（PyTorch 23/24），FP4 则以 `uint8` 双 nibble 打包并显式传入 `output_dtype/x1_dtype/x2_dtype=296`；三者不能混用。

原生 Windows 不能直接执行当前版本 TTK 的完整 CLI：入口依赖 Linux 的 `resource` 模块。`--validate` 也会经过该入口；它只校验用例配置，不代表设备执行通过。`--cpu` 不适用于本算子的 NPU 扩展实现。

## Golden 与精度口径

golden 按 K 的 32 元素组反量化，使用 FP64 保存参考操作数和 MatMul 中间值，在 MatMul 输出处转回 FP32 后加 bias，再执行激活和输出 MX 量化。这样可避免合法 FP8/FP4 × E8M0 在反量化时提前溢出；它是数值参考，不模拟 Cube 的逐块累加舍入。kernel 在 Cube 首次 K 块的 MMAD 中用 bias 初始化累加结果，后续 K 块继续累加。SwiGLU 和两种 GELU 均保留原 epilogue 的 BF16 中间结果；SwiGLU 的舍入位置在激活完成后、MX scale 选择之前。FP8 使用 RNE，OCP 和 BLAS 独立生成 scale。FP4 支持 `rint`、`floor`、`round`，并分别建模 OCP 与 `scaleAlg=2` 的动态 dtype range/cuBLAS 分支。全零组及全 padding 半组的 E8M0 字节为 0（数值 `2^-127`）。swiglu 激活链按 950 VF 默认 `--cce-ftz=true` 建模：Exp/Div 深刷 subnormal，最终 Mul 与 BF16 舍入保留 subnormal。

`reference` 返回 MX 数据码和 E8M0 scale 原始字节。kernel/ACLNN 的 FP4 结果使用原生 `float4_e2m1` 数组，PyTorch E2E 的 FP4 结果按低 nibble 在前打包为 `uint8`；FP8 结果保持相应 FP8 dtype。NumPy 输入返回 NumPy 数组，Torch 输入返回 CPU Torch 张量，避免 golden 依赖 TTK 内部模块或 CPU E8M0 张量支持。

ST 使用 CANN 浮点计算类精度标准，与 TTK `core_modules/comparison/resolve.py` 的 FP8 参数一致：

| 输出 | rtol | atol | 最低匹配比例 | 最大绝对误差硬上限 |
| --- | ---: | ---: | ---: | ---: |
| E4M3FN | 0.25 | 0.0625 | 0.99 | 4 |
| E5M2 | 0.5 | 0.125 | 0.99 | 8 |
| E2M1 | - | - | 1.0 | 原始码逐位相等 |

阈值应用于量化后的 FP8 数值；FP4 原始码、E8M0 shape 和每个 scale 均独立严格核对，NaN payload 不作要求。不能通过改变 scale、反向改变量化值，使解量化乘积相近来绕过 scale 校验。MatMul 归约或激活逼近若使 amax 跨 scale 边界，应保留失败 dump 并分析，不能自动放宽 scale 检查。Kernel UT 对确定性输入使用严格比较。

## 当前验证范围

这些是重点回归用例，不代表已经完成交付 STC 所要求的 1000 条泛化及七条网络性能验收。设备配置已补齐 GELU 的 FP8 ND/WeightNZ，以及 FP4 ND/WeightNZ 场景；真实设备精度回归仍需在 Ascend950 环境执行。

本地已使用 ops-test-kit 的实际 CSV 用例加载器、校验器和参数装配器检查全部 142 条 ST 配置，并逐条执行 NumPy golden 路径及输出 shape 核对。离线检查用源码中的 OpDef、ACLNN 头文件、Python 函数签名提供接口元数据，未加载安装后的 CANN 算子库；E2E 仅复现包导出结构和参数绑定，未执行 NPU 函数体。FP4 的 RNE 编码已与 `ml-dtypes` 全码点核对，WeightNZ 的 FP8/FP4 C0 分别按 32/64 处理。

真实 kernel CPU 模拟器的编译/执行、Torch 输入路径、TTK 完整 CLI、设备精度及性能仍需在目标环境执行并保存版本和日志。性能验收应另保存预热后每条路径三次原始耗时、平均值和同输入非融合基线。
