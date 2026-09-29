# QuantMatmulActivationQuant测试

本目录同时覆盖SwiGLU的MXFP8通路，以及`gelu_tanh` / `gelu_erf`的MXFP8、MXFP4通路。设备用例和CPU golden均位于仓内，不依赖桌面脚本或个人目录。

| 目录 | 用途 |
| --- | --- |
| `ut/op_api` | ND与WeightNz接口分别使用独立的C++/CSV，包含127条ND和75条WeightNz用例 |
| `ut/op_host` | InferShape和Host tiling回归 |
| `ut/op_kernel` | CSV保存用例、tiling参数和期望输出，公共工具头负责加载、校验及执行；`arch35/test_quant_matmul_activation_quant.cpp`统一四组dtype/layout测试 |
| `assets/spec.py` | 自包含的NumPy golden、FP8/FP4/E8M0编解码、比较器及TTK kernel/ACLNN/PyTorch E2E适配 |
| `st/arch35` | 54条kernel、58条ACLNN（49条ND、9条WeightNZ）、30条E2E回归配置；GELU覆盖FP8/FP4 ND与WeightNZ，SwiGLU覆盖FP8 ND与WeightNZ |

## Kernel UT接入

本仓从算子根`CMakeLists.txt`调用`AddOpTestCase`，沿用`cmake/ut.cmake`的源码拷贝、tiling stub和`MULTI_KERNEL_TARGET`路径，并显式声明`quant_batch_matmul_v3`依赖。同一个测试源文件编译为四组dtype/layout对象，使用独立的kernel符号和GoogleTest suite，共17条用例。

`arch35/quant_matmul_activation_quant_cpu_debug_stub.h`仅在CPU调试测试中加载：复用ops-tensor的Blaze兼容定义，将GM-to-L1拷贝转发到模拟器实现，并分别映射UB、L1、L0A/B/C、bias和MX scale buffer的地址。测试执行复用ops-tensor的`kernel_ut_runner.h`检查模拟器子进程错误，PV符号只链接一份。

这里不需要`tiling_registry_stub.cpp`：NN的`AddOpTestCase`检测到现有`quant_matmul_activation_quant_tiling_def.h`后会直接强制包含该文件，跳过注册表生成tiling头的流程；`GET_TILING_DATA_WITH_STRUCT`按指定类型拷贝生产POD，与入口的`REGISTER_NONE_TILING`配合使用，无须另外维护同一结构的注册字段。

算子使用独立的tiling POD：batch结构为104字节，非batch结构为56字节。Host在x1、x2的batch总数均为1时选择`TPL_WITHOUT_BATCH`，包括二维矩阵以及前导batch维全为1的输入。Kernel UT按CSV的batch维度选择对应入口及结构大小，现有用例中16条走非batch、1条走batch；Host UT另外检查序列化大小、batch分发及前导维全为1时的bias/广播形状。

用例由`ut/op_kernel/test_quant_matmul_activation_quant.csv`驱动，组织方式参考`block_attn_res_update`：公共`test_*_utils.h`返回用例列表和加载错误，统一测试入口按`socVersion` / `kernelUtTarget`选择用例并注册参数化测试。缺文件、错误表头、错列、重复名称、非法参数或没有匹配用例都会使CSV加载测试失败。

| CSV字段 | 含义 |
| --- | --- |
| `kernelUtTarget` | `QMMAQ_E4M3` / `QMMAQ_E5M2` / `QMMAQ_E5M2_GELU` / `QMMAQ_E4M3_WEIGHT_NZ`，分别对应9 / 2 / 4 / 2条用例及其编译dtype |
| `m,n,k,batchA,batchB` | 矩阵尺寸和四维batch；batch维度用空格分隔 |
| `activationType,scaleAlg,x1Dtype,x2Dtype,yDtype,transposeX2,fullLoad,biasMode,numBlocks` | 激活、量化、dtype和执行参数；dtype使用GE编码35/36，biasMode为0无bias、1 `[N]`、2 `[B,1,N]` |
| `baseM,baseN,baseK,kL1,scaleKL1,nBufferNum,dbL0C,mTailTile,nTailTile` | 构造tiling的显式字段；A/B共用kL1，尾轮tile直接由CSV控制，避免在C++测试中克隆并篡改case，也避免将整个结构体的ABI字节串固化到表格 |
| `dataPattern` | `batch_constant`保留原确定性数据；`coordinate`随batch、M/N/K坐标及K组变化；`subnormal_bias`用正常FP32 bias构造非零BF16 subnormal激活值；`gate_subnormal`检查FP32 subnormal gate在SiLU的FTZ Div中清零 |
| `expectedY,expectedYScale` | 原始字节用空格分隔；`coordinate`按batch/M顺序保存全部行，其他模式每个batch保存一行，检查时覆盖全部M行 |

输入由工具头中的固定规则生成，期望字节按`assets/spec.py`的`reference`计算。修改尺寸、输入规则或计算属性时须同步更新期望值。无CPU模拟器宏时仍可编译CSV加载部分，kernel执行测试明确标记为跳过。

在具备CANN、Ascend950 tikicpulib、匹配ops-tensor/Blaze的Linux构建环境运行：

```bash
bash build.sh -u --opkernel --ops=quant_matmul_activation_quant --soc=ascend950
```

17条Kernel UT保留GELU基础回归，以及SwiGLU的N=128、WeightNZ FF/FT和subnormal gate覆盖。原始N不满足64对齐的SwiGLU直调用例已移除；接口、图推导和tiling CSV将这些尺寸作为拒绝用例。设备ST的SwiGLU用例使用64对齐的原始N，继续覆盖输出32列尾块、五/六维广播、K尾块、WeightNZ、bias、两种FP8输出和全载/非全载路径。

## TTK ST

`assets/spec.py`顶部通过`__spec__`注册四个入口，注册名分别与CSV的`op_name` / `api_name`完全一致：

| 通路 | 注册名 | TestSpec |
| --- | --- | --- |
| kernel | `quant_matmul_activation_quant` | `QuantMatmulActivationQuantTestSpec` |
| ACLNN | `aclnnQuantMatmulActivationQuant` | `QuantMatmulActivationQuantAclnnTestSpec` |
| ACLNN WeightNZ | `aclnnQuantMatmulActivationQuantWeightNz` | `QuantMatmulActivationQuantAclnnTestSpec` |
| E2E | `cann_ops_nn.ops.quant_matmul_activation_quant` | `QuantMatmulActivationQuantE2ETestSpec` |

E2E使用`cann_ops_nn.ops`导出的函数。不要写成`cann_ops_nn.ops.matmul.quant_matmul_activation_quant.quant_matmul_activation_quant`：`matmul`包已将同名属性导出为函数，TTK执行器逐级访问属性时会报`'function' object has no attribute 'quant_matmul_activation_quant'`。

完整测试需要Linux、CANN、Ascend950，以及包含本次修改的算子包和对应ops-tensor；E2E还需匹配的`torch`、`torch_npu`、`cann_ops_nn`扩展。TTK的FP8/FP4/E8M0输入和设备输出转换需要`ml-dtypes`、`en-dtypes>=0.0.4`。先加载目标CANN环境，再从ops-test-kit源码目录执行；将以下两个路径替换为Linux环境中的实际路径：

```bash
cd /path/to/ops-test-kit
OP_TESTS=/path/to/ops-nn/matmul/quant_matmul_activation_quant/tests
python3 -m ttk info
python3 -m ttk kernel -i "$OP_TESTS/st/arch35/ttk_kernel_quant_matmul_activation_quant_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
python3 -m ttk aclnn -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
python3 -m ttk aclnn -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_weight_nz_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
python3 -m ttk e2e -i "$OP_TESTS/st/arch35/ttk_e2e_quant_matmul_activation_quant_st.csv" --plugin "$OP_TESTS/assets/spec.py" --seed 20260907
```

本地检查的ops-test-kit支持keyword-only tensor：E2E CSV的tensor顺序为`x1,x2,x2_scale,x1_scale,bias`，后两者按Python API签名装配为关键字参数。E2E显式传入x1_scale，绝不使用隐式单位scale；FT和WeightNZ的物理存储场景由kernel/ACLNN用例覆盖。三条通路的y dtype来源不同：kernel用attrs的`y_dtype`（GE 35/36/40），ACLNN用attrs的`yDtype`（GE 35/36/40，必须显式携带——TTK会把yOut提升为float32占位张量，golden无法从张量本身取回MX类型），E2E的FP8用attrs的`output_dtype`（PyTorch 23/24），FP4则以`uint8`双nibble打包并显式传入`output_dtype/x1_dtype/x2_dtype=296`；三者不能混用。

原生Windows不能直接执行当前版本TTK的完整CLI：入口依赖Linux的`resource`模块。`--validate`也会经过该入口；它只校验用例配置，不代表设备执行通过。`--cpu`不适用于本算子的NPU扩展实现。

## Golden与精度口径

golden按K的32元素组反量化，使用FP64保存参考操作数和MatMul中间值，在MatMul输出处转回FP32后加bias，再执行激活和输出MX量化。这样可避免合法FP8/FP4 × E8M0在反量化时提前溢出；它是数值参考，不模拟Cube的逐块累加舍入。kernel在Cube首次K块的MMAD中用bias初始化累加结果，后续K块继续累加。SwiGLU和两种GELU均保留原epilogue的BF16中间结果；SwiGLU的舍入位置在激活完成后、MX scale选择之前。FP8使用RNE，OCP和BLAS独立生成scale。FP4支持`rint`、`floor`、`round`，并分别建模OCP与`scaleAlg=2`的动态dtype range/cuBLAS分支。全零组及全padding半组的E8M0字节为0（数值`2^-127`）。swiglu激活链按950 VF默认`--cce-ftz=true`建模：Exp/Div深刷subnormal，最终Mul与BF16舍入保留subnormal。

`reference`返回MX数据码和E8M0 scale原始字节。kernel/ACLNN的FP4结果使用原生`float4_e2m1`数组，PyTorch E2E的FP4结果按低nibble在前打包为`uint8`；FP8结果保持相应FP8 dtype。NumPy输入返回NumPy数组，Torch输入返回CPU Torch张量，避免golden依赖TTK内部模块或CPU E8M0张量支持。

ST使用CANN浮点计算类精度标准，与TTK `core_modules/comparison/resolve.py`的FP8参数一致：

| 输出 | rtol | atol | 最低匹配比例 | 最大绝对误差硬上限 |
| --- | ---: | ---: | ---: | ---: |
| E4M3FN | 0.25 | 0.0625 | 0.99 | 4 |
| E5M2 | 0.5 | 0.125 | 0.99 | 8 |
| E2M1 | - | - | 1.0 | 原始码逐位相等 |

阈值应用于量化后的FP8数值；FP4原始码、E8M0 shape和每个scale均独立严格核对，NaN payload不作要求。不能通过改变scale、反向改变量化值，使解量化乘积相近来绕过scale校验。MatMul归约或激活逼近若使amax跨scale边界，应保留失败dump并分析，不能自动放宽scale检查。Kernel UT对确定性输入使用严格比较。
