# FastHadamard 算子

## 概述

本算子实现快速哈达玛变换（Fast Hadamard Transform, FHT），对输入张量最后一维（长度 `n` 必须为 2 的幂）按行做长度为 `n` 的哈达玛变换，其余维度折叠为 `batch` 并行处理。变换为非原地操作：结果写入独立输出张量 `out`。

本算子以「生态最简算子」方式交付：单一 `.cpp` 内含 device kernel、`<<<>>>` 启动与 `TORCH_LIBRARY` 注册，编译进 `ascend_ops_nn` PyTorch 扩展后，通过 `torch.ops.ascend_ops_nn.fast_hadamard` 直接调用。

kernel 计算逻辑改编自经过验证的 pto-ISA 参考实现（`pto-kernels/examples/jit_cpp/fast_hadamard/standard/fast_hadamard.cpp`），pto ISA 头文件随 CANN 软件包发布于 `${ASCEND_TOOLKIT_HOME}/include/pto`，无需额外引入第三方依赖。

## 支持的 AI 处理器

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |

> 说明：A2（`ascend910b`）与 A3（`ascend910_93`）共享 `dav-2201` 架构与 pto `npu/a2a3` 后端，同一二进制适用于两者；当前在 A2 上完成验证。kernel 内 UB 布局按 A2/A3 的 184KB Unified Buffer 规划，`blockDim` 采用 A2/A3 的向量核数（40）作为启动网格大小，`batch` 在 kernel 内按实际核数切分，故该常量仅影响启动规模、不影响正确性。

## 算子规格

- 数学表达式（每一行，未归一化）：

  $$ y = H_n \cdot x,\quad H_n = \begin{bmatrix} H_{n/2} & H_{n/2} \\ H_{n/2} & -H_{n/2} \end{bmatrix},\ H_1 = [1] $$

- `torch.ops` 接口：`torch.ops.ascend_ops_nn.fast_hadamard(x: Tensor, out: Tensor) -> int`
  - 输入 `x`：`float16`，`ND`，shape `[..., n]`，`n` 为 2 的幂，`2 < n <= 16384`。
  - 输出 `out`：`float16`，`ND`，shape 与 `x` 相同；由调用方预分配，结果写入其中（返回值为状态码 0）。

## 目录结构

```bash
fast_hadamard/
├── CMakeLists.txt                  // add_sources()，将算子编入 ascend_ops_nn 扩展
├── README.md
├── fast_hadamard.cpp               // device kernel（pto ISA）+ <<<>>> 启动 + torch 注册
├── examples
│   ├── benchmark.py                // torch.ops 时延基准
│   └── rotate_quant_binding.cc     // benchmark 可选 rotate_quant 对比项的 PyTorch 绑定
└── tests
    └── test_fast_hadamard.py       // torch.ops 正确性测试（对照 CPU 参考）
```

## 编译与安装

`--experimental` 会触发 PyTorch 扩展构建：所有 `add_sources()` 算子被打包进 `ascend_ops_nn` 扩展（`_C.abi3.so`）。

```bash
cd ${git_clone_path}
bash build.sh --pkg --experimental --soc=ascend910b \
    --ops=fast_hadamard,fast_hadamard_quant,fast_hadamard_dynamic_quant
# 安装生成的 wheel（注意 --no-deps，避免升级容器内已固定的 torch/TorchNPU）
pip install --no-deps --force-reinstall build_out/ascend_ops_nn-*.whl
```

> `--soc` 支持 `ascend910b`（A2）与 `ascend910_93`（A3），均映射到 `dav-2201`。

## 调用示例

```python
import torch
import torch_npu          # 注册 npu 后端
import ascend_ops_nn      # 注册 torch.ops.ascend_ops_nn.*

x = torch.randn(8, 1024, dtype=torch.float16).npu()
out = torch.empty_like(x)
torch.ops.ascend_ops_nn.fast_hadamard(x, out)
```

## 正确性测试

```bash
ASCEND_RT_VISIBLE_DEVICES=<free-id> python3 tests/test_fast_hadamard.py
```
