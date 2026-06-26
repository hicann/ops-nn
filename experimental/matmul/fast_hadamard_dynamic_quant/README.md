# FastHadamardDynamicQuant 算子

## 概述

本算子实现融合的快速哈达玛变换 + 动态 int4 量化（fp16 -> packed int4 + 每行 fp32 scale）。对输入张量最后一维按 `hadamard_n` 子块做哈达玛变换，按行（row-wise）计算动态量化 scale，再量化为 int4 并以 int8 字节打包（每字节 2 个 int4）。

本算子以「生态最简算子」方式交付，通过 `torch.ops.ascend_ops_nn.fast_hadamard_dynamic_quant` 调用。kernel 计算逻辑取自经过验证的 pto-ISA 参考实现（`pto-kernels/fuse_int4_dynamic_quant/fast_hadamard_dynamic_quant.cpp` 与 `int4_cvt.hpp`），并将 reduce/expand tile 按编译期 `kFullN` 模板化以修复大 batch 下的 UB 越界；pto ISA 头文件随 CANN 软件包发布。

## 支持的 AI 处理器

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |

> 说明：A2（`ascend910b`）与 A3（`ascend910_93`）共享 `dav-2201` 架构与 pto `npu/a2a3` 后端，同一二进制适用于两者；当前在 A2 上完成验证。

## 算子规格

- 每行 `scale = max_abs * (1/sqrt(hadamard_n)) / 7`，量化码 `q = clamp(round(transformed * 7 / max_abs), -8, 7)`，其中 `transformed` 为未归一化的分块哈达玛变换结果、`max_abs` 为该行绝对值最大值。
- `torch.ops` 接口：

  ```python
  torch.ops.ascend_ops_nn.fast_hadamard_dynamic_quant(
      x: Tensor, hadamard_n: int, out: Tensor, row_scales: Tensor) -> int
  ```

  - 输入 `x`：`float16`，`ND`，shape `[..., fullN]`，`fullN` 为 2 的幂、`64 <= fullN <= 16384`，且为 `hadamard_n` 的整数倍。
  - 属性 `hadamard_n`（int）：子块哈达玛长度，默认 0 表示整行（`= fullN`）；非 0 时须为 2 的幂且整除 `fullN`，`2 < hadamard_n <= 16384`。
  - 输出 `out`：`int8`，`ND`，shape `[..., fullN/2]`，每字节打包 2 个 int4；由调用方预分配。
  - 输出 `row_scales`：`float32`，`ND`，shape `[batch]`（`batch = numel(x)/fullN`），每行一个动态 scale；由调用方预分配。

## 目录结构

```bash
fast_hadamard_dynamic_quant/
├── CMakeLists.txt                               // add_sources()
├── README.md
├── fast_hadamard_dynamic_quant.cpp              // device kernel（pto ISA）+ <<<>>> 启动 + torch 注册
└── tests
    └── test_fast_hadamard_dynamic_quant.py      // torch.ops 正确性测试（scale + int4 码对照 CPU 参考）
```

公共的 Hadamard tile helper 与 fp16 -> packed int4 转换模板位于
`experimental/matmul/common/fast_hadamard/`。

## 编译与安装

```bash
cd ${git_clone_path}
bash build.sh --pkg --experimental --soc=ascend910b \
    --ops=fast_hadamard,fast_hadamard_quant,fast_hadamard_dynamic_quant
pip install --no-deps --force-reinstall build_out/ascend_ops_nn-*.whl
```

> `--soc` 支持 `ascend910b`（A2）与 `ascend910_93`（A3）。

## 调用示例

```python
import torch, torch_npu, ascend_ops_nn

x = torch.randn(8, 1024, dtype=torch.float16).npu()
y = torch.empty(8, 512, dtype=torch.int8).npu()        # packed int4: [.., fullN/2]
row_scales = torch.empty(8, dtype=torch.float32).npu()
# hadamard_n=0 -> 整行变换
torch.ops.ascend_ops_nn.fast_hadamard_dynamic_quant(x, 0, y, row_scales)
```

## 正确性测试

```bash
ASCEND_RT_VISIBLE_DEVICES=<free-id> python3 tests/test_fast_hadamard_dynamic_quant.py
```
