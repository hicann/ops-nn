# FastHadamardQuant 算子

## 概述

本算子实现融合的快速哈达玛变换 + 静态 int4 量化（fp16 -> packed int4）。对输入张量最后一维（长度 `n` 为 2 的幂）按行做哈达玛变换，随后按 `scale`/`q_offset`（或分组的 `group_scales`/`group_offsets`）量化为 int4，并以 int8 字节打包存储（每字节 2 个 int4，偶数下标 -> 低 4 位，奇数下标 -> 高 4 位）。

本算子以「生态最简算子」方式交付，通过 `torch.ops.ascend_ops_nn.fast_hadamard_quant` 调用。kernel 计算逻辑取自经过验证的 pto-ISA 参考实现（`pto-kernels/fuse_int4_quant/fast_hadamard_quant.cpp` 与 `int4_cvt.hpp`），pto ISA 头文件随 CANN 软件包发布。

## 支持的 AI 处理器

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |

> 说明：A2（`ascend910b`）与 A3（`ascend910_93`）共享 `dav-2201` 架构与 pto `npu/a2a3` 后端，同一二进制适用于两者；当前在 A2 上完成验证。

## 算子规格

- 量化公式（每个元素）：`q = clamp(round(x * scale + q_offset), -8, 7)`，其中 `x` 为未归一化的哈达玛变换结果。
- `torch.ops` 接口：

  ```python
  torch.ops.ascend_ops_nn.fast_hadamard_quant(
      x: Tensor, group_scales: Tensor?, group_offsets: Tensor?,
      scale: float, group_size: int, q_offset: float, out: Tensor) -> int
  ```

  - 输入 `x`：`float16`，`ND`，shape `[..., n]`，`n` 为 2 的幂，`2 < n <= 16384`。
  - 输入 `group_scales`（可选）：`float16`，`ND`，分组量化 scale，shape `[groups]` 或 `[batch, groups]`。
  - 输入 `group_offsets`（可选）：`float16`，`ND`，分组量化 offset，shape 同上。
  - 属性 `scale`（float，默认 1.0）：`group_scales` 缺省时使用的统一 scale。
  - 属性 `group_size`（int，默认 0 表示整行）：沿 `n` 的量化分组大小，须为 `n` 的偶数因子。
  - 属性 `q_offset`（float，默认 0.0）：`group_offsets` 缺省时使用的统一 offset。
  - 输出 `out`：`int8`，`ND`，shape `[..., n/2]`，每字节打包 2 个 int4；由调用方预分配。

## 目录结构

```bash
fast_hadamard_quant/
├── CMakeLists.txt                       // add_sources()
├── README.md
├── fast_hadamard_quant.cpp              // device kernel（pto ISA）+ <<<>>> 启动 + torch 注册
└── tests
    └── test_fast_hadamard_quant.py      // torch.ops 正确性测试（与参考逐字节比对）
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
y = torch.empty(8, 512, dtype=torch.int8).npu()      # packed int4: [.., n/2]
# 统一 scale、整行量化（group_size=0、无 group_scales/offsets）
torch.ops.ascend_ops_nn.fast_hadamard_quant(x, None, None, 1.0, 0, 0.0, y)
```

## 正确性测试

```bash
ASCEND_RT_VISIBLE_DEVICES=<free-id> python3 tests/test_fast_hadamard_quant.py
```
