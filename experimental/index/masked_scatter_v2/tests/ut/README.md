# MaskedScatterV2 UT

- `op_host/test_masked_scatter_v2_tiling.cpp`：tiling UT（同 shape / dim-first 广播准入 /
  dim-last 广播拒绝 / 小广播回退 / dtype 白名单 / workspace 计数槽大小）。
- `op_kernel/`：kernel UT 占位（对齐仓内 AIV kernel UT 框架后补齐）。
- aclnn 两段式接口由构建系统从 def/tiling 自动生成（单 kernel 算子无需手写 op_api）。
- 端到端 200 case 泛化性能（torch 直调口径，msprof quick：warmup=3/repeats=1，
  npu:6 实测 geomean 1.115x / median 1.114x）见 PR 描述。
