# ST — ApplyAdamD（ATK 用例集，kernel 后端）

> 本目录保留 ATK **`kernel` 后端**精度用例，覆盖已部署 `ApplyAdamD` 的 host tiling + AI Core kernel：
> 经 `aclopCompileAndExecuteV2("ApplyAdamD", 10in/3out/2attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS)`
> 单算子动态执行在真机（ascend910b / dav-2201）launch 已部署的自定义内核，与 `cpu` 后端（手写 FP32 融合 Adam golden）对拍。
> aclnn 两段式调用路径由 `examples/test_aclnn_apply_adam_d.cpp` 覆盖。

## 文件

| 文件 | 作用 |
|---|---|
| `atk_ApplyAdamD.json` | ATK 用例集（约定唯一 ST 工件）：3 dtype × useNesterov{0,1} + 边界(rank0/1/2/4/empty) + 极值(grad=0 / ε=1e-12 / β1_power=1e-6)，判据 `single_bm`（MERE/MARE 浮点社区标准，含 atol 组合容差），`perf=not_key`（软目标）。 |
| `executor_ApplyAdamD.py` | ATK 执行器：`cpu_apply_adam_d`（FP32 融合 Adam golden，BaseApi）+ `kernel_apply_adam_d`（KernelBaseApi，子进程隔离 ACL 上下文经 `aclop_runner.py` 真机 launch）。 |
| `aclop_runner.py` | 独立 ACL 单算子 launch 原语（ctypes → `libascendcl.so` + `libacl_op_compiler.so`）；派发的实现（custom_nn vs builtin）由 `ASCEND_CUSTOM_OPP_PATH` 在 OPP 层决定。 |

## 运行

需先把 `ApplyAdamD` 自定义算子包部署到 OPP 并 `export ASCEND_CUSTOM_OPP_PATH=<vendors/custom_nn>`，再用 ATK 跑该用例集的 `--task accuracy`（kernel vs cpu）。真机实测 **18/18 = 100% 通过**（误差远低于 spec 阈值）。
