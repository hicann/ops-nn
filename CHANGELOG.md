# CHANGELOG

> 本文档记录各版本的重要变更，版本按时间倒序排列。

## v9.0.0-beta.2

发布日期：2026-03-30

本版本新增多项新增特性、问题修复及性能改进，支持最新的950硬件。
我们诚挚欢迎社区反馈，以进一步提升 ops-nn 的稳定性和功能完备性。
使用方式请参阅[官方文档](https://gitcode.com/cann/ops-nn/blob/master/README.md)。

[CANN 9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/)

```
版本目录说明如下：
├── aarch64                 # CPU为ARM类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
├── x86_64                   # CPU为X86类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
```


### 📌 版本配套

**CANN独立升级子包版本配套关系**

| CANN子包版本 | 版本源码标签   | 配套CANN版本|
|--|--|--|
| [cann-ops-math   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/ops-math/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| [cann-ops-nn   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/ops-nn/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| [cann-ops-cv   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/ops-cv/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| [cann-ops-transformer   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/ops-transformer/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| [cann-hccl   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/hccl/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| [cann-hixl   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/hixl/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |

**CANN开源子包版本配套关系**

| CANN子包版本                         | 版本源码标签                                                 | 配套CANN版本        |
| ------------------------------------ | ------------------------------------------------------------ | ------------------- |
| cann-opbase 9.0.0-beta.2             | [v9.0.0-beta.2](https://gitcode.com/cann/opbase/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-oam-tools   9.0.0-beta.2        | [v9.0.0-beta.2](https://gitcode.com/cann/oam-tools/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-asc-tools   9.0.0-beta.2        | [v9.0.0-beta.2](https://gitcode.com/cann/asc-tools/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-asc-devkit   9.0.0-beta.2       | [v9.0.0-beta.2](https://gitcode.com/cann/asc-devkit/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-pto-isa   9.0.0-beta.2          | [v9.0.0-beta.2](https://gitcode.com/cann/pto-isa/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-ge-compiler   9.0.0-beta.2      | [v9.0.0-beta.2](https://gitcode.com/cann/ge/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-ge-executor   9.0.0-beta.2      | [v9.0.0-beta.2](https://gitcode.com/cann/ge/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-graph-autofusion   9.0.0-beta.2 | [v9.0.0-beta.2](https://gitcode.com/cann/graph-autofusion/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-metadef   9.0.0-beta.2          | [v9.0.0-beta.2](https://gitcode.com/cann/metadef/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-dflow-executor   9.0.0-beta.2   | [v9.0.0-beta.2](https://gitcode.com/cann/ge/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-hcomm   9.0.0-beta.2            | [v9.0.0-beta.2](https://gitcode.com/cann/hcomm/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| cann-npu-runtime   9.0.0-beta.2      | [v9.0.0-beta.2](https://gitcode.com/cann/runtime/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |

### 🚀 关键特性

- 【工程能力】静态库工程适配  [!391](https://gitcode.com/cann/ops-nn/pull/391)
- 【工程能力】算子工程适配950 [!450](https://gitcode.com/cann/ops-nn/pull/450)
- 【工程能力】新增支持按模板编译特定算子kernel能力[#1097](https://gitcode.com/cann/ops-nn/issues/1097)
- 【新特性】950支持PyTorch view的算子融合优化 [#864](https://gitcode.com/cann/ops-nn/issues/864)
- 【新特性】950支持SIMD/SIMT新同构编程([#660](https://gitcode.com/cann/ops-nn/issues/660)、[#710](https://gitcode.com/cann/ops-nn/issues/710)、[#668](https://gitcode.com/cann/ops-nn/issues/668)、[#656](https://gitcode.com/cann/ops-nn/issues/656)、[#658](https://gitcode.com/cann/ops-nn/issues/658))

### 🐛 问题修复

- ScatterNd算子确定性计算按列分核偏移修复([!3137](https://gitcode.com/cann/ops-nn/pull/3137))
- scatter_elements 确定性模板([!3092](https://gitcode.com/cann/ops-nn/pull/3092))
- Matmul算子FP32场景下单核切K模板的精度（[!2651](https://gitcode.com/cann/ops-nn/pull/2651)）

## v8.5.0-beta.1

发布日期：2025-12-30

ops-nn 算子首个 Beta 版本 v8.5.0-beta.1 现已发布。
本版本引入了多项新增特性、问题修复及性能改进，目前仍处于测试阶段。
我们诚挚欢迎社区反馈，以进一步提升 ops-nn 的稳定性和功能完备性。
使用方式请参阅[官方文档](https://gitcode.com/cann/ops-nn/blob/master/README.md)。

### 🔗 版本地址

[CANN 8.5.0-beta 1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```
版本目录说明如下：
├── aarch64                 # CPU为ARM类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
├── x86_64                   # CPU为X86类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
```

### 📌 版本配套

**CANN独立升级子包版本配套关系**

| CANN子包版本 | 版本源码标签   | 配套CANN版本|
|--|--|--|
| [cann-ops-math   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-math/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-nn   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-nn/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-cv   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-cv/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-transformer   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-transformer/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hccl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hccl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hixl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hixl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

**CANN开源子包版本配套关系**

| CANN子包版本                         | 版本源码标签                                                 | 配套CANN版本        |
| ------------------------------------ | ------------------------------------------------------------ | ------------------- |
| cann-opbase 8.5.0-beta.1             | [v8.5.0-beta.1](https://gitcode.com/cann/opbase/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-oam-tools   8.5.0-beta.1        | [v8.5.0-beta.1](https://gitcode.com/cann/oam-tools/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-asc-tools   8.5.0-beta.1        | [v8.5.0-beta.1](https://gitcode.com/cann/asc-tools/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-asc-devkit   8.5.0-beta.1       | [v8.5.0-beta.1](https://gitcode.com/cann/asc-devkit/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-pto-isa   8.5.0-beta.1          | [v8.5.0-beta.1](https://gitcode.com/cann/pto-isa/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-ge-compiler   8.5.0-beta.1      | [v8.5.0-beta.1](https://gitcode.com/cann/ge/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-ge-executor   8.5.0-beta.1      | [v8.5.0-beta.1](https://gitcode.com/cann/ge/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-graph-autofusion   8.5.0-beta.1 | [v8.5.0-beta.1](https://gitcode.com/cann/graph-autofusion/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-metadef   8.5.0-beta.1          | [v8.5.0-beta.1](https://gitcode.com/cann/metadef/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-dflow-executor   8.5.0-beta.1   | [v8.5.0-beta.1](https://gitcode.com/cann/ge/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-hcomm   8.5.0-beta.1            | [v8.5.0-beta.1](https://gitcode.com/cann/hcomm/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| cann-npu-runtime   8.5.0-beta.1      | [v8.5.0-beta.1](https://gitcode.com/cann/runtime/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

### 🚀 关键特性

- 【工程能力】nn类onnx算子插件支持。([#452](https://gitcode.com/cann/ops-nn/pull/452))
- 【工程能力】增加编译选项oom、asan、mssanitizer、build-type等工程级稳定性与可调试性能力。([#391](https://gitcode.com/cann/ops-nn/pull/391))
- 【算子实现】部分算子新增对KirinX90支持。([#609](https://gitcode.com/cann/ops-nn/pull/609)、[#610](https://gitcode.com/cann/ops-nn/pull/610)、[#612](https://gitcode.com/cann/ops-nn/pull/612))
- 【算子实现】新支持[稀疏4:2量化matmul算子](matmul/sparse4to2quant_matmul)，针对稀疏矩阵使能硬件加速能力。([#429](https://gitcode.com/cann/ops-nn/pull/429))
- 【资料优化】增加QUICK_START，离线编译模式，aicore/aicpu/graph模式下开发指南完善。([#702](https://gitcode.com/cann/ops-nn/pull/702)、[#562](https://gitcode.com/cann/ops-nn/pull/562))
- 【资料优化】优化贡献指南中新算子贡献流程。([#294](https://gitcode.com/cann/ops-nn/pull/294))
- 【性能优化】增加asc_opc算子并行编译能力，优化编译效率；增加ccache，优化编译时长。([#692](https://gitcode.com/cann/ops-nn/pull/692))

### 🐛 问题修复

- 修复conv类算子编译告警问题。([Issue33](https://gitcode.com/cann/ops-nn/issues/33))
- 使用constexpr修饰if使能编译优化。([Issue98](https://gitcode.com/cann/ops-nn/issues/98))
- add_example样例算子执行调用问题修复。([Issue245](https://gitcode.com/cann/ops-nn/issues/245))
