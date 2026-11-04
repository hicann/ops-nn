# CHANGELOG

> This document records the important changes of each version, with versions sorted in reverse chronological order.

## v9.0.0-beta.2

Release date: 2026-03-30

This version delivers multiple new features, bug fixes, and performance improvements, and supports the latest 950 hardware.
We sincerely welcome community feedback to further improve the stability and functional completeness of ops-nn.
For usage, refer to the [official documentation](https://gitcode.com/cann/ops-nn/blob/master/README.md).

[CANN 9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/)

```text
The version directory structure is as follows:
├── aarch64                 # CPU is of the ARM type
│   ├── ops                  # ops operator package directory, used to archive operator subpackages
│   ├── ...
├── x86_64                   # CPU is of the X86 type
│   ├── ops                  # ops operator package directory, used to archive operator subpackages
│   ├── ...
```

### 📌 Version Compatibility

**Version compatibility of CANN independent upgrade subpackages**

| CANN subpackage version | Version source tag   | Compatible CANN version|
|--|--|--|
| [cann-ops-math   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/ops-math/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| [cann-ops-nn   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/ops-nn/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| [cann-ops-cv   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/ops-cv/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| [cann-ops-transformer   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/ops-transformer/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| [cann-hccl   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/hccl/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |
| [cann-hixl   9.0.0-beta.2](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0-beta.2/) | [v9.0.0-beta.2](https://gitcode.com/cann/hixl/tags/v9.0.0-beta.2) | CANN   9.0.0-beta.2 |

**Version compatibility of CANN open-source subpackages**

| CANN subpackage version                         | Version source tag                                                 | Compatible CANN version        |
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

### 🚀 Key Features

- [Engineering capability] Adaptation of the static library project  [!391](https://gitcode.com/cann/ops-nn/pull/391)
- [Engineering capability] Adaptation of the operator project to 950 [!450](https://gitcode.com/cann/ops-nn/pull/450)
- [Engineering capability] Added support for compiling the kernel of a specific operator by template [#1097](https://gitcode.com/cann/ops-nn/issues/1097)
- [New feature] 950 supports operator fusion optimization of the PyTorch view [#864](https://gitcode.com/cann/ops-nn/issues/864)
- [New feature] 950 supports the new homogeneous SIMD/SIMT programming ([#660](https://gitcode.com/cann/ops-nn/issues/660), [#710](https://gitcode.com/cann/ops-nn/issues/710), [#668](https://gitcode.com/cann/ops-nn/issues/668), [#656](https://gitcode.com/cann/ops-nn/issues/656), [#658](https://gitcode.com/cann/ops-nn/issues/658))

### 🐛 Bug Fixes

- Fixed the column-based core-splitting offset in deterministic computation of the ScatterNd operator ([!3137](https://gitcode.com/cann/ops-nn/pull/3137))
- Deterministic template of scatter_elements ([!3092](https://gitcode.com/cann/ops-nn/pull/3092))
- Accuracy of the single-core K-splitting template of the Matmul operator in the FP32 scenario ([!2651](https://gitcode.com/cann/ops-nn/pull/2651))

## v8.5.0-beta.1

Release date: 2025-12-30

The first Beta version v8.5.0-beta.1 of ops-nn operators is now released.
This version introduces multiple new features, bug fixes, and performance improvements, and it is still in the testing stage.
We sincerely welcome community feedback to further improve the stability and functional completeness of ops-nn.
For usage, refer to the [official documentation](https://gitcode.com/cann/ops-nn/blob/master/README.md).

### 🔗 Version Address

[CANN 8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```text
The version directory structure is as follows:
├── aarch64                 # CPU is of the ARM type
│   ├── ops                  # ops operator package directory, used to archive operator subpackages
│   ├── ...
├── x86_64                   # CPU is of the X86 type
│   ├── ops                  # ops operator package directory, used to archive operator subpackages
│   ├── ...
```

### 📌 Version Compatibility

**Version compatibility of CANN independent upgrade subpackages**

| CANN subpackage version | Version source tag   | Compatible CANN version|
|--|--|--|
| [cann-ops-math   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-math/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-nn   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-nn/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-cv   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-cv/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-transformer   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-transformer/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hccl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hccl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hixl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hixl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

**Version compatibility of CANN open-source subpackages**

| CANN subpackage version                         | Version source tag                                                 | Compatible CANN version        |
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

### 🚀 Key Features

- [Engineering capability] Support for nn-type onnx operator plugins. ([#452](https://gitcode.com/cann/ops-nn/pull/452))
- [Engineering capability] Added project-level stability and debuggability capabilities such as the compile options oom, asan, mssanitizer, and build-type. ([#391](https://gitcode.com/cann/ops-nn/pull/391))
- [Operator implementation] Some operators add support for KirinX90. ([#609](https://gitcode.com/cann/ops-nn/pull/609), [#610](https://gitcode.com/cann/ops-nn/pull/610), [#612](https://gitcode.com/cann/ops-nn/pull/612))
- [Operator implementation] Added support for the [sparse 4:2 quantization matmul operator](matmul/sparse4to2quant_matmul), enabling hardware acceleration for sparse matrices. ([#429](https://gitcode.com/cann/ops-nn/pull/429))
- [Documentation optimization] Added QUICK_START and offline compilation mode, and improved the development guides for the aicore/aicpu/graph modes. ([#702](https://gitcode.com/cann/ops-nn/pull/702), [#562](https://gitcode.com/cann/ops-nn/pull/562))
- [Documentation optimization] Optimized the new operator contribution process in the contribution guide. ([#294](https://gitcode.com/cann/ops-nn/pull/294))
- [Performance optimization] Added the asc_opc operator parallel compilation capability to improve compilation efficiency, and added ccache to shorten compilation time. ([#692](https://gitcode.com/cann/ops-nn/pull/692))

### 🐛 Bug Fixes

- Fixed the compilation warning issue of conv-type operators. ([Issue33](https://gitcode.com/cann/ops-nn/issues/33))
- Used constexpr-decorated if statements to enable compilation optimization. ([Issue98](https://gitcode.com/cann/ops-nn/issues/98))
- Fixed the execution invocation issue of the add_example sample operator. ([Issue245](https://gitcode.com/cann/ops-nn/issues/245))
