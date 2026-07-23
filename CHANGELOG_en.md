# CHANGELOG

> This document records important changes in each version. Versions are arranged in reverse chronological order.

## v8.5.0-beta.1

Release Date: 2025-12-30

The first Beta version of ops-nn operator v8.5.0-beta.1 has been released.
This version introduces multiple new features, problem fixes, and performance improvements, and is currently in the testing stage.
We sincerely welcome community feedback to further improve the stability and functional completeness of ops-nn.
For usage, please refer to the [Official Documentation](https://gitcode.com/cann/ops-nn/blob/master/README.md).

### 🔗 Version Address

[CANN 8.5.0-beta 1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```text
The version directory description is as follows:
├── aarch64                 # CPU is ARM type
│   ├── ops                  # ops operator package directory, used to archive operator sub-packages
│   ├── ...
├── x86_64                   # CPU is X86 type
│   ├── ops                  # ops operator package directory, used to archive operator sub-packages
│   ├── ...
```

### 📌 Version Compatibility

**CANN Independent Upgrade Sub-package Version Compatibility Relationship**

| CANN Sub-package Version | Version Source Code Tag   | Compatible CANN Version|
|--|--|--|
| [cann-ops-math   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-math/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-nn   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-nn/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-cv   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-cv/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-transformer   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-transformer/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hccl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hccl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hixl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hixl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

**CANN Open Source Sub-package Version Compatibility Relationship**

| CANN Sub-package Version                         | Version Source Code Tag                                                 | Compatible CANN Version        |
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

- [Engineering Capability] nn class onnx operator plugin support. ([#452](https://gitcode.com/cann/ops-nn/pull/452))
- [Engineering Capability] Added compilation options oom, asan, mssanitizer, build-type, and other engineering-level stability and debuggability capabilities. ([#391](https://gitcode.com/cann/ops-nn/pull/391))
- [Operator Implementation] Some operators added support for KirinX90. ([#609](https://gitcode.com/cann/ops-nn/pull/609), [#610](https://gitcode.com/cann/ops-nn/pull/610), [#612](https://gitcode.com/cann/ops-nn/pull/612))
- [Operator Implementation] Newly supported [sparse 4:2 quantization matmul operator](matmul/sparse4to2quant_matmul), enabling hardware acceleration capabilities for sparse matrices. ([#429](https://gitcode.com/cann/ops-nn/pull/429))
- [Documentation Optimization] Added QUICK_START, offline compilation mode, aicore/aicpu/graph mode development guide improvement. ([#702](https://gitcode.com/cann/ops-nn/pull/702), [#562](https://gitcode.com/cann/ops-nn/pull/562))
- [Documentation Optimization] Optimized the new operator contribution process in the contribution guide. ([#294](https://gitcode.com/cann/ops-nn/pull/294))
- [Performance Optimization] Added asc_opc operator parallel compilation capability, optimized compilation efficiency; added ccache, optimized compilation duration. ([#692](https://gitcode.com/cann/ops-nn/pull/692))

### 🐛 Problem Fixes

- Fixed conv class operator compilation warning issues. ([Issue33](https://gitcode.com/cann/ops-nn/issues/33))
- Used constexpr to modify if to enable compilation optimization. ([Issue98](https://gitcode.com/cann/ops-nn/issues/98))
- add_example sample operator execution invocation problem fix. ([Issue245](https://gitcode.com/cann/ops-nn/issues/245))
