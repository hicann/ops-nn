# ops-nn

## 🔥Latest News

- [2026/01] Added [QuickStart](QUICKSTART_en.md) to guide beginners through zero-based deployment of operator projects (supporting Docker environments), operator development, and contribution processes.
- [2025/12] Open-source operators support Ascend 950PR/Ascend 950DT/KirinX90, which can be developed and debugged through the [CANN Simulator](docs/en/debug/cann_simulator.md) simulation tool; optimized guideline documents, focusing on the [Operator Development Guide](docs/en/develop/aicore_develop_guide.md), clarifying minimum deliverables and key sample code, providing guidance for migrating operators from the Ascend/samples repository to this project; newly supported [sparse 4:2 quantization matmul operator](matmul/sparse4to2quant_matmul), enabling hardware acceleration capabilities for sparse matrices.
- [2025/11] Newly supported operators: [index_fill](index/index_fill/), [masked_scatter](index/masked_scatter/), [scatter](index/scatter/), [tf_scatter_add](index/tf_scatter_add/), [fused_cross_entropy_loss_with_max_sum](loss/fused_cross_entropy_loss_with_max_sum/).
- [2025/10] Added experimental directory, improved [Contribution Guide](CONTRIBUTING_en.md), supporting developers to debug and contribute custom operators.
- [2025/09] The ops-nn project was first released, with open-source operators supporting Atlas A2/A3 series products.

## 🚀Overview

ops-nn is a high-level operator library that provides neural network computing capabilities in the [CANN](https://hiascend.com/software/cann) (Compute Architecture for Neural Networks) operator library, including matmul, activation, and other types of operators. The operator library architecture is shown below:

<!--<img src="docs/en/figures/architecture.png" alt="Architecture Diagram"  width="700px" height="320px">-->

## 📝Version Compatibility

The source code of this project will be released along with the CANN software version. For the correspondence between CANN software versions and project tags, refer to the relevant version descriptions in the [release repository](https://gitcode.com/cann/release-management).
Note that to ensure smooth custom development of your source code, select the matching CANN version and Gitcode tag source code. Using the master branch may pose version mismatch risks.

## ⚡️Quick Start

If you want to **understand and quickly experience the project from scratch**, visit the following documents. You can first learn about the project operator information, then try operator invocation, development, contribution, and so on.

1. [Operator List](docs/zh/op_list.md): Complete operator information of the project for quick query.
2. [QuickStart](QUICKSTART_en.md): Provides a minimalist quick start guide **based on WebIDE or Docker environment**, including environment setup, compilation and deployment, operator invocation/development/debugging, contribution, and so on.

    > **Note**: Whether using WebIDE or Docker environment, the latest commercial release version of CANN software package is provided by default, which is currently CANN 8.5.0. If you want to manually install the CANN package or experience the latest capabilities of the master branch, refer to the steps in [Learning Tutorials](#learning-tutorials) to complete environment setup, compilation and execution, operator development, and other operations.

## 📖Learning Tutorials

If you have completed the **Quick Start** learning, have a certain understanding of this project, and want to **deeply understand and experience the project**, visit the following documents.

These documents provide diverse scenarios and more comprehensive operational guidance for you to apply to various AI business scenarios.

1. [Environment Deployment](docs/en/context/quick_install.md): Guide for setting up the **basic environment**, providing installation methods for third-party dependencies and software packages in various scenarios.
2. [Operator Invocation](docs/en/invocation/quick_op_invocation.md): Guide for operator **source code compilation and execution**, providing methods for operator package compilation (including online/offline scenarios) and operator running (including executing operator samples and UT) in different scenarios.
3. [Operator Development](docs/en/develop/aicore_develop_guide.md): Guide for **developing new operators** based on this project engineering, providing guidance for creating operator projects, implementing Tiling and Kernel core deliverables.
4. [Operator Debugging and Tuning](docs/en/debug/op_debug_prof.md): Provides common **operator debugging and tuning** methods, such as DumpTensor, msProf, Simulator, and so on.

In addition to the above guidelines, other documents are also provided, such as [Operator Invocation Methods](docs/en/invocation/op_invocation.md), terminology concepts, build parameter introduction, and so on. For complete documentation, visit [docs](docs/README_en.md).

## 🔍Directory Structure

The key directories are as follows. For detailed directory introduction, see [Project Directory](./docs/en/context/dir_structure.md).

```text
├── activation                         # activation class operators
├── cmake                              # project compilation directory
├── common                             # project common header files and common source code
├── control                            # control class operators
├── conv                               # conv class operators
├── docs                               # project documentation introduction
├── examples                           # end-to-end operator development and invocation examples
├── experimental                       # user-defined operator storage directory
├── foreach                            # foreach class operators
├── index                              # index class operators
├── loss                               # loss class operators
├── matmul                             # matmul class operators
│   ├── transpose_batch_mat_mul        # all deliverables of transpose_batch_mat_mul operator, such as Tiling, Kernel, and so on
│   │   ├── docs                       # operator documentation
│   │   ├── examples                   # operator usage examples
│   │   ├── op_graph                   # operator graph construction related directory
│   │   ├── op_host                    # operator information library, Tiling, InferShape related implementation directory
│   │   │   └── op_api                 # operator aclnn interface implementation directory
│   │   ├── op_kernel                  # operator Kernel directory
│   │   ├── CMakeLists.txt             # operator compilation configuration file
│   │   └── README.md                  # operator introduction document
│   ├── ...
│   └── CMakeLists.txt                 # operator compilation configuration file
├── ...
├── rnn                                # rnn class operators
├── scripts                            # script directory, containing custom operator and Kernel build related configuration files
├── tests                              # test project directory
├── vfusion                            # vfusion class operators
├── CMakeLists.txt
├── README.md
├── build.sh                           # project compilation script
├── install_deps.sh                    # dependency package installation script
└── requirements.txt                   # third-party dependency packages required by the project
```

## 💬Related Information

- [Contribution Guide](CONTRIBUTING_en.md)
- [Security Statement](SECURITY_en.md)
- [License](LICENSE)
- [Affiliated SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-nn)

## 🤝Contact Us

The functions and documentation of this project are being continuously updated and improved. We recommend that you follow the latest version.

- **Issue Feedback**: Submit issues through GitCode [Issues](https://gitcode.com/cann/ops-nn/issues).
- **Community Interaction**: Participate in discussions through GitCode [Discussions](https://gitcode.com/cann/ops-nn/discussions).
- **Technical Column**: Access technical articles through GitCode [Wiki](https://gitcode.com/cann/ops-nn/wiki), such as serialized tutorials and best practices.

  |Technical Topic|Sample|
  |----|----|
  |Operator Performance Optimization|[MatMul Operator Performance Optimization Practice and Effect Analysis](https://gitcode.com/cann/ops-nn/wiki/MatMul%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%E4%B8%8E%E6%95%88%E6%9E%9C%E5%88%86%E6%9E%90.md)|
  |Operator Performance Optimization|[MatMul Operator VCV Performance Optimization Practice and Effect Analysis](https://gitcode.com/cann/ops-nn/wiki/MatMul%E7%AE%97%E5%AD%90VCV%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%E4%B8%8E%E6%95%88%E6%9E%9C%E5%88%86%E6%9E%90.md)|
