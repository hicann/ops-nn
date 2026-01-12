# ops-nn

## 🔥Latest News

- [2025/09] ops-nn项目首次上线。

## 🚀概述

ops-nn是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供神经网络计算能力的高阶算子库，包括matmul类、activation类等算子，算子库架构图如下：

<img src="docs/zh/figures/architecture.png" alt="架构图"  width="700px" height="320px">

## ⚡️快速入门

若您希望**从零到一了解并快速体验项目**，请访问如下文档。可以先了解项目算子信息，再尝试算子调用、开发、贡献等操作。

1. [算子列表](docs/zh/op_list.md)：项目全量算子信息，方便快速查询。

2. [QuickStart](QUICKSTART.md)：端到端快速上手指南，包括搭建环境、编译部署、算子调用/开发/调试调优、贡献等过程。

## 📖学习教程

若您**已完成快速入门**学习，对本项目有了一定认知，并希望**深入了解和体验项目**，请访问如下文档。

这些文档提供了多样化的场景介绍和更全面的操作指导，方便您应用于各种AI业务场景。

1. [环境部署](docs/zh/context/quick_install.md)：搭建基础环境的指南，提供了多种场景下第三方依赖和软件包安装方法等。
2. [算子调用](docs/zh/invocation/quick_op_invocation.md)：编译部署并调用算子的指南，提供了多种编译算子包以及运行算子的方法（包括执行算子样例和UT）等。
3. [算子开发](docs/zh/develop/aicore_develop_guide.md)：基于本项目工程开发新算子的指南，提供了创建算子工程、实现Tiling和Kernel核心交付件等指导。
4. [算子调试调优](docs/zh/debug/op_debug_prof.md)：提供了常见的算子调试和调优方法，如DumpTensor、msProf、Simulator等。

除了上述指南，还提供了其他文档介绍，例如算子调用方式和流程、算子基本概念等，全量文档介绍请访问[docs](docs/README.md)。

## 🔍目录结构
关键目录如下，详细目录介绍参见[项目目录](./docs/zh/context/dir_structure.md)。
```
├── activation                         # activation类算子
├── cmake                              # 项目工程编译目录
├── common                             # 项目公共头文件和公共源码
├── control                            # control类算子
├── conv                               # conv类算子
├── docs                               # 项目文档介绍
├── examples                           # 端到端算子开发和调用示例
├── experimental                       # 用户自定义算子存放目录
├── foreach                            # foreach类算子
├── index                              # index类算子
├── loss                               # loss类算子
├── matmul                             # matmul类算子
│   ├── transpose_batch_mat_mul        # transpose_batch_mat_mul算子所有交付件，如Tiling、Kernel等
│   │   ├── docs                       # 算子说明文档
│   │   ├── examples                   # 算子使用示例
│   │   ├── op_graph                   # 算子构图相关目录
│   │   ├── op_host                    # 算子信息库、Tiling、InferShape相关实现目录
│   │   │   └── op_api                 # 算子aclnn接口实现目录
│   │   ├── op_kernel                  # 算子Kernel目录
│   │   ├── CMakeLists.txt             # 算子编译配置文件
│   │   └── README.md                  # 算子介绍文档
│   ├── ...
│   └── CMakeLists.txt                 # 算子编译配置文件
├── ...
├── rnn                                # rnn类算子
├── scripts                            # 脚本目录，包含自定义算子、Kernel构建相关配置文件
├── tests                              # 测试工程目录
├── vfusion                            # vfusion类算子
├── CMakeLists.txt
├── README.md
├── build.sh                           # 项目工程编译脚本
├── install_deps.sh                    # 安装依赖包脚本
└── requirements.txt                   # 项目需要的第三方依赖包
```

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)