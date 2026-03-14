# 项目目录

> 本章罗列的部分目录是可选的，请以实际交付件为准。尤其**单算子目录**，不同场景下交付件有差异，具体说明如下：
>
> - 若缺少op_host目录，可能是调用了其他算子op_host实现，调用逻辑参见该算子op_api或op_graph目录下源码实现；也可能是Kernel暂无Ascend C实现，如有需要，欢迎开发者参考[贡献指南](../../../CONTRIBUTING.md)补充贡献该算子。
> - 若缺少op_kernel目录，可能是调用了其他算子op_kernel实现，调用逻辑参见该算子op_api或op_graph目录下源码实现；也可能是Kernel暂无Ascend C实现，如有需要，欢迎开发者参考[贡献指南](../../../CONTRIBUTING.md)补充贡献该算子。
> - 若缺少op_api目录，说明该算子暂不支持aclnn调用。
> - 若缺少op_graph目录，说明该算子暂不支持图模式调用。

项目全量目录层级介绍如下：
```
├── cmake                                               # 项目工程编译目录
│   ├── aclnn_ops_nn.h.in                               # aclnn汇总头文件模板
│   └── ...
├── common                                              # 项目公共头文件和公共代码
│   ├── CMakeLists.txt
│   ├── inc                                             # 公共头文件目录
│   └── src                                             # 公共代码目录
├── ${op_class}                                         # 算子分类，如activation、conv、matmul类算子
│   ├── ${op_name}                                      # 算子工程目录，${op_name}表示算子名（小写下划线形式）
│   │   ├── CMakeLists.txt                              # 算子CMakeList入口
│   │   ├── README.md                                   # 算子介绍文档
│   │   ├── docs                                        # 算子文档目录
│   │   │   └── aclnn${OpName}.md                       # 算子aclnn接口介绍文档，${OpName}表示算子名（大驼峰形式）
│   │   ├── examples                                    # 算子调用示例目录
│   │   │   ├── test_aclnn_${op_name}.cpp               # 算子通过aclnn调用的示例
│   │   │   └── test_geir_${op_name}.cpp                # 算子通过geir调用的示例
│   │   ├── op_graph                                    # 图融合相关实现
│   │   │   ├── CMakeLists.txt                          # op_graph侧CMakeList文件
│   │   │   ├── ${op_name}_graph_infer.cpp              # InferDataType文件，实现算子数据类型推导
│   │   │   ├── ${op_name}_proto.h                      # 算子原型定义，用于图优化和融合阶段识别算子
│   │   │   └── fusion_pass                             # 算子融合规则目录
│   │   ├── op_host                                     # Host侧实现
│   │   │   ├── CMakeLists.txt                          # Host侧CMakeList文件
│   │   │   ├── config                                  # 可选，二进制配置文件，若未配置工程自动生成
│   │   │   │   ├── ${soc_version}                      # 算子在NPU上配置的二进制信息，${soc_version}表示NPU型号
│   │   │   │   │   ├── ${op_name}_binary.json          # 算子二进制配置文件
│   │   │   │   │   └── ${op_name}_simplified_key.ini   # 算子SimplifiedKey配置信息
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_def.cpp                      # 算子信息库，定义算子基本信息，如名称、输入输出、数据类型等
│   │   │   ├── ${op_name}_infershape.cpp               # 可选，InferShape实现，根据算子形状推导输出shape，若未配置则输出shape与输入shape一样
│   │   │   ├── ${op_name}_tiling_${sub_case}.cpp       # 可选，针对某些子场景下的Tiling优化，${sub_case}表示子场景，如${op_name}_tiling_arch35是针对arch35架构的优化，若无该文件表明该算子没有对应子场景的特定Tiling策略
│   │   │   ├── ${op_name}_tiling_${sub_case}.h         # 可选，${sub_case}子场景下Tiling实现用的头文件
│   │   │   ├── ${op_name}_tiling.cpp                   # 可选，若无该文件表明对应场景下无Tiling实现(将张量划分为多个小块，区分数据类型进行并行计算)
│   │   │   ├── ${op_name}_tiling.h                     # 可选，Tiling实现用的头文件
│   │   │   └── op_api                                  # 可选，算子aclnn实现文件目录，若未配置工程自动生成
│   │   │       ├── aclnn_${op_name}.cpp                # 算子aclnn接口实现文件
│   │   │       ├── aclnn_${op_name}.h                  # 算子aclnn接口实现头文件
│   │   │       ├── ${op_name}.cpp                      # 算子l0接口实现文件
│   │   │       ├── ${op_name}.h                        # 算子l0接口实现头文件
│   │   │       └── CMakeLists.txt
│   │   │── op_kernel                                   # AI Core算子Device侧Kernel实现
│   │   │   ├── ${sub_case}                             # 可选，${sub_case}子场景使用的目录
│   │   │   │   ├── ${op_name}_${model}.h               # 算子kernel实现文件，${model}表示用户自定义文件名后缀，通常为Tiling模板名
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_tiling_key.h                 # 可选，TilingKey文件，定义Tiling策略的Key，标识不同划分方式，若未配置表明该算子无相应的Tiling策略
│   │   │   ├── ${op_name}_tiling_data.h                # 可选，TilingData文件，存储Tiling策略相关配置信息，如块大小、并行度，若未配置表明该算子无相应的Tiling策略
│   │   │   ├── ${op_name}.cpp                          # Kernel入口文件，包含主函数和调度逻辑
│   │   │   └── ${op_name}.h                            # Kernel实现文件，定义Kernel头文件，包含函数声明、结构定义、逻辑实现
│   │   │── op_kernel_aicpu                             # 可选，AI CPU算子Device侧Kernel实现
│   │   │   ├── ${op_name}_aicpu.cpp                    # Kernel入口文件，包含主函数和调度逻辑
│   │   │   └── ${op_name}_aicpu.h                      # Kernel头文件，包含函数声明、结构定义、逻辑实现
│   │   └── tests                                       # 算子测试用例目录
│   │       ├── CMakeLists.txt
│   │       └── ut                                      # 可选，UT测试用例，根据实际情况开发相应的用例
│   │           ├── CMakeLists.txt                      # UT用例CMakeList文件
│   │           ├── graph_plugin                        # graph_plugin测试用例目录
│   │           │   ├── CMakeLists.txt
│   │           │   └── fusion_pass                     # 融合规则测试用例目录
│   │           │       └── CMakeLists.txt
│   │           ├── op_host                             # op_host测试用例目录
│   │           │   ├── CMakeLists.txt
│   │           │   ├── ${op_name}_regbase_tiling.h
│   │           │   ├── op_api                          # op_api测试用例目录
│   │           │   │   ├── CMakeLists.txt
│   │           │   │   └── test_aclnn_${op_name}.cpp   # 算子aclnn测试用例文件
│   │           │   ├── test_${op_name}_${sub_case}.cpp # ${sub_case}子场景下op_host测试用例文件
│   │           │   ├── test_${op_name}.cpp             # op_host测试用例文件
│   │           │   ├── test_${op_name}_infershape.cpp  # 算子InferShape测试用例文件
│   │           │   └── test_${op_name}_tiling.cpp      # 算子Tiling测试用例文件
│   │           └── op_kernel                           # op_kernel测试用例目录
│   │               ├── CMakeLists.txt
│   │               │── test_${op_name}.cpp             # 算子Kernel测试用例文件
│   │               └── ${op_name}_data                 # 可选，op_kernel测试用例中依赖的数据比对和生成脚本，若未配置需在对应用例中手动实现
│   │                   ├── compare_data.py             # 数据脚本
│   │                   └── gen_data.py                 # 数据生成脚本
│   └── ...
├── docs                                                # 项目相关文档目录
├── examples                                            # 端到端算子开发和调用示例
│   ├── add_example                                     # AI Core算子示例目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件 
│   │   ├── examples                                    # 算子使用示例目录
│   │   ├── op_graph                                    # 算子构图相关目录
│   │   ├── op_host                                     # 算子信息库、Tiling、InferShape相关实现目录
│   │   ├── op_kernel                                   # 算子Kernel目录
│   │   └── tests                                       # 算子测试用例目录
│   ├── add_example_aicpu                               # AI CPU算子示例目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件
│   │   ├── examples                                    # 算子使用示例目录
│   │   ├── op_graph                                    # 算子构图相关目录
│   │   ├── op_host                                     # 算子信息库、InferShape相关实现
│   │   ├── op_kernel_aicpu                             # 算子Kernel目录
│   │   └── tests                                       # 算子测试用例目录
│   ├── fast_kernel_launch_example                      # 轻量级，高性能的算子开发工程模板
│   │   ├── ascend_ops                                  # 示例算子实现目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件
│   │   ├── README.md                                   # 轻量级，高性能的算子开发工程说明资料
│   │   ├── requirements.txt
│   │   └── setup.py                                    # 构建脚本
│   ├── CMakeLists.txt
│   └── README.md                                       # 项目示例介绍文档
├── experimental                                        # 用户自定义算子存放目录
├── scripts                                             # 脚本目录，包含自定义算子、Kernel构建相关配置文件
├── tests                                               # 项目级测试目录
│   ├── requirements.txt                                # 测试用例依赖的第三方组件
│   └── ut                                              # UT用例工程
│       ├── CMakeLists.txt                              # UT工程的CMakeList脚本
│       ├── common                                      # UT工程中使用的公共代码
│       ├── op_api                                      # op_api测试工程
│       ├── op_host                                     # op_host测试工程
│       └── op_kernel                                   # op_kernel测试工程
├── CMakeLists.txt                                      # 项目工程CMakeList入口
├── CONTRIBUTING.md                                     # 项目贡献指南文件
├── LICENSE                                             # 项目开源许可证信息
├── OAT.xml                                             # 配置脚本，代码仓工具使用，用于检查License是否规范
├── README.md                                           # 项目工程总介绍文档
├── SECURITY.md                                         # 项目安全声明文件
├── build.sh                                            # 项目工程编译脚本
├── install_deps.sh                                     # 项目安装依赖包脚本
├── requirements.txt                                    # 项目的第三方依赖包
└── version.info                                        # 项目版本信息
```
