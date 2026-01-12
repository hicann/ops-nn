# 项目文档

## 目录说明

关键目录结构如下：

```
├── context                            # 公共目录，存放包括基本概念、项目目录介绍、build参数说明等文档
│   ├── dir_structure.md    
│   ├── build.md               
│   └── ...
├── debug                              # 算子调试调优文档目录
│   ├── op_debug_prof.md
│   └── ...
├── develop                            # 算子开发文档目录(包括AI Core和AI CPU算子开发指南等)
│   ├── aicore_develop_guide.md
│   ├── aicpu_develop_guide.md
│   └── ...
├── figures                            # 图片目录
├── invocation                         # 算子调用文档目录（包括aclnn调用、图模式调用等）
│   ├──op_invocation.md
│   └── ...
├── op_api_list.md                     # 全量算子接口列表（aclnn）
├── op_list.md                         # 全量算子列表
└── README
```

## 文档说明

项目全量文档如下，请按需获取对应内容。

| 文档                                             | 说明                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| [算子列表](zh/op_list.md)                        | 介绍项目包含的所有算子清单。                                 |
| [aclnn列表](zh/op_api_list.md)                   | 介绍项目包含的所有算子API，通过该API可直调算子。             |
| [环境部署](zh/context/quick_install.md)          | 介绍基础环境搭建过程，包括不同场景下软件包和第三方依赖的获取和安装。 |
| [算子调用](zh/invocation/quick_op_invocation.md) | 介绍如何调用项目算子，包括不同场景下编译和执行算子包、UT等。 |
| [算子开发](zh/develop/aicore_develop_guide.md)   | 介绍如何基于本项目工程开发新算子，包括算子原型定义、Tiling实现、Kernel实现等。 |
| [算子调用方式](zh/invocation/op_invocation.md)   | 介绍多种算子调用方式和调用流程，例如aclnn调用、图模式调用等。 |
| [算子调试调优](zh/debug/op_debug_prof.md)        | 介绍常见的算子调试、调优方法。                               |

## 附录

| 文档                                | 说明                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [算子基本概念](zh/context/基本概念.md) | 介绍算子领域相关的基础概念和术语，如量化/稀疏、数据类型、数据格式等。 |
| [build参数说明](zh/context/build.md)   | 介绍本项目build.sh脚本的功能和参数含义。                 |