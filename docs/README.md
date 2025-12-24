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

为方便开发者快速熟悉本项目，可按需获取对应文档，文档内容包括：

| 文档                                             | 说明                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| [算子列表](zh/op_list.md)                        | 介绍项目包含的所有算子。                                 |
| [aclnn列表](zh/op_api_list.md)                   | 介绍项目包含的所有aclnn前缀的算子API，通过该API可直调算子。  |
| [环境部署](zh/context/quick_install.md)          | 介绍项目的基础环境搭建，包括软件包和第三方依赖的获取和安装。 |
| [算子调用](zh/invocation/quick_op_invocation.md) | 介绍如何快速调用项目内算子，包括编译执行算子包和UT等。       |
| [算子开发](zh/develop/aicore_develop_guide.md)   | 介绍自定义算子的开发流程，包括算子原型定义、Tiling实现、Kernel实现等。 |
| [算子调用方式](zh/invocation/op_invocation.md)   | 介绍不同算子调用方式，包括aclnn调用、图模式调用等，方便算子快速应用于AI业务中。 |
| [算子调试调优](zh/debug/op_debug_prof.md)        | 介绍常见的算子调试、调优方法。                               |
| [图融合Pass列表](zh/op_pass_list.md)               | 介绍项目支持的所有图融合规则。                               |

## 附录
| 文档                                | 说明                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [算子基本概念](zh/context/基本概念.md) | 介绍算子相关的基础概念和特性，包括量化/稀疏等术语、数据类型、数据格式等。 |
| [build参数说明](zh/context/build.md)   | 介绍项目中build.sh脚本功能和具体参数。                       |