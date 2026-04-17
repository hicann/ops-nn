# 文档中心

## 目录结构

Docs目录结构说明如下：

```
├── zh
  ├── context                            # 公共文档，如术语、基础概念等
  ├── debug                              # 算子调试指导文档
  │   ├── op_debug_prof.md
  │   ├── ...
  ├── develop                            # 算子开发指导文档
  │   ├── aicore_develop_guide.md
  │   ├── aicpu_develop_guide.md
  │   ├── ...
  ├── figures                            # 图片目录
  ├── install                            # 环境安装和编译指导文档
  │   ├── build.md  
  │   ├── compile.md  
  │   ├── quick_install.md 
  │   └── ...
  ├── invocation                         # 算子调用指导文档（包括aclnn调用、图模式调用等）
  │   ├── quick_op_invocation.md
  │   ├── ...
  ├── op_api_list.md                     # 全量算子接口列表（aclnn）
  ├── op_list.md                         # 全量算子列表      
├── CONTRIBUTING_DOCS.md                 # 文档贡献说明
├── QUICKSTART.md                        # 快速入门
└── README.md                            
```

## 进阶教程
### 指南类文档

| 文档                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [源码构建指南](zh/install/compile.md)                        | 介绍联网/未联网场景下不同的源码构建方式和验证方法。          |
| [算子调用指南](zh/invocation/quick_op_invocation.md)         | 介绍调用算子样例方法和不同算子调用方式（如PyTorch/aclnn/图等）。 |
| [标准算子开发指南](zh/develop/aicore_develop_guide.md)       | 介绍如何基于标准工程定义算子原型、实现Tiling和Kernel，此类算子称为“标准算子”。<br>标准算子支持aclnn和图模式调用。 |
| [简易算子开发指南](../examples/fast_kernel_launch_example/README.md) | 介绍如何基于简易工程实现fast_kernel_launch，即`<<<>>>`方式，此类算子称为“简易算子”。<br>简易算子仅支持PyTorch调用。 |
| [算子调试调优](zh/debug/op_debug_prof.md)                    | 介绍常见的算子功能调试和性能调优方法（如数据采集和仿真流水等）。 |

### API类文档

| 文档        | 说明                  |
| ----------------------- | ---------------------- |
| [算子列表](zh/op_list.md)                        | 介绍项目包含的所有算子清单。                                 |
| [aclnn列表](zh/op_api_list.md)                   | 介绍项目包含的所有算子aclnn API清单。为方便用户在Host侧调用算子，提供了C语言API，即aclnn前缀的API。 |

### 工具类文档
| 文档        | 说明                  |
| ----------------------- | ---------------------- |
| [Simulator仿真工具](zh/debug/cann_sim.md) | 面向算子开发场景的SoC级仿真工具，用于分析运行在AI仿真器上AI任务在各阶段精度和性能数据。 |

### 更多文档

- 样例类文档：可参考[cann-samples](https://gitcode.com/cann/cann-samples)仓算子样例。
- 实践类文档：**建设中**，请您访问项目wiki进行搜索。
  |实践专题|样例|
  |----|----|
  |算子性能优化|[MatMul算子性能优化实践与效果分析](https://gitcode.com/cann/ops-nn/wiki/MatMul%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%E4%B8%8E%E6%95%88%E6%9E%9C%E5%88%86%E6%9E%90.md)|
  |算子性能优化|[MatMul算子VCV性能优化实践与效果分析](https://gitcode.com/cann/ops-nn/wiki/MatMul%E7%AE%97%E5%AD%90VCV%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%E4%B8%8E%E6%95%88%E6%9E%9C%E5%88%86%E6%9E%90.md)|

- FAQ类文档：**建设中**，请您访问项目wiki进行搜索。

## 附录

| 文档                                | 说明                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| [算子基本概念](zh/context/基本概念.md) | 介绍算子领域相关的基础概念和术语，如量化/稀疏、数据类型、数据格式等。 |
| [build参数说明](zh/install/build.md)   | 介绍本项目build.sh参数功能和取值，包括源码编译、算子调用、调试等。 |

