# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。


开发者贡献场景主要包括：

- 算子Bug修复

  如果您在本项目中发现了某些算子Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。

- 算子优化

  如果您对本项目中某些算子实现有泛化性增强/性能优化思路，希望着手实现这些优化点，欢迎您对算子进行优化贡献。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对优化点进行说明，并提供您的设计方案，
  然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行跟踪优化。

- 贡献新算子

  如果您有全新的算子想基于NPU进行设计实现，欢迎您在Issue中提出新的想法和设计。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue提供新增算子说明和设计方案，项目成员会与您进行沟通确认，并在`experimental`目录下为您的算子提供一个合适的`contrib`目录分类，您可以将新增算子贡献到对应目录下。

  同时，您需要在提交的Issue中评论“/assign”或“/assign @yourself”，认领该Issue并在后续完成新算子上库。

  新增算子的交付件通常比较多，您可以参考如下列表检查最小交付件集合，其中`${op_name}`表示新增算子名称。
  ```
  ${op_class}                                          # 算子分类
  ├── ${op_name}                                       # 算子名
  │   ├── op_host                                      # 算子信息库、Tiling、InferShape相关实现
  │   │   ├── ${op_name}_def.cpp                       # 算子信息库定义文件
  │   │   ├── ${op_name}_tiling.cpp                    # 算子Tiling实现文件
  │   │   └── CMakeLists.txt
  │   ├── op_kernel                                    # 算子Kernel目录
  │   │   ├── ${op_name}.cpp
  │   │   ├── ${op_name}.h
  │   │   ├── ${op_name}_tiling_data.h
  │   │   └── ${op_name}_tiling_key.h
  │   ├── CMakeLists.txt                               # 算子编译配置文件，保留原文件即可
  │   └── README.md                                    # 算子说明文档
  ```

- 文档纠错

  如果您在本项目中发现某些算子文档描述错误，欢迎您新建Issue进行反馈和修复。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类Issue指出对应文档的问题，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您纠正对应文档描述。

- 帮助解决他人Issue

  如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

  如果对应Issue需要进行代码修改，您可以在Issue评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您，跟踪协助解决问题。