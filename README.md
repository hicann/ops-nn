# ops-nn

## 🔥Latest News

- [2026/01] 新增[QuickStart](docs/QUICKSTART.md)，指导新手零基础入门算子项目部署（支持Docker环境）、算子开发和贡献流程。
- [2025/12] 开源算子支持Ascend 950PR/Ascend 950DT/KirinX90，可以通过[CANN Simulator](./docs/zh/debug/cann_sim.md)仿真工具开发调试；优化指南类文档，聚焦[算子开发指南](docs/zh/develop/aicore_develop_guide.md)，明确最小交付件和关键示例代码，针对Ascend/samples仓算子提供迁移本项目的指导；新支持[稀疏4:2量化matmul算子](matmul/sparse4to2quant_matmul)，针对稀疏矩阵使能硬件加速能力。
- [2025/11] 新支持算子[index_fill](../index/index_fill/)、[masked_scatter](index/masked_scatter/)、[scatter](index/scatter/)、[tf_scatter_add](index/tf_scatter_add/)、[fused_cross_entropy_loss_with_max_sum](loss/fused_cross_entropy_loss_with_max_sum/)。
- [2025/10] 新增experimental目录，完善[贡献指南](CONTRIBUTING.md)，支持开发者调试并贡献自定义算子。
- [2025/09] ops-nn项目首次上线，开源算子支持Atlas A2/A3系列产品。

## 🚀概述

ops-nn是[CANN](https://hiascend.com/software/cann)（Compute Architecture for Neural Networks）算子库中提供神经网络计算能力的高阶算子库，包括matmul类、activation类等算子，算子库架构图如下：

<img src="docs/zh/figures/architecture.png" alt="架构图"  width="700px" height="320px">

## 📌版本配套

本项目源码会跟随CANN软件版本发布，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明。
请注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与Gitcode标签源码，使用master分支可能存在版本不匹配的风险。

## 🛠️环境准备

[环境部署](docs/zh/install/quick_install.md)是体验本项目能力的前提，请先完成NPU驱动、CANN包安装等，确保环境正常。

## ⬇️源码下载

环境准备好后，下载与CANN版本配套的分支源码，命令如下，\$\{tag\_version\}替换为分支标签名。
 	 
> 说明：若环境中已存在配套分支源码，**可跳过本步骤**，例如WebIDE默认已提供最新商发版CANN对应的源码。

```bash
git clone -b ${tag_version} https://gitcode.com/cann/ops-nn.git
```

说明：对于WebIDE环境，已默认提供最新商发CANN版本配套的源码，如需获取其他版本源码，参考上述命令获取。

## 📖学习教程

- [快速入门](docs/QUICKSTART.md)：从零开始快速体验项目核心基础能力，涵盖源码编译、算子调用、开发与调试等操作。
- [进阶教程](docs/README.md)：如需深入了解项目编译部署、算子调用、开发、调试调优等能力，请查阅文档中心获取详细指引。

## 💬相关信息

- [目录结构](docs/zh/install/dir_structure.md)
- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-basic)

-----
PS：本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-nn/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-nn/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-nn/wiki)获取技术文章，如系列化教程、优秀实践等。
