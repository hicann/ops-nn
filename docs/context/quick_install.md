# 环境部署
## 前提条件

使用本项目前，请确保如下基础依赖、NPU驱动和固件已安装。

1. **安装依赖**

   本项目源码编译用到的依赖如下，请注意版本要求。

   - python >= 3.7.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
   - dos2unix
   - gawk
   - googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

   上述依赖包可通过项目根目录install\_deps.sh安装，命令如下，若遇到不支持系统，请参考该文件自行适配。
   ```bash
   bash install_deps.sh
   ```

2. **安装驱动与固件（运行态依赖）**

   运行算子时必须安装驱动与固件，若仅编译算子，可跳过本操作，安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。

## 软件包安装

1. **安装社区尝鲜版CANN toolkit包**

    根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包，下载链接为[toolkit x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ops/Ascend-cann-toolkit_8.5.0.alpha001_linux-x86_64.run)、[toolkit aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ops/Ascend-cann-toolkit_8.5.0.alpha001_linux-aarch64.run)。
    
    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --full --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

2. **安装社区版CANN legacy包（运行态依赖）**

    运行算子时必须安装本包，若仅编译算子，可跳过本操作。

    根据产品型号和环境架构，下载对应`cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run`包，下载链接如下：

    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：[legacy x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ops/cann-910b-ops-legacy_8.5.0.alpha001_linux-x86_64.run)、[legacy aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ops/cann-910b-ops-legacy_8.5.0.alpha001_linux-aarch64.run)。
    - Atlas A3 训练系列产品/Atlas A3 推理系列产品：[legacy x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ops/cann-910_93-ops-legacy_8.5.0.alpha001_linux-x86_64.run)、[legacy aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ops/cann-910_93-ops-legacy_8.5.0.alpha001_linux-aarch64.run)。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run
    # 安装命令
    ./cann-${soc_name}-ops-legacy_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```
    - \$\{soc\_name\}：表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。
    - \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

3. **安装社区版CANN ops-math包（可选）**

    如需本地运行项目算子，需额外安装此包，否则跳过本操作。

    根据产品型号和环境架构，下载对应`cann-${soc_name}-ops-math_${cann_version}_linux-${arch}.run`包，下载链接如下：

    - Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件：[ops-math x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ops/cann-910b-ops-math_8.5.0.alpha001_linux-x86_64.run)、[ops-math aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ops/cann-910b-ops-math_8.5.0.alpha001_linux-aarch64.run)。
    - Atlas A3 训练系列产品/Atlas A3 推理系列产品：[ops-math x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ops/cann-910_93-ops-math_8.5.0.alpha001_linux-x86_64.run)、[ops-math aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/community/ops/cann-910_93-ops-math_8.5.0.alpha001_linux-aarch64.run)。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x cann-${soc_name}-ops-math_${cann_version}_linux-${arch}.run
    # 安装命令
    ./cann-${soc_name}-ops-math_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```

    - \$\{soc\_name\}：表示NPU型号名称，即${soc_version}删除“ascend”后剩余的内容。
    - ${install_path}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

## 环境变量配置

请根据实际场景，选择合适的命令配置环境变量。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/set_env.sh
# 指定路径安装
# source ${install_path}/set_env.sh
```

## 源码下载

```bash
# 下载项目源码，以master分支为例
git clone https://gitcode.com/cann/ops-nn.git
# 安装根目录requirements.txt依赖
cd ops-nn
pip3 install -r requirements.txt
```