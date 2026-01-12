# 环境部署


基于本项目进行[算子调用](../invocation/quick_op_invocation.md)或[算子开发](../develop/aicore_develop_guide.md)之前，请您先参考下面步骤完成基础环境搭建。

环境搭建一般分为如下场景，您可以按需安装：

- 编译态：针对仅编译不运行本项目的场景，只需安装前置依赖和CANN toolkit包。
- 运行态：针对运行本项目的场景（编译运行或纯运行），除了安装前置依赖和CANN toolkit包，还需安装驱动与固件、CANN ops包。

## 前提条件

使用本项目前，请确保如下基础依赖、NPU驱动和固件已安装。

1. **安装依赖**

   本项目源码编译用到的依赖如下，请注意版本要求。

   - python >= 3.7.0（建议版本 <= 3.10） 
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
   - dos2unix
   - gawk
   - make
   - googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

   上述依赖包可通过项目根目录install\_deps.sh安装，命令如下，若遇到不支持系统，请参考该文件自行适配。
   ```bash
   bash install_deps.sh
   ```

2. **安装驱动与固件（运行态依赖）**

   运行算子时必须安装驱动与固件，若仅编译算子，可跳过本操作。
   
   单击[下载链接](https://www.hiascend.com/hardware/firmware-drivers/community)，根据实际产品型号和环境架构，获取对应的`Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run`、`Ascend-hdk-<chip_type>-npu-firmware_<version>.run`包。

   安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。

## 环境准备

### 1. 下载软件包

根据实际产品型号和环境架构，获取`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`、`Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run`。其中ops包是运行态依赖，若仅编译算子，可以不安装此包。
- Atlas A2/A3系列产品：单击[下载链接](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1)获取软件包。
- Ascend 950PR/Ascend 950DT产品：单击[下载链接](https://mirror-centralrepo.devcloud.cn-north-4.huaweicloud.com/artifactory/cann-run-release/software/9.0.0-alpha.1/)获取软件包。

### 2. 安装软件包
1. **安装社区CANN toolkit包**

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

2. **安装社区版CANN ops包（运行态依赖）**

    运行算子时必须安装本包，若仅编译算子，可跳过本操作。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
    ```
    
    - \$\{soc\_name\}：表示NPU型号名称。
    - \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

## 环境验证

安装完CANN包或进入Docker容器后，需验证环境和驱动是否正常。

-   **检查NPU设备**：
    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```
-   **检查CANN安装**：
    ```bash
    # 查看CANN Toolkit版本信息（默认路径安装）
    cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    ```

## 环境变量配置

按需选择合适的命令使环境变量生效。
```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

## 源码下载

```bash
# 下载项目源码，以master分支为例
git clone https://gitcode.com/cann/ops-nn.git
# 安装根目录requirements.txt依赖
cd ops-nn
pip3 install -r requirements.txt
```