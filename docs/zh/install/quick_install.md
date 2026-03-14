# 环境部署

请您在学习QuickStart或各类教程操作之前，请先参考下面步骤完成基础环境搭建。

注意本文提到的编译态和运行态场景含义如下，请按需安装：

- 编译态：针对仅编译不运行本项目的场景，只需安装前置依赖和CANN toolkit包。
- 运行态：针对运行本项目的场景（编译运行或纯运行），除了安装前置依赖和CANN toolkit包，还需安装驱动与固件、CANN ops包。

## 前提条件

编译本项目前，请确保编译环境的基础库依赖、NPU驱动和固件已安装。

1. **安装依赖**

   - python >= 3.7.0（建议版本 <= 3.10）
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
   - dos2unix
   - gawk
   - patch
   - make

   上述依赖包请注意版本号，也可通过项目根目录install\_deps.sh一键安装，命令如下，若遇到不支持系统，请参考该文件自行适配。
   ```bash
   bash install_deps.sh
   ```

2. **安装驱动与固件**（运行态依赖）

   运行算子时必须安装驱动与固件，若仅编译算子，可跳过本操作。

   单击[下载链接](https://www.hiascend.com/hardware/firmware-drivers/community)，根据实际产品型号和环境架构，获取对应的`Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run`、`Ascend-hdk-<chip_type>-npu-firmware_<version>.run`包。

   安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》。

## 环境准备

本项目提供了多种CANN包（`Ascend-cann-toolkit`和`Ascend-cann-ops`）安装方式，请按需选择。

| CANN安装方式 | 说明 |使用场景|
| :--- | :--- | :--- |
| WebIDE | 一站式开发平台，提供了在线直接运行的昇腾环境。当前可提供单机算力，**默认安装最新商发版CANN软件包**（目前是CANN 8.5.0）。 | 适用于没有昇腾设备的开发者。|
| Docker | Docker镜像是一种高效部署方式，目前仅适用于Atlas A2系列产品，OS仅支持Ubuntu操作系统。**默认安装最新商发版CANN软件包**（目前是CANN 8.5.0） |适用有昇腾设备，需要快速搭建环境的开发者。|
| 手动安装 | - |适用有昇腾设备，想体验手动安装CANN包或体验最新master分支能力的开发者。|

### 方式1：WebIDE环境

对于无昇腾设备的开发者，可直接使用WebIDE开发平台，即“**算子一站式开发平台**”，该平台为您提供在线可直接运行的昇腾环境，环境中已安装必备的软件包，无需手动安装。更多关于开发平台的介绍请参考[LINK](https://gitcode.com/org/cann/discussions/54)。

1. 进入开源项目，单击“`云开发`”按钮，使用已认证过的华为云账号登录。若未注册或认证，请根据页面提示进行注册和认证。

   <img src="../figures/cloudIDE.png" alt="云平台"  width="750px" height="90px">

2. 根据页面提示创建并启动云开发环境，单击“`连接 > WebIDE `”进入算子一站式开发平台，开源项目的资源默认在`/mnt/workspace`目录下。

   <img src="../figures/webIDE.png" alt="云平台"  width="1000px" height="150px">


### 方式2：Docker部署

对于有昇腾设备的开发者，若您想快速搭建昇腾环境，可使用Docker镜像部署。



> **说明**：镜像文件比较大，下载需要一定时间，请您耐心等待。

1.**下载镜像**

- 步骤1：以root用户登录宿主机。确保宿主机已安装Docker引擎（版本1.11.2及以上）。
- 步骤2：从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)拉取已预集成CANN软件包及`ops-nn`所需依赖的镜像。命令如下，根据实际架构选择：

    ```bash
    # 示例：拉取ARM架构的CANN开发镜像
    docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
    # 示例：拉取X86架构的CANN开发镜像
    docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
    ```

2.**运行Docker**
拉取镜像后，需要以特定参数启动容器，以便容器内能访问宿主的昇腾设备。

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```
| 参数 | 说明 | 注意事项 |
| :--- | :--- | :--- |
| `--name cann_container` | 为容器指定名称，便于管理。 | 可自定义。 |
| `--device /dev/davinci0` | 核心：将宿主机的NPU设备卡映射到容器内，可指定映射多张NPU设备卡。 | 必须根据实际情况调整：`davinci0`对应系统中的第0张NPU卡。请先在宿主机执行 `npu-smi info`命令，根据输出显示的设备号（如`NPU 0`, `NPU 1`）来修改此编号。|
| `--device /dev/davinci_manager` | 映射NPU设备管理接口。 |  |
| `--device /dev/devmm_svm` | 映射设备内存管理接口。 |  |
| `--device /dev/hisi_hdc` | 映射主机与设备间的通信接口。 |  |
| `-v /usr/local/dcmi:/usr/local/dcmi` | 挂载设备容器管理接口（DCMI）相关工具和库。 | |
| `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | 挂载`npu-smi`工具。 | 使容器内可以直接运行此命令来查询NPU状态和性能信息。|
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | 关键挂载：将宿主机的NPU驱动库映射到容器内。 | |
| `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | 挂载驱动版本信息文件。 | |
| `-v /etc/ascend_install.info:/etc/ascend_install.info` | 挂载CANN软件安装信息文件。 | |
| `-it` | `-i`（交互式）和 `-t`（分配伪终端）的组合参数。 | |
| `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops` | 指定要运行的Docker镜像。 |请确保此镜像名和标签（tag）与你通过`docker pull`拉取的镜像完全一致。 |
| `bash` | 容器启动后立即执行的命令。 | |

### 方式3：手动安装CANN

对于有昇腾设备的开发者，若您想手动安装CANN包，请根据下述描述，选择对应的安装指导。

**场景1：已发布版本**

若您想体验**官网正式发布的CANN包**能力，请访问[CANN官网下载中心](https://www.hiascend.com/cann/download)，根据产品和环境架构选择对应版本的软件包（仅支持CANN 8.5.0及后续版本）进行安装。


**场景2：master版本**

若您想体验**master分支最新能力**，请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)，根据产品和环境架构选择对应版本的软件包，关键安装命令如下，更多安装指导参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》。

1. 安装CANN toolkit包。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
    ```
    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{install\_path\}：表示指定安装路径，默认安装在`/usr/local/Ascend`目录。

2. 安装CANN ops包（运行态依赖）。

 	ops包是运行态依赖，若仅编译算子，可以不安装此包。

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
    ```

    - \$\{cann\_version\}：表示CANN包版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。
    - \$\{soc\_name\}：表示NPU型号名称。
    - \$\{install\_path\}：表示指定安装路径，ops包需与toolkit包安装在相同路径，root用户默认安装在`/usr/local/Ascend`目录。

## 环境验证

安装完CANN包后，需验证环境和驱动是否正常。

-   **检查NPU设备**

    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```
-   **检查CANN安装**

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

通过如下命令下载项目源码，\$\{tag\_version\}请替换为版本分支标签名，源码版本与CANN版本配套关系参见[release仓库](https://gitcode.com/cann/release-management)。

```bash
# 下载项目对应分支源码
git clone -b ${tag_version} https://gitcode.com/cann/ops-nn.git
```

对于WebIDE或Docker环境，已默认提供最新商发版本的项目源码，如需获取其他版本的源码，也需通过上述命令下载源码。

对于手动安装CANN场景，安装CANN包和下载源码后，还需额外安装python基础库依赖。

```bash
# 安装根目录requirements.txt依赖
cd ops-nn
pip3 install -r requirements.txt
```
> [!NOTE] 注意
>
> - gitcode平台在使用HTTPS协议的时候要配置并使用个人访问令牌代替登录密码进行克隆，推送等操作。
> - 若您的编译环境无法访问网络，无法通过git指令下载代码，请先在联网环境中下载源码，再手动上传至目标环境。