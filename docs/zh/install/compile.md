# 源码构建

源码构建前，请参考本项目README完成环境准备和源码下载，此处不再赘述。

## 安装构建工具

> **说明**：对于WebIDE或Docker环境，默认已安装源码编译所需的构建工具，无需手动安装，可跳过本章操作。

请先确保编译环境的基础库依赖已安装，注意满足版本号要求。若您想体验使用Ascend C和PyTorch Extension能力开发自定义NPU算子，请参考[指南](../develop/torch_extension_develop_guide.md)。

- python >= 3.7.0
- gcc >= 7.3.0
- cmake >= 3.16.0
- pigz（可选，安装后可提升打包速度，建议版本 >= 2.4）
- dos2unix
- gawk
- make
- patch
- googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

上述依赖可通过项目根目录install\_deps.sh一键安装，命令如下，若遇到不支持系统，请参考该文件自行适配。

```bash
bash install_deps.sh
```

安装完上述依赖后，可通过项目根目录requirements.txt继续安装python三方库依赖，命令如下。

```bash
pip3 install -r requirements.txt
```

## 安装第三方依赖

> **说明**：对于WebIDE或Docker环境，默认联网，编译过程中会自动下载第三方依赖，无需手动安装，可跳过本章操作。

本项目编译过程依赖的第三方开源软件列表如下：

| 开源软件 | 版本 | 下载地址 |
|---|---|---|
| json | 3.11.3 | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| makeself | 2.5.0 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
| eigen | 5.0.0 | [eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz) |
| protobuf | 25.1.0 | [protobuf-25.1.tar.gz](https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz) |
| abseil-cpp | 20230802.1 | [abseil-cpp-20230802.1.tar.gz](https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz) |
| opbase(自CANN 9.0.0及以后版本需要下载) | master | [opbase](https://gitcode.com/cann/opbase) |

若您的编译环境可以访问网络，请参考[联网编译](#联网编译)，编译脚本会自动联网下载第三方软件。否则，请参考[未联网编译](#未联网编译)手动下载第三方软件。

准备好开源第三方软件后，可采用如下编译方式，请按需选择：

- **自定义算子包**：

  选择部分算子编译生成的包称为自定义算子包，以**挂载**形式作用于CANN包，不改变原始包内容。生成的自定义算子包优先级高于原始CANN包。该包支持aclnn和图模式调用AI Core、AI CPU算子。

- **ops-nn包**：

  选择整个项目编译生成的包称为ops-nn包，可**完整替换**CANN包对应部分。该包支持aclnn和图模式调用AI Core算子。

- **ops-nn静态库**：

  > 说明：若您需要**基于本项目进行二次发布**并且对**软件包大小有要求**时，建议采用静态库编译，该库可以链接您的应用开发程序，仅保留业务所需的算子，从而实现软件最小化部署。

  表示整个项目编译为一个静态库文件，包含libcann_nn_static.a和aclnn接口头文件。该包仅支持aclnn调用AI Core算子。

## 联网编译

若在有互联网的环境下编译，编译过程中会自动安装第三方依赖，无需手动安装。不同场景下源码编译和部署命令如下：

### 自定义算子包

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}] [-j${n}]
    # 以TransposeBatchMatMul算子编译为例
    # bash build.sh --pkg --soc=ascend910b --ops=transpose_batch_mat_mul -j16
    # 编译experimental贡献目录下的用户算子（以TransposeBatchMatMul算子为例，编译时请以实际贡献算子为准）
    # bash build.sh --pkg --experimental --soc=ascend910b --ops=transpose_batch_mat_mul -j16
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas A2 推理系列产品使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子。格式形如"transpose_batch_mat_mul,fatrelu_mul,..."，多算子之间用英文逗号","分隔。
    - --experimental（可选）：表示编译用户保存在experimental贡献目录下的算子。
    - -j（可选）：指定编译线程数，加快编译速度。

    若\$\{vendor\_name\}和\$\{op\_list\}都不传入编译的是ops-nn包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。当提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-ops-nn-${vendor_name}_linux-${arch}.run" successfully created.
    ```

    编译成功后，run包存放于项目根目录的build_out目录下。

2. **安装自定义算子包**

    ```bash
    ./build_out/cann-ops-nn-${vendor_name}_linux-${arch}.run
    ```

    自定义算子包默认安装路径为```${ASCEND_HOME_PATH}/opp/vendors```，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/cann。

    > 说明：可通过配置--install-path=\$\{install\_path\}参数定义算子包安装目录，在使用自定义算子包前，需执行source \$\{install_path\}/vendors/${vendor_name}/bin/set_env.bash命令，set\_env.bash脚本将自定义算子包安装路径追加到环境变量ASCEND\_CUSTOM\_OPP\_PATH中，使自定义算子包在当前环境中生效。

3. **（可选）卸载自定义算子包。**

    自定义算子包安装后在```${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_nn/scripts```目录会生成`uninstall.sh`，通过该脚本可卸载自定义算子包，命令如下：

    ```bash
    bash ${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_nn/scripts/uninstall.sh
    ```

### ops-nn包

1. **编译ops-nn包**

    进入项目根目录，执行如下编译命令：

    ```bash
    # 编译除experimental目录外的所有算子
    bash build.sh --pkg --soc=${soc_version} [-j${n}]
    # 编译experimental目录下的所有算子
    # bash build.sh --pkg --experimental --soc=${soc_version} [-j${n}]
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas A2 推理系列产品使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --experimental（可选）：表示编译用户保存在experimental目录下的算子。
    - -j（可选）：指定编译线程数，加快编译速度。

    若提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-${soc_name}-ops-nn_${cann_version}_linux-${arch}.run" successfully created.
    ```
    
    \$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，run包存放于build_out目录下。

2. **安装ops-nn包**

    ```bash
    # 安装命令
    ./build_out/cann-${soc_name}-ops-nn_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```

    \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

3. **（可选）卸载ops-nn包**

    ```bash
    # 卸载命令
    ./${install_path}/cann/share/info/ops_nn/script/uninstall.sh
    ```

### ops-nn静态库

> 说明：静态库仅支持Atlas A2、Atlas A3系列产品。experimental算子暂不支持使用静态库。

1. **编译ops-nn静态库**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --static --soc=${soc_version} [-j${n}]
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"。
    - -j（可选）：指定编译线程数，加快编译速度。
    若提示如下信息，说明编译并压缩成功。

    ```bash
    [SUCCESS] Build static lib success!
    Successfully created compressed package: ${repo_path}/build_out/cann-${soc_name}-ops-nn-static_${cann_version}_linux-${arch}.tar.gz
    ```

    \$\{repo\_path\}表示项目根目录，\$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，压缩包存放于build_out目录下。

2. **解压ops-nn静态库**

    进入build_out目录执行解压命令：

    ```bash
    tar -zxvf ./cann-${soc_name}-ops-nn-static_${cann_version}_linux-${arch}.tar.gz -C ${static_lib_path}
    ```

    \$\{static\_lib\_path\}表示静态库解压路径。解压后目录结构如下：

    ```
    ├── cann-${soc_name}-ops-nn-static_${cann_version}_linux-${arch}
    │   ├── lib64
    │   │   ├── libcann_nn_static.a               # 静态库文件
    │   └── include
    |       ├── ...                                 # aclnn接口头文件
    ```

## 未联网编译

若在没有连接互联网的环境下编译，需要提前准备好依赖的第三方软件，再进行源码编译。具体过程如下：

2. **下载第三方依赖**

    在联网环境中提前下载第三方软件，目前有如下方式，请按需选择：

    - 方式1：根据[第三方软件依赖](#第三方软件依赖)提供的表格手动下载，若从其他地址下载，请确保版本号一致。
    
    - 方式2：通过[third_lib_download.py](../../../scripts/tools/third_lib_download.py)脚本一键下载，该脚本在本项目`scripts/tools/`目录，下载该脚本并执行如下命令：
    
        ```bash
        python ${scripts_dir}/third_lib_download.py
        ```

    \$\{scripts\_dir\}表示脚本存放路径，下载的第三方软件包默认存放在当前脚本所在目录。

3. **编译算子包**

    将下载好的第三方软件上传至离线环境，可存放在`third_party`目录或自定义目录下。**推荐前者，其编译命令与联网编译场景下的命令一致。**

    - **third\_party目录**（推荐）
    
        请在本项目根目录创建`third_party`目录（若有则无需创建），将第三方软件拷贝到该指定目录。此时编译命令与联网编译命令一致，具体参考[联网编译](#联网编译)。
    
    - **自定义目录**
    
        在离线环境的任意位置新建`${cann_3rd_lib_path}`目录，将第三方软件拷贝到该目录，请确保该目录有权限访问。

        ```bash
        mkdir -p ${cann_3rd_lib_path}
        ```
        
        此时编译命令需在联网编译命令基础上额外增加`--cann_3rd_lib_path=${cann_3rd_lib_path}`用于指定第三方软件所在路径。假设存放路径为`/path/cann_3rd_lib_path`，不同编译方式对应的命令如下：
        
        - 自定义算子包
        
            ```bash
            bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}] --cann_3rd_lib_path=${cann_3rd_lib_path}
            # 以TransposeBatchMatMul算子编译为例
            # bash build.sh --pkg --soc=ascend910b --ops=transpose_batch_mat_mul -j16 --cann_3rd_lib_path=/path/cann_3rd_lib_path
            ```
            
        - ops-nn整包
        
            ```bash
            bash build.sh --pkg --soc=${soc_version} --cann_3rd_lib_path=${cann_3rd_lib_path}
            # bash build.sh --pkg --soc=ascend910b --cann_3rd_lib_path=/path/cann_3rd_lib_path
            ```
            
        - ops-nn静态库
        
            ```bash
            bash build.sh --pkg --static --soc=${soc_version} --cann_3rd_lib_path=${cann_3rd_lib_path}
            # bash build.sh --pkg --static --soc=ascend910b --cann_3rd_lib_path=/path/cann_3rd_lib_path
            ```

4. **安装/卸载算子包**

    未联网和联网场景下编译得到算子包结果一样，默认存放于项目根目录build_out目录下，并且安装和卸载的操作命令也一样，具体参见[联网编译](#联网编译)。

## 本地验证 

源码包部署后，可通过项目根目录build.sh执行UT用例，验证项目功能是否正常。

> 说明：执行UT用例依赖googletest单元测试框架，详细介绍参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

```bash
# 安装根目录下test相关requirements.txt依赖
pip3 install -r tests/requirements.txt
# 方式1: 编译并执行指定算子和对应功能的UT测试用例（选其一）
bash build.sh -u --[opapi|ophost|opkernel] --ops=transpose_batch_mat_mul
# 方式2: 编译并执行所有的UT测试用例
# bash build.sh -u
# 方式3: 编译所有的UT测试用例但不执行
# bash build.sh -u --noexec
# 方式4: 编译并执行对应功能的UT测试用例（选其一）
# bash build.sh -u --[opapi|ophost|opkernel]
# 方式5: 编译对应功能的UT测试用例但不执行（选其一）
# bash build.sh -u --noexec --[opapi|ophost|opkernel]
# 方式6: 执行UT测试用例时可指定soc编译
# bash build.sh -u --[opapi|ophost|opkernel] [--soc=${soc_version}]
```

以验证ophost功能是否正常为例，执行如下命令：

```bash
bash build.sh -u --ophost
```

执行完成后出现如下内容，表示执行成功。

```bash
Global Environment TearDown
[==========] ${n} tests from ${m} test suites ran. (${x} ms total)
[  PASSED  ] ${n} tests.
[100%] Built target nn_op_host_ut
```

\$\{n\}表示执行了n个用例，\$\{m\}表示m项测试，\$\{x\}表示执行用例消耗的时间，单位为毫秒。
