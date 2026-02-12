# 算子调用
## 前提条件

- 环境部署：调用算子之前，请先参考[环境部署](../context/quick_install.md)完成基础环境搭建。
- 调用算子列表：项目可调用的算子参见[算子列表](../op_list.md)，算子对应的aclnn接口参见[aclnn列表](../op_api_list.md)。
- build.sh：算子调用依赖根目录build.sh脚本，可通过`bash build.sh --help`命令查看功能，参数介绍参考[build参数说明](../context/build.md)。

## 编译执行

基于社区版CANN包对算子源码修改时，可采用如下方式进行源码编译：

- [自定义算子包](#自定义算子包)：选择部分算子编译生成的包称为自定义算子包，以**挂载**形式作用于CANN包，不改变原始包内容。生成的自定义算子包优先级高于原始CANN包。该包支持aclnn和图模式调用AI Core、AI CPU算子。

- [ops-nn包](#ops-nn包)：选择整个项目编译生成的包称为ops-nn包，可**完整替换**CANN包对应部分。该包支持aclnn和图模式调用AI Core算子。

- [ops-nn静态库](#ops-nn静态库)：指整个项目编译为一个静态库文件，包含libcann_nn_static.a和aclnn接口头文件。该包仅支持aclnn调用AI Core算子。

  >说明：若您需要**基于本项目进行二次发布**并且对**软件包大小有要求**时，建议采用静态库编译，该库可以链接您的应用开发程序，仅保留业务所需的算子，从而实现软件最小化部署。

### 自定义算子包

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：

    > **说明**：编译过程依赖第三方开源软件，联网场景会自动下载，离线编译场景需要自行安装，具体参考[离线编译](../context/build_offline.md)。
    
    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    # 以TransposeBatchMatMul算子编译为例
    # bash build.sh --pkg --soc=ascend910b --vendor_name=transpose_batch_mat_mul --ops=transpose_batch_mat_mul
    # 编译experimental贡献目录下的用户算子
    # bash build.sh --pkg --experimental --soc=ascend910b --ops=${experimental_op}
    ```
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子。格式形如"transpose_batch_mat_mul,gemm,..."，多算子之间用英文逗号","分隔。
    - --experimental（可选）：表示编译experimental贡献目录下的算子，${experimental_op}为新贡献算子目录名，贡献说明参见[贡献指南](../../../CONTRIBUTING.md)。
    
    若\$\{vendor\_name\}和\$\{op\_list\}都不传入编译的是ops-nn包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。当提示如下信息，说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-nn-${vendor_name}-linux.${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。
    
2. **安装自定义算子包**
   
    ```bash
    ./cann-ops-nn-${vendor_name}-linux.${arch}.run
    ```
    
    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/cann。

3. **（可选）卸载自定义算子包**

    自定义算子包安装后在```${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_nn/scripts```目录会生成`uninstall.sh`脚本，通过执行该脚本可卸载自定义算子包，具体命令如下：
    ```bash
    bash ${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_nn/scripts/uninstall.sh
    ```

### ops-nn包

1. **编译ops-nn包**

    进入项目根目录，执行如下编译命令：

    > **说明**：编译过程依赖第三方开源软件，联网场景会自动下载，离线编译场景需要自行安装，具体参考[离线编译](../context/build_offline.md)。

    ```bash
    # 编译除experimental贡献目录外的所有算子
    bash build.sh --pkg [--jit] --soc=${soc_version}
    # 编译experimental贡献目录下的所有算子
    # bash build.sh --pkg --experimental [--jit] --soc=${soc_version}
    ```
    - --jit（可选）：设置后表示不编译算子二进制文件，如需使用aclnn调用算子，该选项无需设置。
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --experimental（可选）：表示编译experimental贡献目录下的算子。

    若提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-${soc_name}-ops-nn_${cann_version}_linux-${arch}.run" successfully created.
    ```

   \$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，run包存放于build_out目录下。

2. **安装ops-nn包**

    ```bash
    # 安装命令
    ./cann-${soc_name}-ops-nn_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
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
   bash build.sh --pkg --static --soc=${soc_version}
    ```
   \$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"。

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
    │   │   ├── libcann_nn_static.a                # 静态库文件
    │   └── include
    |       ├── ...                                 # aclnn接口头文件
    ```

## 本地验证 

通过项目根目录build.sh执行算子和UT用例。目前算子支持API方式（aclnn接口）和图模式调用，**推荐aclnn调用**。

### 执行算子样例

> **说明**：Ascend 950PR产品使用仿真执行算子样例，请见[仿真指导](../debug/op_debug_prof.md#方式二针对ascend-950pr)。

- 基于**自定义算子包**执行算子样例，包安装后，执行如下命令：
    ```bash
    bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--example_name=${example_name}] [--vendor_name=${vendor_name}] [--soc=${soc_version}]
    # 以TransposeBatchMatMul算子执行test_aclnn_transpose_batch_mat_mul.cpp为例
    # bash build.sh --run_example transpose_batch_mat_mul eager cust --example_name=transpose_batch_mat_mul --vendor_name=transpose_batch_mat_mul
    ```
    
    - \$\{op\}：表示待执行算子，算子名为小写下划线形式，如transpose_batch_mat_mul。
    - \$\{mode\}：表示执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
    - \$\{pkg_mode\}：表示包模式，目前仅支持cust，即自定义算子包。
    - \$\{example_name\}（可选）：表示待执行样例名，名称为各个算子examples文件夹下的文件名称，去掉`test_aclnn_`前缀和`.cpp`后缀。
    - \$\{vendor\_name\}（可选）：与构建的自定义算子包设置一致，默认名为custom。
    - \$\{soc_version\}（可选）：表示NPU型号。当设置为"ascend950"时会额外运行"arch35"目录下的示例文件。
    
    说明：\$\{mode\}为graph时，不指定\$\{pkg_mode\}和\$\{vendor\_name\}
    
- 基于**ops-nn包**执行算子样例，安装后，执行命令如下：
    ```bash
    bash build.sh --run_example ${op} ${mode} [--soc=${soc_version}]
    # 以TransposeBatchMatMul算子example执行为例
    # bash build.sh --run_example transpose_batch_mat_mul eager
    ```
    
    - \$\{op\}：表示待执行算子，算子名为小写下划线形式，如transpose_batch_mat_mul。
    - \$\{mode\}：表示算子执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
    - \$\{soc_version\}（可选）：表示NPU型号。当设置为"ascend950"时会额外运行"arch35"目录下的示例文件。
    
- 基于**ops-nn静态库**执行算子样例：

    1. **前提条件**
    
        ops-nn静态库依赖于ops-legacy静态库和ops-math静态库，将上述静态库准备好，解压并将所有lib64、include目录移动至统一目录\$\{static\_lib\_path\}下。
    
        > 说明：ops-legacy静态库`cann-${soc_name}-ops-legacy-static_${cann_version}_linux-${arch}.tar.gz`需单击[下载链接](https://mirror-centralrepo.devcloud.cn-north-4.huaweicloud.com/artifactory/cann-run-release/software/9.0.0/)获取， ops-nn静态库、ops-math静态库暂未提供软件包，请通过本地编译生成。
    
    2. **创建run.sh**
    
        在待执行算子`examples\test_aclnn_${op_name}.cpp`同级目录下创建run.sh文件。
    
        以TransposeBatchMatMul算子执行test_aclnn_transpose_batch_mat_mul.cpp为例，示例如下:
        
        ```bash
        # 环境变量生效
        if [ -n "$ASCEND_INSTALL_PATH" ]; then
            _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
        elif [ -n "$ASCEND_HOME_PATH" ]; then
            _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
        else
            _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
        fi
    
        source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    
        # 编译可执行文件
        g++ test_aclnn_transpose_batch_mat_mul.cpp -I ${static_lib_path}/include -L ${static_lib_path}/lib64 -L ${ASCEND_HOME_PATH}/lib64 -Wl,--allow-multiple-definition \
            -Wl,--start-group -lcann_nn_static -lcann_math_static -lcann_legacy_static -Wl,--end-group -lgraph -lgraph_base \
            -lpthread -lmmpa -lmetadef -lascendalog -lregister -lopp_registry -lops_base -lascendcl -ltiling_api -lplatform \
            -ldl -lc_sec -lnnopbase -lruntime -lerror_manager -lunified_dlog -o test_aclnn_transpose_batch_mat_mul   # 替换为实际算子可执行文件名
    
        # 执行程序
        ./test_aclnn_transpose_batch_mat_mul
        ```
    
        \$\{static\_lib\_path}表示静态库统一放置路径；\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/cann；最终可执行文件名请替换为实际算子可执行文件名。
        其中lcann\_nn\_static、lcann\_math\_static、lcann\_legacy\_static表示算子依赖的静态库文件，从静态库统一放置路径\$\{static\_lib\_path\}中获取；lgraph、lmetadef等表示算子依赖的底层库文件，可在CANN toolkit包获取。
        
    3. **执行run.sh**
    
        ```bash
        bash run.sh
        ```
    

无论上述哪种方式，算子样例执行后会打印结果，以TransposeBatchMatMul算子执行为例：

```
result[0] is: 0.000000
result[1] is: 0.000000
result[2] is: 0.000000
result[3] is: 0.000000
...
```
### 执行算子UT

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

假设验证ophost功能是否正常，执行如下命令：

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