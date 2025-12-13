# 算子调用
## 前提条件

- 环境部署：调用项目算子之前，请先参考[环境部署](../context/quick_install.md)完成基础环境搭建。
- 调用算子列表：项目可调用的算子参见[算子列表](../op_list.md)，算子对应的aclnn接口参见[aclnn列表](../op_api_list.md)。

## 编译执行

基于社区版CANN包对算子源码修改时，可采用如下方式进行源码编译：

- [自定义算子包](#自定义算子包)：选择部分算子编译生成的包称为自定义算子包，以**挂载**形式作用于CANN包，不改变原始包内容。生成的自定义算子包优先级高于原始CANN包。该包支持aclnn方式和图模式调用算子。

- [ops-nn包](#ops-nn包)：选择整个项目编译生成的包称为ops-nn包，可**完整替换**CANN包对应部分。该包支持aclnn方式和图模式调用算子。

### 自定义算子包

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    # 以TransposeBatchMatMul算子编译为例
    # bash build.sh --pkg --soc=ascend910b --vendor_name=transpose_batch_mat_mul --ops=transpose_batch_mat_mul
    # 编译experimental贡献目录下的算子
    # bash build.sh --pkg --experimental --soc=ascend910b --ops=${experimental_op}
    ```
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子。格式形如"transpose_batch_mat_mul,gemm,..."，多算子之间用英文逗号","分隔。
    - --experimental（可选）：表示编译experimental贡献目录下的算子，${experimental_op}为新贡献算子目录名，贡献说明参见[贡献指南](../../CONTRIBUTING.md)。

    说明：若\$\{vendor\_name\}和\$\{op\_list\}都不传入编译的是built-in包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。

    若提示如下信息，说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-nn-${vendor_name}-linux.${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。

2. **安装自定义算子包**

    ```bash
    ./cann-ops-nn-${vendor_name}-linux.${arch}.run
    ```

    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/latest。注意自定义算子包不支持卸载。

3. **（可选）卸载自定义算子包**

    自定义算子包安装后在`${ASCEND_HOME_PATH}/vendors/custom_nn/scripts`目录下会生成`uninstall.sh`脚本，通过执行该脚本可卸载自定义算子包，具体命令如下：
    ```bash
    bash ${ASCEND_HOME_PATH}/vendors/custom_nn/scripts/uninstall.sh
    ```

### ops-nn包

1. **编译ops-nn包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg [--jit] --soc=${soc_version}
    ```
    - --jit（可选）：设置后表示不编译算子二进制文件，如需使用aclnn调用算子，该选项无需设置。
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。

    若提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-${soc_name}-ops-nn_${cann_version}_linux-${arch}.run" successfully created.
    ```

   \$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，run包存放于build_out目录下。

2. **安装ops-nn包**

    ```bash
    ./cann-${soc_name}-ops-nn_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```

    \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

3. **（可选）卸载ops-nn包**

    ```bash
    # 卸载命令
    ./${install_path}/cann/share/info/ops_nn/script/uninstall.sh
    ```

## 本地验证

通过项目根目录build.sh脚本，可快速调用算子和UT用例，验证项目功能是否正常，build参数介绍参见[build参数说明](../context/build.md)。目前算子支持API方式（aclnn接口）和图模式调用，**推荐aclnn调用**。

- **执行算子样例**

    - 完成ops-nn包安装后，执行命令如下：
        ```bash
        bash build.sh --run_example ${op} ${mode}
        # 以TransposeBatchMatMul算子example执行为例
        # bash build.sh --run_example transpose_batch_mat_mul eager
        ```

        - \$\{op\}：表示待执行算子，算子名为小写下划线形式，如transpose_batch_mat_mul。
        - \$\{mode\}：表示算子执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。

    - 完成自定义算子包安装后，执行命令如下：
        ```bash
        bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--example_name=${example_name}] [--vendor_name=${vendor_name}]
        # 以TransposeBatchMatMul算子执行test_aclnn_transpose_batch_mat_mul.cpp为例
        # bash build.sh --run_example transpose_batch_mat_mul eager cust --example_name=transpose_batch_mat_mul --vendor_name=transpose_batch_mat_mul
        ```

        - \$\{op\}：表示待执行算子，算子名为小写下划线形式，如transpose_batch_mat_mul。
        - \$\{mode\}：表示执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
        - \$\{pkg_mode\}：表示包模式，目前仅支持cust，即自定义算子包。
        - \$\{example_name\}（可选）：表示待执行样例名，名称为各个算子examples文件夹下的文件名称，去掉`test_aclnn_`前缀和`.cpp`后缀。
        - \$\{vendor\_name\}（可选）：与构建的自定义算子包设置一致，默认名为custom。

        说明：\$\{mode\}为graph时，不指定\$\{pkg_mode\}和\$\{vendor\_name\}

    执行算子样例后会打印执行结果，以TransposeBatchMatMul算子为例，结果如下：

    ```
    result[0] is: 0.000000
    result[1] is: 0.000000
    result[2] is: 0.000000
    result[3] is: 0.000000
    ...
    ```
- **执行算子UT**

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