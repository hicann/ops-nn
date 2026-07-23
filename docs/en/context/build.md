# build Parameter Description

## Introduction

build.sh is the build script of this project, located in the project root directory by default. Its function is to automatically compile, link, and configure the source code, and finally generate executable files, library files, or other target files that can be installed or run directly. Specifically, the script configures different parameters to achieve multiple functions, including building multiple target libraries (such as libophost_nn.so), compiling operator packages, executing unit tests, etc.

## Usage

1. **Configure Environment Variables**

   Complete the basic environment setup by referring to [Environment Deployment](../context/quick_install.md).

   ```bash
   # Default path installation, taking root user as an example
   source /usr/local/Ascend/cann/set_env.sh
   ```

2. **Build Command Format**

   Taking the compile operator package command as an example, the format is as follows, where `--vendor_name` and `--ops` are optional in this scenario.

   ```bash
   bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
   ```

   For the meaning of all parameters, refer to the parameter description section below. Choose the appropriate parameters according to the actual situation.

## Parameter Description

build.sh supports multiple functions. You can view all function parameters through the following command.

```bash
bash build.sh --help
```

| Parameter Name              | Optional/Required  | Parameter Description                                                                        |
|------------------|--------|-----------------------------------------------------------------------------|
| -j${n}           | Optional     | Specifies the number of compilation threads. ${n} is the specific number of threads. The default value is 8 (such as -j8). If the number of threads exceeds the number of CPU cores, it will be automatically adjusted to the number of CPU cores.   |
| -v               | Optional     | View CMake compilation configuration information.                                                              |
| -O${n}           | Optional     | Specifies the compilation optimization level. Supports O0/O1/O2/O3 (such as -O3). ${n} is the optimization level identifier.                                |
| -u               | Optional     | Enables unit test (UT) compilation mode and compiles all UT targets.                                                    |
| --help, -h       | Optional     | Prints script usage help information.                                                               |
| --ops            | Optional     | Specifies the operators to be compiled, such as mat_mul_v3, mse_loss. Multiple operators are separated by English commas ",". Cannot be used with --ophost and --opapi at the same time. |
| --soc            | Optional     | Specifies the NPU model. Only 1 NPU model is supported per compilation.                                                   |
| --jit            | Optional     | In the static graph scenario, when compiling the `cann-${soc_name}-ops-nn_${cann_version}_linux-${arch}.run` package, you do not need to compile the operator binary files (the graph runtime will compile online). You can configure this option to improve compilation speed. |
| --static         | Optional     | When configured, it means generating a static library file, including libcann_nn_static.a and aclnn interface header files. Combined with the --pkg parameter, it generates a static library compressed package.|
| --vendor_name    | Optional     | Specifies the name of the custom operator package. The default value is custom.                                                   |
| --build-type     | Optional     | Enables debug mode. Optional types: Release/Debug. The default is Release. When the value is Debug, it cannot be used with --mssanitizer, --oom, --dump_cce at the same time         |
| --debug          | Optional     | Enables debug mode.                                                                     |
| --cov            | Optional     | Reserved parameter, developers do not need to pay attention for now.                                                              |
| --noexec         | Optional     | Only compiles the unit test binary file without automatically executing the compiled UT executable file.                                              |
| --opkernel       | Optional     | Compiles the binary kernel.                                                     |
| --pkg            | Optional     | Generates the installation package. Cannot be used with -u (UT mode) or --ophost, --opapi at the same time.                           |
| --asan           | Optional     | Enables host-side ASAN (AddressSanitizer) memory detection function.                                           |
| --valgrind       | Optional     | Reserved parameter, developers do not need to pay attention for now.                                                              |
| --make_clean     | Optional     | Executes basic cleanup operations (cleans compilation products). The script exits after execution.                                          |
| --make_clean_all | Optional     | Executes complete cleanup operations (deletes all compilation-related files). The script exits after execution.                                   |
| --ophost         | Optional     | Compiles the libophost_nn.so library. Cannot be used with --pkg, --ops at the same time.                                       |
| --opapi          | Optional     | Compiles the libopapi_nn.so library. Cannot be used with --pkg, --ops at the same time.                                        |
| --run_example    | Optional     | Compiles the sample of the specified operator and mode and executes the compiled executable file. Use --run_example --help to view the usage.     |
| --genop          | Optional     | Creates the AI Core custom operator initial directory.                                                           |
| --genop_aicpu    | Optional     | Creates the AI CPU custom operator initial directory.                                                            |
| --experimental   | Optional     | Compiles user operators in the experimental directory.                                                           |
| --mssanitizer    | Optional     | Enables kernel-side mssanitizer memory detection function.                                                  |
| --oom            | Optional     | Enables kernel-side oom memory detection function.                                                   |
| --dump_cce       | Optional     | Enables kernel-side dump precompiled file function.                                                      |
| --cann_3rd_lib_path| Optional   | The directory where third-party libraries are stored in the offline compilation scenario.                                                   |
| --simulator      | Optional     | Used in combination with --run_example to enable simulator mode to execute --run_example tasks. In simulator mode, the corresponding simulator library will be linked according to soc_version.          |
| --bisheng_flags  | Optional     | Specifies the BiSheng compiler compilation parameters. Multiple compilation parameters are separated by English commas ",". Cannot be used with --mssanitizer, --oom, --dump_cce at the same time.     |
| --kernel_template_input    | Optional     | Specifies the tilingKey template when compiling the kernel. Only one template can be specified. Used with --ops and only one operator can be specified. It will not compile the binary files of other operators that this operator depends on.     |
