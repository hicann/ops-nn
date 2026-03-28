# MatmulFp32算子

##  概述
本样例的MatmulFP32算子实现使用自定义算子工程，其kernel包含两个模板：基础模板以及AL1全载模板。本工程默认使用基础模板，若要使用AL1全载模板，则需要在tiling阶段开启AL1全载的开关且同时满足AL1全载的tiling的条件。example的用例分别使用两个模板的精度均正常，示例通过msprof工具采集了模板的性能数据。
##  支持的AI处理器
| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 目录结构介绍
```
├── CMakeLists.txt                        // 编译工程文件
├── README.md                             // 算子样例工程介绍文件
├── examples                              // 样例算子的aclnn接口样例目录
│   ├── README.md                         // 样例算子aclnn接口介绍文档
│   ├── inc                               // 头文件目录
│   │   ├── common.h                      // 声明公共方案类，用于读取二进制文件
│   │   └── operator_desc.h               // 算子描述声明文件，包含算子输入/输出，算子类型以及输入描述与输出描述
│   │   ├── op_runner.h                   // 算子运行相关信息声明文件，包含算子输入/输出个数，输入/输出大小等
│   ├── input                             // 存放脚本生成的输入数据目录
│   ├── output                            // 存放算子运行处处数据和真值数据的目录
│   ├── run.sh                            // 编译 + 执行aclnn接口的脚本文件
│   ├── scripts                
│   │   ├── gen_data.py                   // 输入数据和真值数据生成脚本文件
│   │   └── verify_result.py              // 真值对比文件
│   └── src
│       ├── CMakeLists.txt                // 编译工程文件
│       ├── common.cpp                    // 公共方法类的实现，用户读取二进制文件
│       ├── main.cpp                      // 单算子调用应用的入口
│       ├── op_runner.cpp                 // 算子运行相关信息实现，包含算子输入/输出个数，输入/输出大小等
│       └── operator_desc.cpp             // 算子描述实现，包含算子输入/输出，算子类型以及输入描述与输出描述
├── op_host
│   ├── CMakeLists.txt                    // 编译工程文件
│   ├── matmul_fp32_def.cpp               // 算子信息库，定义算子基本信息，如名称、输入输出、数据类型等
│   ├── matmul_fp32_infershape.cpp        // 算子形状推导，在运行时推导输出shape
│   └── matmul_fp32_tiling.cpp            // Tiling实现，将张量划分为多个小块，区分数据类型进行并行计算
├── op_kernel
│   ├── matmul_fp32.cpp                   // Kernel入口文件，包含主函数和调度逻辑
│   ├── matmul_fp32_al1_fullload_kernel.h  // AL1全载模板Kernel实现文件
│   ├── matmul_fp32_base_block.h           // 基本模板block层计算文件
│   ├── matmul_fp32_base_kernel.h          // 基础模板Kernel实现文件
│   ├── matmul_fp32_common.h               // Kernel模板公共头文件
│   ├── matmul_fp32_tiling_data.h          // Tilingdata文件，存储Tiling策略相关的配置数据，如块大小等
│   └── matmul_fp32_tiling_key.h           // Tilingkey文件，定义Tiling策略的Key,标识不同的划分方式                     
└── tests                                  // UT实现
    └── ut                                 // tiling/kernel/aclnn UT实现
```

## 功能说明

- 算子功能：
本样例算子实现的是Matmul算子，支持bias，其数学表达式为：
  $$
  C =A * B +bias
  $$

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">M * K</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">K * N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_fp32</td></tr>
  </table>


## 编译运行 
- 配置环境变量  
以命令行方式下载样例代码，master分支为例 
```bash
cd ${git_clone_path}/experimental/matmul/matmul_fp32
```
根据当前环境上CANN开发套件包（toolkit包+ops包）的安装方式，选择对应配置环境变量的命令。  
  - 默认路径，root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=/usr/local/Ascend/cann
    ```
  - 默认路径，非root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=$HOME/Ascend/cann
    ```
  - 指定路径install_path，安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=${install_path}/cann
    ```


- 编译与安装自定义算子包
```bash
# 切换到工程根目录
cd ${git_clone_path}  
# 编译样例算子run包
bash build.sh --pkg  --soc=ascend910b --vendor_name=custom --ops=matmul_fp32 --experimental 
#安装自定义算子run包
./build_out/cann-ops-nn-${vendor_name}-${arch}_linux.run
```


- 编译+执行aclnn接口样例，采集样例性能：
```bash
# 切换matmul_fp32 aclnn执行用例目录
cd ${git_clone_path}/experimental/matmul/matmul_fp32/examples
# 编译+执行aclnn接口+采集性能数据
bash run.sh
# 切换aclnn用例性能数据目录
cd ${git_clone_path}/experimental/matmul/matmul_fp32/examples/output/msprof_result
```