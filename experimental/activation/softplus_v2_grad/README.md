## `SoftplusV2Grad`自定义算子样例说明 

本样例通过`Ascend C`编程语言实现了`SoftplusV2Grad`算子。

### 算子描述

`SoftplusV2Grad`是Softplus算子的反向传播。

### 算子规格描述

| 名称       | 类别 | dtype          | format | shape |
| ---------- | ---- | -------------- | ------ | ----- |
| gradOutput | 输入 | fp16/fp32/bf16 | ND     | all   |
| self       | 输入 | fp16/fp32/bf16 | ND     | all   |
| beta       | 输入 | float          | -      | -     |
| threshold  | 输入 | float          | -      | -     |
| gradInput  | 输出 | fp16/fp32/bf16 | ND     | all   |

### 支持的产品型号

本样例支持如下产品型号：

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品

### 目录结构介绍

```
├── docs                        // 算子文档目录
├── example                     // 调用示例目录
├── op_host                     // host目录
├── op_kernel                   // kernel目录
└── tests                       // 测试用例目录
└── SoftplusBackward.json       // SoftplusV2Grad算子原型定义json
```

### 环境要求

编译运行此样例前，请完成CANN开发运行环境的部署。

### 算子包编译部署

  - 进入到仓库目录

    ```bash
    cd ${git_clone_path}/ops-nn/
    ```

  - 执行编译

    ```bash
    bash build.sh --pkg --soc=ascend910b --ops=softplus_v2_grad
    ```

  - 部署算子包

    ```bash
    ./build_out/cann-ops-nn-custom-linux.aarch64.run
    ```

### 算子调用

<table>
    <th>目录</th><th>描述</th>
    <tr>
        <td><a href="./examples/AclNNInvocationNaive"> AclNNInvocationNaive</td><td>通过aclnn调用的方式调用SoftplusV2Grad算子。</td>
    </tr>
    <tr>
        <td><a href="./examples/ascendoptest"> ascendoptest</td><td>通过ascendoptest调用的方式调用SoftplusV2Grad算子。</td>
    </tr>
</table>


### 参考算子

Abs：[ops-math/math/abs/README.md · CANN/ops-math - GitCode]

### 更新说明

| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/17 | 新增本readme |
| 2025/11/17 | 新增参考算子 |

参考：
## 贡献说明
 	 
 	 | 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
 	 | ---- | ---- | ---- | ---- | ---- |
 	 | ilovescrapy | 个人开发者 | ReluGrad | 2025/12/26 | ReluGrad算子适配开源仓 |