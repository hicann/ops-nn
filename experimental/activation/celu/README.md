# Celu

> **Copyright (c) 2026 Huawei Technologies Co., Ltd.**
>
> This program is free software, you can redistribute it and/or modify it under the terms and conditions of
> the CANN Open Software License Agreement Version 2.0 (the "License").
> Please refer to the License for details. You may not use this file except in compliance with the License.
> THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
> INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
> See LICENSE in the root of the software repository for the full text of the License.
>
> **AI Disclaimer**: Portions of this code were AI-generated and have been technically reviewed for functional
> accuracy and security.

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |
| <term>Kirin xxx</term> | × |

## 功能说明

- **算子功能**：Celu（Continuously Differentiable Exponential Linear Unit）是一种连续可微的激活函数，对正输入返回固定值，对负输入返回指数衰减曲线。
- **计算公式**：

  $$
  y = \begin{cases}
  \alpha_3 \times 3 & \text{if } x \geq 0 \\
  \alpha_1 \times \left(\exp\left(\frac{x}{\alpha_2}\right) - 1\right) & \text{if } x < 0
  \end{cases}
  $$

  其中：
  - 当 `x >= 0` 时，输出为常数 `alpha3 * 3`
  - 当 `x < 0` 时，输出为指数衰减形式 `alpha1 * (exp(x / alpha2) - 1)`
- **精度保护**：针对 `exp` 操作的溢出风险，对负区输入执行 Mins 截断保护（FP16 截断阈值为 11.0，FP32 截断阈值为 87.0），防止指数运算溢出。

## 目录结构

```
celu/
├── op_host/                    # Host 侧代码
│   ├── celu_def.cpp            # 算子定义（输入输出规格、属性定义）
│   ├── celu_infershape.cpp     # 形状推导
│   └── celu_tiling.cpp         # Tiling 配置
├── op_kernel/                  # Kernel 侧代码
│   ├── celu.cpp                # Kernel 主入口（模板分发）
│   ├── celu.h                  # Kernel 头文件
│   ├── celu_tiling_data.h      # TilingData 结构体定义
│   └── celu_tiling_key.h       # TilingKey 分支定义
├── op_api/                     # ACLNN 接口（自动生成）
│   ├── aclnn_celu.cpp          # L2 API 实现
│   ├── aclnn_celu.h            # L2 API 头文件
│   ├── celu.cpp                # L0 API 实现
│   └── celu.h                  # L0 API 头文件
├── tests/                      # 测试代码
│   ├── ut/                     # 单元测试
│   └── st/                     # 系统测试
├── examples/                   # 调用示例
│   └── arch35/
│       └── test_aclnn_celu.cpp # aclnn 两段式调用示例
├── CMakeLists.txt              # 构建配置
└── build.sh                    # 构建脚本
```

## 快速验证

```bash
# 1. 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 编译算子（仅支持 Ascend950）
bash build.sh --soc=ascend950 -j8

# 3. 运行 UT 测试
bash build.sh -u --soc=ascend950

# 4. 运行 ST 测试（Real 模式，需 NPU）
bash build.sh -s --soc=ascend950
```

## 参数说明

<table style="table-layout: fixed; width: 100%"><colgroup>
<col style="width: 12%">
<col style="width: 12%">
<col style="width: 32%">
<col style="width: 24%">
<col style="width: 20%">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>公式中的输入 x，任意形状的张量。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha1</td>
    <td>属性</td>
    <td><ul><li>负区指数曲线的缩放系数，对应公式中的 alpha1。</li><li>默认值为 1.0。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>alpha2</td>
    <td>属性</td>
    <td><ul><li>负区指数曲线的斜率系数，对应公式中的 alpha2，必须大于 0。</li><li>默认值为 1.0。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>alpha3</td>
    <td>属性</td>
    <td><ul><li>正区的固定输出值，对应公式中的 alpha3。</li><li>默认值为 1.0。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>公式中的输出 y，与输入 x 形状相同的张量。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- 本算子仅支持 Ascend 950 芯片架构（DAV_3510 / arch35），不支持其他芯片。
- 输入 tensor 支持的数据类型为 FLOAT16 和 FLOAT，输出与输入数据类型一致。
- 输入 tensor 的格式必须为 ND。
- alpha2 属性必须大于 0，否则可能导致除零或数值不稳定。
- 输入支持任意 shape（包括 0 元素空 tensor），算子内部已处理空 tensor 边界情况。
- 大 tensor 场景下，算子内部采用多核切分和 UB Tiling 策略，自动分配至多个 AI Core 并行处理。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn 调用</td>
    <td><a href="examples/arch35/test_aclnn_celu.cpp">test_aclnn_celu</a></td>
    <td>参见下方调用示例。完成算子编译后，可直接运行验证。</td>
  </tr>
</tbody></table>

### C++ 调用示例（aclnn 两段式调用）

```cpp
#include "acl/acl.h"
#include "aclnn_celu.h"

int main()
{
    // 1. 初始化 ACL 运行时环境
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 2. 创建输入 tensor（4x4，FLOAT 类型，ND 格式）
    std::vector<int64_t> xShape = {4, 4};
    std::vector<float> xHostData = {
        1.0f, -1.0f, 2.0f, -2.0f,
        0.5f, -0.5f, 3.0f, -3.0f,
        0.0f, -0.1f, 4.0f, -4.0f,
        0.1f, -5.0f, 1.5f, -1.5f
    };
    void* xDeviceAddr = nullptr;
    aclrtMalloc(&xDeviceAddr, xHostData.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(xDeviceAddr, xHostData.size() * sizeof(float),
                xHostData.data(), ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor* x = aclCreateTensor(
        xShape.data(), xShape.size(), ACL_FLOAT,
        nullptr, 0, ACL_FORMAT_ND, xShape.data(), xShape.size(), xDeviceAddr);

    // 3. 创建输出 tensor
    void* outDeviceAddr = nullptr;
    aclrtMalloc(&outDeviceAddr, xHostData.size() * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclTensor* out = aclCreateTensor(
        xShape.data(), xShape.size(), ACL_FLOAT,
        nullptr, 0, ACL_FORMAT_ND, xShape.data(), xShape.size(), outDeviceAddr);

    // 4. 设置属性参数
    double alpha1 = 1.0, alpha2 = 1.0, alpha3 = 1.0;

    // 5. 第一段：获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnCeluGetWorkspaceSize(x, alpha1, alpha2, alpha3, out, &workspaceSize, &executor);

    // 6. 分配 workspace 内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 7. 第二段：执行计算
    aclnnCelu(workspaceAddr, workspaceSize, executor, stream);

    // 8. 同步并拷贝结果回 Host
    aclrtSynchronizeStream(stream);
    std::vector<float> result(xHostData.size());
    aclrtMemcpy(result.data(), result.size() * sizeof(float),
                outDeviceAddr, ACL_MEMCPY_DEVICE_TO_HOST);

    // 9. 打印结果并清理资源
    for (size_t i = 0; i < result.size(); i++) {
        printf("celu input[%zu] = %f, output[%zu] = %f\n", i, xHostData[i], i, result[i]);
    }

    // 释放资源
    aclDestroyTensor(x);
    aclDestroyTensor(out);
    aclrtFree(xDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```

## 参考资源

* [算子设计文档](docs/celu设计文档.md) <!-- 待开发阶段补充 -->
