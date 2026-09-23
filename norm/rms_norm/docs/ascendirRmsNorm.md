# RmsNorm

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->
- <term>Kirin X90 处理器系列产品</term>：支持
- <term>Kirin 9030 处理器系列产品</term>：支持

## 功能说明

- 算子功能：

  对输入张量末尾与`gamma`对应的若干维执行均方根归一化，并使用`gamma`对归一化结果进行缩放。与LayerNorm相比，RmsNorm不执行减均值操作，常用于大模型的归一化计算。算子同时输出均方根的倒数`rstd`，可供反向计算使用。

- 计算公式：

  对每个非归一化位置$p$，将对应的归一化后缀展平为$n$个元素，则：

  $$
  rstd_p=\frac{1}{\sqrt{\frac{1}{n}\sum_{q=1}^{n}x_{p,q}^2+\epsilon}}
  $$

  $$
  y_{p,q}=x_{p,q}\cdot rstd_p\cdot \gamma_q
  $$

  其中，$n$表示`gamma`的元素个数，$q$表示展平后的归一化索引，$p$表示`gamma`所对应维度之外的非归一化索引，$\epsilon$表示添加到均方值中、用于提高数值稳定性的数。

  下文令$j=\operatorname{rank}(gamma)$，$i=\operatorname{rank}(x)-j$，其中$j\geq 1$、$i\geq 0$。`x`的shape记为[A1,...,Ai,R1,...,Rj]，`gamma`的shape记为[R1,...,Rj]；当$i=0$时，[A1,...,Ai]为空序列。

## Ascend IR定义

Ascend IR定义所在头文件路径：[rms_norm_proto.h](../op_graph/rms_norm_proto.h)

```c++
REG_OP(RmsNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(RmsNorm)
```

## 参数说明

| 参数名 | 输入/属性/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| x (Tensor) | 必选输入 | 待归一化的输入张量，对应公式中的$x$。 | 各维大小必须大于0，不支持空Tensor；有效数据类型组合见“约束说明”。 | float32、float16、bfloat16 | ND | 1-8维，形状为[A1,...,Ai,R1,...,Rj] |
| gamma (Tensor) | 必选输入 | 归一化计算的缩放因子，对应公式中的$\gamma$。 | 各维大小必须大于0；与`x`的shape关系见“约束说明”；有效数据类型组合见“约束说明”。 | float32、float16、bfloat16 | ND | 1-8维，形状为[R1,...,Rj] |
| epsilon (float) | 可选属性 | 添加到均方值中的数，对应公式中的$\epsilon$。 | 取值不能小于0，默认值为1e-6。 | - | - | - |
| y (Tensor) | 必选输出 | 归一化并缩放后的结果，对应公式中的$y$。 | shape和数据类型与`x`相同。 | float32、float16、bfloat16 | ND | 与`x`一致 |
| rstd (Tensor) | 必选输出 | 均方根的倒数，对应公式中的$rstd$。 | 数据类型固定为float32；与`gamma`对应的末尾各维大小均为1，其余维度与`x`相同。 | float32 | ND | 与`x`同维，形状为[A1,...,Ai,1,...,1]，末尾共`j`个1 |

- <term>Atlas推理系列产品</term>、<term>Kirin X90 处理器系列产品</term>、<term>Kirin 9030 处理器系列产品</term>：`x`、`gamma`和`y`不支持bfloat16。

## 约束说明

- `gamma`的维度数不能大于`x`的维度数。若`gamma`为`j`维，则`gamma`的shape必须与`x`的末尾`j`个维度完全相同，即`x`的shape为[A1,...,Ai,R1,...,Rj]时，`gamma`的shape必须为[R1,...,Rj]。
- <term>Atlas推理系列产品</term>：归一化维[R1,...,Rj]的数据量（R1×...×Rj×单个元素字节数）必须大于等于32 Bytes。
- 各产品支持的数据类型组合如下：

  - <term>Ascend 950PR&950DT系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：

    | `x`数据类型 | `gamma`数据类型 | `y`数据类型 | `rstd`数据类型 |
    | --- | --- | --- | --- |
    | float16 | float32 | float16 | float32 |
    | bfloat16 | float32 | bfloat16 | float32 |
    | float16 | float16 | float16 | float32 |
    | bfloat16 | bfloat16 | bfloat16 | float32 |
    | float32 | float32 | float32 | float32 |

  - <term>Atlas推理系列产品</term>、<term>Kirin X90 处理器系列产品</term>、<term>Kirin 9030 处理器系列产品</term>：

    | `x`数据类型 | `gamma`数据类型 | `y`数据类型 | `rstd`数据类型 |
    | --- | --- | --- | --- |
    | float16 | float16 | float16 | float32 |
    | float32 | float32 | float32 | float32 |

- 默认确定性实现。

## 调用示例

示例代码如下，仅供参考。GE图模式的编译和执行过程请参考[算子调用](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式)。

```c++
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_rms_norm.cpp
 * \brief GE graph construction sample for RmsNorm.
 */

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "../op_graph/rms_norm_proto.h"

#define FAILED (-1)
#define SUCCESS 0

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)              \
    do {                                     \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

using namespace ge;
using std::vector;

namespace {
std::string GetGeError()
{
    const AscendString errorMessage = GEGetErrorMsgV2();
    return errorMessage.GetString() == nullptr ? "" : errorMessage.GetString();
}

int32_t GenerateFloatData(const vector<int64_t>& shape, float value, TensorDesc& desc, Tensor& tensor)
{
    int64_t elementCount = 1;
    for (const int64_t dim : shape) {
        CHECK_RET(dim > 0, LOG_PRINT("[ERROR] Input dimensions must be positive.\n"); return FAILED);
        elementCount *= dim;
    }
    desc.SetRealDimCnt(shape.size());
    auto ret = tensor.SetTensorDesc(desc);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetTensorDesc failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    vector<float> data(elementCount, value);
    ret = tensor.SetData(reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(float));
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetData failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    return SUCCESS;
}

#define ADD_INPUT(index, inputName, inputShape, inputValue)                                                           \
    do {                                                                                                              \
        auto inputOp = op::Data("input_" #index).set_attr_index((index) - 1);                                         \
        TensorDesc inputDesc(Shape(inputShape), FORMAT_ND, DT_FLOAT);                                                 \
        inputDesc.SetPlacement(kPlacementHost);                                                                       \
        Tensor inputTensor;                                                                                           \
        auto dataRet = GenerateFloatData(inputShape, inputValue, inputDesc, inputTensor);                             \
        CHECK_RET(dataRet == SUCCESS, return FAILED);                                                                 \
        auto ret = inputOp.update_input_desc_x(inputDesc);                                                            \
        CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] Data::update_input_desc_x failed, status=%u, error=%s\n",  \
                                                  ret, GetGeError().c_str());                                         \
                  return FAILED);                                                                                     \
        ret = inputOp.update_output_desc_y(inputDesc);                                                                \
        CHECK_RET(ret == GRAPH_SUCCESS, LOG_PRINT("[ERROR] Data::update_output_desc_y failed, status=%u, error=%s\n", \
                                                  ret, GetGeError().c_str());                                         \
                  return FAILED);                                                                                     \
        ret = node.update_input_desc_##inputName(inputDesc);                                                          \
        CHECK_RET(ret == GRAPH_SUCCESS,                                                                               \
                  LOG_PRINT("[ERROR] Operator::update_input_desc_" #inputName " failed, status=%u, error=%s\n", ret,  \
                            GetGeError().c_str());                                                                    \
                  return FAILED);                                                                                     \
        node.set_input_##inputName(inputOp);                                                                          \
        inputTensors.push_back(inputTensor);                                                                          \
        inputOps.push_back(inputOp);                                                                                  \
        ret = graph.AddOp(inputOp);                                                                                   \
        CHECK_RET(ret == GRAPH_SUCCESS,                                                                               \
                  LOG_PRINT("[ERROR] Graph::AddOp failed, status=%u, error=%s\n", ret, GetGeError().c_str());         \
                  return FAILED);                                                                                     \
    } while (0)

#define SET_OUTPUT(outputName, outputShape)                                                                            \
    do {                                                                                                               \
        TensorDesc outputDesc(Shape(outputShape), FORMAT_ND, DT_FLOAT);                                                \
        auto ret = node.update_output_desc_##outputName(outputDesc);                                                   \
        CHECK_RET(ret == GRAPH_SUCCESS,                                                                                \
                  LOG_PRINT("[ERROR] Operator::update_output_desc_" #outputName " failed, status=%u, error=%s\n", ret, \
                            GetGeError().c_str());                                                                     \
                  return FAILED);                                                                                      \
    } while (0)

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps)
{
    auto node = op::RmsNorm("rms_norm");
    vector<int64_t> xShape = {2, 16};
    vector<int64_t> gammaShape = {16};
    vector<int64_t> rstdShape = {2, 1};

    ADD_INPUT(1, x, xShape, 2.0f);
    ADD_INPUT(2, gamma, gammaShape, 2.0f);
    node.set_attr_epsilon(1e-6f);

    SET_OUTPUT(y, xShape);
    SET_OUTPUT(rstd, rstdShape);
    outputOps.push_back(node);
    return SUCCESS;
}

bool CheckOutput(const Tensor& tensor, const std::string& name, const vector<int64_t>& shape,
                 const vector<float>& expectedPattern)
{
    const auto desc = tensor.GetTensorDesc();
    size_t count = 1;
    for (const int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    CHECK_RET(desc.GetShape().GetDims() == shape && desc.GetDataType() == DT_FLOAT && desc.GetFormat() == FORMAT_ND &&
                  tensor.GetSize() == count * sizeof(float) && tensor.GetData() != nullptr,
              LOG_PRINT("[CHECK] %s FAIL: unexpected shape, dtype, format, or data size\n", name.c_str());
              return false);
    vector<float> values(count);
    std::memcpy(values.data(), tensor.GetData(), count * sizeof(float));
    CHECK_RET(!expectedPattern.empty() && count % expectedPattern.size() == 0,
              LOG_PRINT("[CHECK] %s FAIL: invalid expected pattern\n", name.c_str());
              return false);
    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i) {
        const float expected = expectedPattern[i % expectedPattern.size()];
        const float tolerance = 1e-4f + 1e-5f * std::fabs(expected);
        if (!std::isfinite(values[i]) || std::fabs(values[i] - expected) > tolerance) {
            if (mismatches == 0) {
                LOG_PRINT("[CHECK] %s[%zu]=%.9g expected=%.9g tolerance=%.9g\n", name.c_str(), i, values[i], expected,
                          tolerance);
            }
            ++mismatches;
        }
    }
    LOG_PRINT("[CHECK] %s first values=[", name.c_str());
    for (size_t i = 0; i < count && i < 4; ++i) {
        LOG_PRINT("%s%.9g", i == 0 ? "" : ", ", values[i]);
    }
    LOG_PRINT("], checked=%zu, mismatches=%zu: %s\n", count, mismatches, mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

int32_t ValidateOutputs(const vector<Tensor>& outputs)
{
    CHECK_RET(outputs.size() == 2, LOG_PRINT("[CHECK] FAIL: expected 2 outputs, got %zu\n", outputs.size());
              return FAILED);
    bool ok = true;
    ok = CheckOutput(outputs[0], "y", {2, 16}, {2.0f}) && ok;
    ok = CheckOutput(outputs[1], "rstd", {2, 1}, {0.5f}) && ok;
    LOG_PRINT("[CHECK] total: %s\n", ok ? "PASS" : "FAIL");
    return ok ? SUCCESS : FAILED;
}

int32_t RunGraph(Graph& graph, const vector<Tensor>& inputTensors)
{
    std::map<AscendString, AscendString> buildOptions;
    Session* session = new (std::nothrow) Session(buildOptions);
    CHECK_RET(session != nullptr, LOG_PRINT("[ERROR] Session allocation failed.\n"); return FAILED);

    constexpr uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions;
    auto ret = session->AddGraph(graphId, graph, graphOptions);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Session::AddGraph failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              delete session; return FAILED);

    vector<Tensor> outputTensors;
    ret = session->RunGraph(graphId, inputTensors, outputTensors);
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Session::RunGraph failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              delete session; return FAILED);
    LOG_PRINT("RmsNorm graph run success, output count: %zu\n", outputTensors.size());
    const int32_t result = ValidateOutputs(outputTensors);
    delete session;
    return result;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    auto status = GEInitialize(globalOptions);
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEInitialize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);

    Graph graph("rms_norm_graph");
    vector<Tensor> inputTensors;
    vector<Operator> inputOps;
    vector<Operator> outputOps;
    int32_t ret = BuildGraph(graph, inputTensors, inputOps, outputOps);
    if (ret == SUCCESS) {
        graph.SetInputs(inputOps).SetOutputs(outputOps);
        ret = RunGraph(graph, inputTensors);
    }

    status = GEFinalize();
    CHECK_RET(status == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] GEFinalize failed, status=%u, error=%s\n", status, GetGeError().c_str());
              return FAILED);
    return ret;
}
```
