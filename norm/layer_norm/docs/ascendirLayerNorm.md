# LayerNorm

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
- <term>Atlas 200I/500 A2推理产品</term>：支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

- 算子功能：

  对指定层进行均值为0、标准差为1的归一化计算。
  - 归一化：对输入张量的每个样本进行归一化处理，使得每个样本的均值为0，方差为1。
  - 缩放和偏移：在归一化之后，可以通过缩放因子和偏移量进一步调整归一化后的输出，以适应不同的模型需求。

- 计算公式：

  $$
  mean = {E}[x]
  $$

  $$
  variance = \mathrm{Var}[x]
  $$

  $$
  y = w \times (\frac{x - mean}{ \sqrt{\mathrm{Var}[x] + eps}}) + b
  $$

  其中，E[x]表示输入的均值，Var[x]表示输入的方差。

## Ascend IR定义

Ascend IR定义所在头文件路径：[layer_norm_proto.h](../op_graph/layer_norm_proto.h)

```c++
REG_OP(LayerNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(begin_norm_axis, Int, 0)
    .ATTR(begin_params_axis, Int, 0)
    .ATTR(epsilon, Float, 0.0000001f)
    .OP_END_FACTORY_REG(LayerNorm)
```

## 参数说明

| 参数名 | 输入/属性/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| x (Tensor) | 必选输入 | 待归一化的输入，对应公式中的 `x`。 | 归一化范围由 `begin_norm_axis` 指定；不支持空Tensor。 | float32、float16、bfloat16 | ND | 至少1维，形状为[A1,...,Ai,R1,...,Rj] |
| gamma (Tensor) | 必选输入 | 缩放权重，对应公式中的`w`。 | 必须提供；shape与 `x` 从 `begin_params_axis` 到最后一维的shape一致，类型与 `x` 一致或为float32。 | float32、float16、bfloat16 | ND | `x` 从 `begin_params_axis` 开始的后缀shape |
| beta (Tensor) | 必选输入 | 偏移量，对应公式中的`b`。 | 必须提供；shape和类型与 `gamma` 相同。 | float32、float16、bfloat16 | ND | 与 `gamma` 一致 |
| begin_norm_axis (int) | 可选属性 | 开始执行归一化计算的轴。 | 默认值为0；支持负数索引，取值范围为[-rank(`x`), rank(`x`)-1]。 | - | - | - |
| begin_params_axis (int) | 可选属性 | `gamma`、`beta` 对应 `x` 的起始轴。 | 默认值为0；支持负数索引，取值范围为[-rank(`x`), rank(`x`)-1]。 | - | - | - |
| epsilon (float) | 可选属性 | 为保证数值稳定而加到方差上的值，对应公式中的`eps`。 | GE IR默认值为1e-7。 | - | - | - |
| y (Tensor) | 必选输出 | 归一化、缩放与偏移后的结果。 | shape和类型与 `x` 相同。 | float32、float16、bfloat16 | ND | 与 `x` 一致 |
| mean (Tensor) | 必选输出 | 输入在归一化维度上的均值。 | 类型与 `gamma`、`beta` 相同；从 `begin_norm_axis` 开始的维度大小均为1。 | float32、float16、bfloat16 | ND | 与 `x` 同维，形状为[A1,...,Ai,1,...,1] |
| variance (Tensor) | 必选输出 | 输入在归一化维度上的方差。 | shape和类型与 `mean` 相同。 | float32、float16、bfloat16 | ND | 与 `mean` 一致 |

## 约束说明

- `gamma`与`beta`均为必选输入，数据类型必须相同，并且为`x`的数据类型或float32；`y`的数据类型与`x`相同，`mean`和`variance`的数据类型与`gamma`相同。
- `gamma`与`beta`的shape必须相同。将`begin_params_axis`换算为非负索引后，`gamma`的shape必须等于`x`从该轴到最后一维的shape，即满足`begin_params_axis + rank(gamma) = rank(x)`。
- `mean`和`variance`的shape由`begin_norm_axis`决定：该轴之前的维度与`x`一致，该轴及之后的维度均为1。

## 调用示例

示例代码如下，仅供参考。GE图模式的编译和执行过程请参考[算子调用](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式)。

```c++
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_layer_norm.cpp
 * \brief GE graph construction sample for LayerNorm.
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
#include "../op_graph/layer_norm_proto.h"

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
    vector<float> data(elementCount, static_cast<float>(value));
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
    auto node = op::LayerNorm("layer_norm");
    vector<int64_t> xShape = {2, 3, 4};
    vector<int64_t> parameterShape = {4};
    vector<int64_t> statisticShape = {2, 3, 1};

    ADD_INPUT(1, x, xShape, 1.0f);
    // Six identical rows [1, 1, 3, 3]: mean=2, variance=1.
    const float row[] = {1.0f, 1.0f, 3.0f, 3.0f};
    vector<float> xValues(24);
    for (size_t i = 0; i < xValues.size(); ++i) {
        xValues[i] = row[i % 4];
    }
    auto ret = inputTensors[0].SetData(reinterpret_cast<uint8_t*>(xValues.data()), xValues.size() * sizeof(float));
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetData(x) failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ADD_INPUT(2, gamma, parameterShape, 2.0f);
    ADD_INPUT(3, beta, parameterShape, 3.0f);
    node.set_attr_begin_norm_axis(2);
    node.set_attr_begin_params_axis(2);
    node.set_attr_epsilon(3.0f);

    SET_OUTPUT(y, xShape);
    SET_OUTPUT(mean, statisticShape);
    SET_OUTPUT(variance, statisticShape);
    outputOps.push_back(node);
    return SUCCESS;
}

// These examples use simple FP32 inputs, so every output has a simple
// analytically known value. Check all elements, not only the printed preview.
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
    LOG_PRINT("] expected pattern=[");
    for (size_t i = 0; i < expectedPattern.size(); ++i) {
        LOG_PRINT("%s%.9g", i == 0 ? "" : ", ", expectedPattern[i]);
    }
    LOG_PRINT("], checked=%zu, mismatches=%zu: %s\n", count, mismatches, mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0;
}

int32_t ValidateOutputs(const vector<Tensor>& outputs)
{
    CHECK_RET(outputs.size() == 3, LOG_PRINT("[CHECK] FAIL: expected 3 outputs, got %zu\n", outputs.size());
              return FAILED);
    bool ok = true;
    // mean=2, variance=1; sqrt(variance+epsilon)=sqrt(1+3)=2.
    // gamma=2, beta=3 -> y=(x-2)/2*2+3=[2,2,4,4] per row.
    ok = CheckOutput(outputs[0], "y", {2, 3, 4}, {2.0f, 2.0f, 4.0f, 4.0f}) && ok;
    ok = CheckOutput(outputs[1], "mean", {2, 3, 1}, {2.0f}) && ok;
    ok = CheckOutput(outputs[2], "variance", {2, 3, 1}, {1.0f}) && ok;
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
    LOG_PRINT("LayerNorm graph run success, output count: %zu\n", outputTensors.size());
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

    Graph graph("layer_norm_graph");
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
