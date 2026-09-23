# LayerNormGradV3

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
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 算子功能：[LayerNormV4](../../layer_norm_v4/README.md)的反向传播。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。
- 计算公式：

  $$
  res\_for\_gamma = (input - mean) \times rstd
  $$

  $$
  dy\_g = gradOut \times weight
  $$

  $$
  temp_1 = 1/N \times \sum_{reduce\_axis\_1} gradOut \times weight
  $$

  $$
  temp_2 = 1/N \times (input - mean) \times rstd \times \sum_{reduce\_axis\_1}(gradOut \times weight \times (input - mean) \times rstd)
  $$

  $$
  gradInputOut = (gradOut \times weight - (temp_1 + temp_2)) \times rstd
  $$

  $$
  gradWeightOut =  \sum_{reduce\_axis\_0}gradOut \times (input - mean) \times rstd
  $$

  $$
  gradBiasOut = \sum_{reduce\_axis\_0}gradOut
  $$

  其中，N为进行归一化计算的轴的维度，即归一化轴维度的大小。

## Ascend IR定义

Ascend IR定义所在头文件路径：[layer_norm_grad_v3_proto.h](../op_graph/layer_norm_grad_v3_proto.h)

```c++
REG_OP(LayerNormGradV3)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(pd_beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(output_mask, ListBool, {true, true, true})
    .OP_END_FACTORY_REG(LayerNormGradV3)
```

## 参数说明

| 参数名 | 输入/属性/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| dy (Tensor) | 必选输入 | 正向输出的梯度，对应公式中的`gradOut`。 | 支持空Tensor；至少为1维。 | float32、float16、bfloat16 | ND | 至少1维，形状为[A1,...,Ai,R1,...,Rj] |
| x (Tensor) | 必选输入 | 正向层归一化的输入，对应公式中的`input`。 | 支持空Tensor；数据类型和shape必须与`dy`一致。 | float32、float16、bfloat16 | ND | 与`dy`一致 |
| rstd (Tensor) | 必选输入 | 正向计算得到的标准差倒数，对应公式中的`rstd`。 | 支持空Tensor；shape必须与`mean`一致。 | float32、float16、bfloat16 | ND | 与`x`同维，形状为[A1,...,Ai,1,...,1] |
| mean (Tensor) | 必选输入 | 正向计算得到的均值，对应公式中的`mean`。 | 支持空Tensor；shape必须与`rstd`一致。 | float32、float16、bfloat16 | ND | 与`rstd`一致 |
| gamma (Tensor) | 必选输入 | 正向计算使用的缩放权重，对应公式中的`weight`。 | 支持空Tensor；至少为1维，shape必须与`dy`的末尾若干维一致。 | float32、float16、bfloat16 | ND | 至少1维，形状为[R1,...,Rj] |
| output_mask (list bool) | 可选属性 | 标记`pd_x`、`pd_gamma`和`pd_beta`三个输出是否有效，列表元素按上述输出顺序一一对应。 | 长度必须为3，默认值为{true, true, true}；元素为false时，对应输出中的数据无意义。 | - | - | - |
| pd_x (Tensor) | 必选输出 | 输入`x`的梯度，对应公式中的`gradInputOut`。 | 支持空Tensor；由`output_mask[0]`标记是否有效；数据类型和shape必须与`x`、`dy`一致。 | float32、float16、bfloat16 | ND | 与`dy`一致 |
| pd_gamma (Tensor) | 必选输出 | 缩放权重`gamma`的梯度，对应公式中的`gradWeightOut`。 | 支持空Tensor；由`output_mask[1]`标记是否有效；shape与`gamma`一致；数据类型与`gamma`相同。 | float32、float16、bfloat16 | ND | 与`gamma`一致 |
| pd_beta (Tensor) | 必选输出 | 偏置的梯度，对应公式中的`gradBiasOut`。 | 支持空Tensor；由`output_mask[2]`标记是否有效；shape与`gamma`一致；数据类型与`gamma`相同。 | float32、float16、bfloat16 | ND | 与`gamma`一致 |

- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：不支持空Tensor。

## 约束说明

- `dy`、`x`和`pd_x`的数据类型及shape必须相同；`rstd`和`mean`的shape必须相同。
- `gamma`至少为1维，需满足`rank(gamma) <= rank(dy)`，且`gamma`的各维度必须与`dy`的末尾对应维度相等；`rstd`和`mean`的非归一化维度与`x`一致，归一化维度大小均为1。

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
 * \file test_geir_layer_norm_grad_v3.cpp
 * \brief GE graph construction sample for LayerNormGradV3.
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
#include "../op_graph/layer_norm_grad_v3_proto.h"

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
    auto node = op::LayerNormGradV3("layer_norm_grad_v3");
    vector<int64_t> xShape = {2, 3, 4};
    vector<int64_t> statisticShape = {2, 3, 1};
    vector<int64_t> parameterShape = {4};

    ADD_INPUT(1, dy, xShape, 1.0f);
    ADD_INPUT(2, x, xShape, 0.0f);
    ADD_INPUT(3, rstd, statisticShape, 0.5f);
    ADD_INPUT(4, mean, statisticShape, 0.0f);
    ADD_INPUT(5, gamma, parameterShape, 4.0f);
    vector<bool> outputMask = {true, true, true};
    node.set_attr_output_mask(outputMask);

    // Six rows: x=[-2,2,-2,2], dy=[1,2,3,4], mean=0, variance=4.
    const float xRow[] = {-2.0f, 2.0f, -2.0f, 2.0f};
    const float dyRow[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const size_t count = inputTensors[0].GetSize() / sizeof(float);
    vector<float> xValues(count);
    vector<float> dyValues(count);
    for (size_t i = 0; i < count; ++i) {
        xValues[i] = xRow[i % 4];
        dyValues[i] = dyRow[i % 4];
    }
    auto ret = inputTensors[0].SetData(reinterpret_cast<uint8_t*>(dyValues.data()), dyValues.size() * sizeof(float));
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetData(dy) failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    ret = inputTensors[1].SetData(reinterpret_cast<uint8_t*>(xValues.data()), xValues.size() * sizeof(float));
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetData(x) failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);

    SET_OUTPUT(pd_x, xShape);
    SET_OUTPUT(pd_gamma, parameterShape);
    SET_OUTPUT(pd_beta, parameterShape);
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
    // xhat=[-1,1,-1,1], gamma*rstd=2, mean(dy)=2.5, mean(dy*xhat)=0.5.
    // dx=2*(dy-2.5-xhat*0.5); dgamma=6*dy*xhat; dbeta=6*dy.
    ok = CheckOutput(outputs[0], "dx", {2, 3, 4}, {-2.0f, -2.0f, 2.0f, 2.0f}) && ok;
    ok = CheckOutput(outputs[1], "dgamma", {4}, {-6.0f, 12.0f, -18.0f, 24.0f}) && ok;
    ok = CheckOutput(outputs[2], "dbeta", {4}, {6.0f, 12.0f, 18.0f, 24.0f}) && ok;
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
    LOG_PRINT("LayerNormGradV3 graph run success, output count: %zu\n", outputTensors.size());
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

    Graph graph("layer_norm_grad_v3_graph");
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
