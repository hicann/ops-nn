# LayerNormV4

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->
- <term>Kirin X90 processor series products</term>: Supported
- <term>Kirin 9030 processor series products</term>: Supported

## Function

- Operator function:

  Performs the normalization computation with a mean of 0 and a standard deviation of 1 on the specified layer.
  - Normalization: normalizes each sample of the input tensor so that the mean of each sample is 0 and the variance is 1.
  - Scaling and offset: after normalization, the normalized output can be further adjusted by a scaling factor and an offset to meet the requirements of different models.

- Computation formula:

  $$
  mean = {E}[x]
  $$

  $$
  rstd = \frac{1}{ \sqrt{\mathrm{Var}[x] + eps}}
  $$

  $$
  y = w*((x - mean) * rstd) + b
  $$

  where E[x] denotes the mean of the input, and Var[x] denotes the variance of the input.

## Ascend IR Definition

The header file containing the Ascend IR definition: [layer_norm_v4_proto.h](../op_graph/layer_norm_v4_proto.h)

```c++
REG_OP(LayerNormV4)
    .INPUT(x, "T1")
    .INPUT(normalized_shape, "T2")
    .OPTIONAL_INPUT(gamma, "T3")
    .OPTIONAL_INPUT(beta, "T4")
    .OUTPUT(y, "T5")
    .OUTPUT(mean, "T6")
    .OUTPUT(rstd, "T6")
    .ATTR(epsilon, Float, 0.00001f)
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_INT32, DT_INT64}))
    .DATATYPE(T3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T4, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T5, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T6, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(LayerNormV4)
```

## Parameters

| Parameter name | Input/Attribute/Output | Description | Usage instructions | Data type | Data format | Dimensions (shape) |
| --- | --- | --- | --- | --- | --- | --- |
| x (Tensor) | Required input | Input tensor to be normalized, corresponding to `x` in the formula. | Empty tensors are not supported; the size of each dimension must be greater than 0. | float32, float16, bfloat16 | ND | At least 1 dimension, with the shape [A1,...,Ai,R1,...,Rj] |
| normalized_shape (Tensor) | Required input | Denotes the dimensions on which the norm computation is performed, with the values [R1,...,Rj]. | The length is less than or equal to the shape length of input `x`, and it cannot be empty; a scalar tensor is supported, in which case `j=1`, indicating normalization over the last dimension of `x`. | int32, int64 | ND | 0 or 1 dimension |
| gamma (Tensor) | Optional input | Scaling weight, corresponding to `w` in the formula. | When it exists, the shape must be [R1,...,Rj]; when it is not provided, it is processed as an all-1 tensor of the same shape. | float32, float16, bfloat16 | ND | At least 1 dimension when it exists, with the shape [R1,...,Rj] |
| beta (Tensor) | Optional input | Offset, corresponding to `b` in the formula. | When it exists, the shape must be [R1,...,Rj]; when it is not provided, it is processed as an all-0 tensor of the same shape. | float32, float16, bfloat16 | ND | At least 1 dimension when it exists, with the shape [R1,...,Rj] |
| epsilon (float) | Optional attribute | Value added to the variance to ensure numerical stability, corresponding to `eps` in the formula. | The default value is 0.00001. | - | - | - |
| y (Tensor) | Required output | Layer normalization computation result, corresponding to `y` in the formula. | The data type and shape are consistent with `x`. | float32, float16, bfloat16 | ND | Consistent with `x` |
| mean (Tensor) | Required output | Mean over the normalized dimensions, corresponding to `mean` in the formula. | The data type of the current implementation is float32; the trailing `j` dimensions all have a size of 1. | float32, float16, bfloat16 | ND | Same dimensions as `x`, with the shape [A1,...,Ai,1,...,1] |
| rstd (Tensor) | Required output | Reciprocal of the standard deviation over the normalized dimensions, corresponding to `rstd` in the formula. | The data type of the current implementation is float32, and the shape is consistent with `mean`. | float32, float16, bfloat16 | ND | Consistent with `mean` |

## Constraints

- The data type of `y` must be the same as that of `x`; `mean` and `rstd` currently support only float32.
- When both `gamma` and `beta` exist, both of them are at least 1 dimension, and their shapes and data types must be the same; their data type is the data type of `x` or float32. When only one of the two inputs exists, that input is at least 1 dimension, and its data type is the data type of `x` or float32.

## Invocation Sample

The following is the sample code, for reference only. For the compilation and execution process in GE graph mode, refer to [Operator Invocation](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式).

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
 * \file test_geir_layer_norm_v4.cpp
 * \brief GE graph construction sample for LayerNormV4.
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
#include "../op_graph/layer_norm_v4_proto.h"

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

int32_t GenerateData(const vector<int64_t>& shape, DataType dtype, double value, TensorDesc& desc, Tensor& tensor)
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
    if (dtype == DT_FLOAT) {
        vector<float> data(elementCount, static_cast<float>(value));
        ret = tensor.SetData(reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(float));
        CHECK_RET(ret == GRAPH_SUCCESS,
                  LOG_PRINT("[ERROR] Tensor::SetData failed, status=%u, error=%s\n", ret, GetGeError().c_str());
                  return FAILED);
        return SUCCESS;
    }
    if (dtype == DT_INT32) {
        vector<int32_t> data(elementCount, static_cast<int32_t>(value));
        ret = tensor.SetData(reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(int32_t));
        CHECK_RET(ret == GRAPH_SUCCESS,
                  LOG_PRINT("[ERROR] Tensor::SetData failed, status=%u, error=%s\n", ret, GetGeError().c_str());
                  return FAILED);
        return SUCCESS;
    }
    LOG_PRINT("[ERROR] Unsupported input dtype: %d\n", static_cast<int>(dtype));
    return FAILED;
}

#define ADD_INPUT(index, inputName, inputDtype, inputShape, inputValue)                                               \
    do {                                                                                                              \
        auto inputOp = op::Data("input_" #index).set_attr_index((index) - 1);                                         \
        TensorDesc inputDesc(Shape(inputShape), FORMAT_ND, inputDtype);                                               \
        inputDesc.SetPlacement(kPlacementHost);                                                                       \
        Tensor inputTensor;                                                                                           \
        auto dataRet = GenerateData(inputShape, inputDtype, inputValue, inputDesc, inputTensor);                      \
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

#define SET_OUTPUT(outputName, outputDtype, outputShape)                                                               \
    do {                                                                                                               \
        TensorDesc outputDesc(Shape(outputShape), FORMAT_ND, outputDtype);                                             \
        auto ret = node.update_output_desc_##outputName(outputDesc);                                                   \
        CHECK_RET(ret == GRAPH_SUCCESS,                                                                                \
                  LOG_PRINT("[ERROR] Operator::update_output_desc_" #outputName " failed, status=%u, error=%s\n", ret, \
                            GetGeError().c_str());                                                                     \
                  return FAILED);                                                                                      \
    } while (0)

int32_t BuildGraph(Graph& graph, vector<Tensor>& inputTensors, vector<Operator>& inputOps, vector<Operator>& outputOps)
{
    auto node = op::LayerNormV4("layer_norm_v4");
    vector<int64_t> xShape = {2, 3, 4};
    vector<int64_t> normalizedShapeTensorShape = {1};
    vector<int64_t> parameterShape = {4};
    vector<int64_t> statisticShape = {2, 3, 1};

    ADD_INPUT(1, x, DT_FLOAT, xShape, 1.0);
    // Six identical rows [1, 2, 2, 3]: mean=2, variance=0.5.
    const float row[] = {1.0f, 2.0f, 2.0f, 3.0f};
    vector<float> xValues(24);
    for (size_t i = 0; i < xValues.size(); ++i) {
        xValues[i] = row[i % 4];
    }
    auto ret = inputTensors[0].SetData(reinterpret_cast<uint8_t*>(xValues.data()), xValues.size() * sizeof(float));
    CHECK_RET(ret == GRAPH_SUCCESS,
              LOG_PRINT("[ERROR] Tensor::SetData(x) failed, status=%u, error=%s\n", ret, GetGeError().c_str());
              return FAILED);
    // normalized_shape is a one-element Tensor whose value is 4.
    ADD_INPUT(2, normalized_shape, DT_INT32, normalizedShapeTensorShape, 4);
    ADD_INPUT(3, gamma, DT_FLOAT, parameterShape, 2.0);
    ADD_INPUT(4, beta, DT_FLOAT, parameterShape, 3.0);
    node.set_attr_epsilon(0.5f);

    SET_OUTPUT(y, DT_FLOAT, xShape);
    SET_OUTPUT(mean, DT_FLOAT, statisticShape);
    SET_OUTPUT(rstd, DT_FLOAT, statisticShape);
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
    // mean=2, variance=0.5, epsilon=0.5 -> rstd=1.
    // gamma=2, beta=3 -> y=(x-2)*2+3=[1,3,3,5] per row.
    ok = CheckOutput(outputs[0], "y", {2, 3, 4}, {1.0f, 3.0f, 3.0f, 5.0f}) && ok;
    ok = CheckOutput(outputs[1], "mean", {2, 3, 1}, {2.0f}) && ok;
    ok = CheckOutput(outputs[2], "rstd", {2, 3, 1}, {1.0f}) && ok;
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
    LOG_PRINT("LayerNormV4 graph run success, output count: %zu\n", outputTensors.size());
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

    Graph graph("layer_norm_v4_graph");
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
