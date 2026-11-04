# RmsNormGrad

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

## Function

- Description:

  Computes the backward propagation result of RmsNorm. Based on the gradient `dy` passed in backward propagation, the forward input `x`, the forward intermediate result `rstd`, and the scaling factor `gamma`, it computes the input gradient `dx` and the gradient `dgamma` of the scaling factor.

- Formula:

  For each non-normalized position $p$, flatten the corresponding normalized suffix into $n$ elements, and let $rstd_p=1/\operatorname{Rms}(\mathbf{x}_p)$. Then:

  $$
  m_p=\frac{1}{n}\sum_{q=1}^{n}(dy_{p,q}\cdot \gamma_q\cdot x_{p,q}\cdot rstd_p)
  $$

  $$
  dx_{p,q}=(dy_{p,q}\cdot \gamma_q-rstd_p\cdot x_{p,q}\cdot m_p)\cdot rstd_p
  $$

  $$
  dgamma_q=\sum_p(dy_{p,q}\cdot x_{p,q}\cdot rstd_p)
  $$

  Here, $n$ denotes the number of elements of `gamma`, $q$ denotes the flattened normalization index, and $p$ denotes the non-normalized index that participates in gradient accumulation outside the dimensions corresponding to `gamma`.

  In the following text, let $j=\operatorname{rank}(gamma)$ and $i=\operatorname{rank}(x)-j$, where $j\geq 1$ and $i\geq 0$. The shapes of `x` and `dy` are denoted as [A1,...,Ai,R1,...,Rj], and the shape of `gamma` is denoted as [R1,...,Rj]; when $i=0$, [A1,...,Ai] is an empty sequence.

## Ascend IR Definition

The path of the header file where the Ascend IR is defined: [rms_norm_grad_proto.h](../op_graph/rms_norm_grad_proto.h)

```c++
REG_OP(RmsNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OP_END_FACTORY_REG(RmsNormGrad)
```

## Parameters

| Parameter name | Input/Attribute/Output | Description | Usage instructions | Data type | Data format | Dimensions (shape) |
| --- | --- | --- | --- | --- | --- | --- |
| dy (Tensor) | Required input | Gradient passed in backward propagation, corresponding to $dy$ in the formula. | The shape and data type must be the same as those of `x`; for valid data type combinations, see "Constraints". | float32, float16, bfloat16 | ND | 1-8 dimensions, with the shape [A1,...,Ai,R1,...,Rj] |
| x (Tensor) | Required input | Input of the forward operator, corresponding to $x$ in the formula. | The shape and data type must be the same as those of `dy`. | float32, float16, bfloat16 | ND | Consistent with `dy` |
| rstd (Tensor) | Required input | Reciprocal of the root mean square computed in the forward pass, corresponding to $rstd$ in the formula. | The data type must be float32; for the shape relationship, see "Constraints". | float32 | ND | Based on [A1,...,Ai]; dimensions of size 1 can be removed from or appended to the end |
| gamma (Tensor) | Required input | Scaling factor of the forward normalization computation, corresponding to $\gamma$ in the formula. | For the shape relationship with `x`, see "Constraints"; for valid data type combinations, see "Constraints". | float32, float16, bfloat16 | ND | 1-8 dimensions, with the shape [R1,...,Rj] |
| dx (Tensor) | Required output | Gradient of the input `x`, corresponding to $dx$ in the formula. | The shape is the same as that of `x`, and the data type is the same as that of `dy`. | float32, float16, bfloat16 | ND | Consistent with `x` |
| dgamma (Tensor) | Required output | Gradient of the scaling factor `gamma`, corresponding to $dgamma$ in the formula. | The shape is the same as that of `gamma`, and the data type is fixed to float32. | float32 | ND | Consistent with `gamma` |

- <term>Atlas inference products</term>: `dy`, `x`, `gamma`, and `dx` do not support bfloat16.

## Constraints

- The shapes of `dy` and `x` must be the same. The number of dimensions of `gamma` cannot be greater than that of `x`; if `gamma` has `j` dimensions, the shape of `gamma` must be exactly the same as the last `j` dimensions of `x`, that is, when the shape of `x` is [A1,...,Ai,R1,...,Rj], the shape of `gamma` must be [R1,...,Rj].
- The shape of `rstd` is based on [A1,...,Ai]: zero or more dimensions of size 1 can be removed from the end, or zero or more dimensions of size 1 can be appended to the end; the number of its elements must equal A1×...×Ai, and when $i=0$, this product is counted as 1. In the non-empty scenario, the `rstd` output by RmsNorm satisfies this constraint and can be directly used as input.
- <term>Ascend 950PR&950DT products</term>: dimensions of size 0 in the non-normalized dimensions [A1,...,Ai] are supported. In this case, `dy`, `x`, `rstd`, and `dx` are empty tensors, and `dgamma` is all 0; the size of each dimension of the normalized dimensions [R1,...,Rj] and `gamma` must be greater than 0.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>: empty tensors are not supported, and the size of each dimension of the input tensors must be greater than 0.
- <term>Atlas inference products</term>: the data volume of the normalized dimensions [R1,...,Rj] (R1×...×Rj× the number of bytes of a single element) must be greater than or equal to 32 Bytes.
- The data type combinations supported by each product are as follows:

  - <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:

    | `dy` data type | `x` data type | `rstd` data type | `gamma` data type | `dx` data type | `dgamma` data type |
    | --- | --- | --- | --- | --- | --- |
    | float16 | float16 | float32 | float32 | float16 | float32 |
    | bfloat16 | bfloat16 | float32 | float32 | bfloat16 | float32 |
    | float16 | float16 | float32 | float16 | float16 | float32 |
    | float32 | float32 | float32 | float32 | float32 | float32 |
    | bfloat16 | bfloat16 | float32 | bfloat16 | bfloat16 | float32 |

  - <term>Atlas inference products</term>:

    | `dy` data type | `x` data type | `rstd` data type | `gamma` data type | `dx` data type | `dgamma` data type |
    | --- | --- | --- | --- | --- | --- |
    | float16 | float16 | float32 | float16 | float16 | float32 |
    | float32 | float32 | float32 | float32 | float32 | float32 |

- <term>Ascend 950PR&950DT products</term>: deterministic implementation by default.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>: non-deterministic implementation by default.

## Invocation Sample

The sample code is as follows, for reference only. For the compilation and execution process in GE graph mode, refer to [Operator Invocation](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式).

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
 * \file test_geir_rms_norm_grad.cpp
 * \brief GE graph construction sample for RmsNormGrad.
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
#include "../op_graph/rms_norm_grad_proto.h"

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
    auto node = op::RmsNormGrad("rms_norm_grad");
    vector<int64_t> xShape = {2, 16};
    vector<int64_t> rstdShape = {2, 1};
    vector<int64_t> gammaShape = {16};

    ADD_INPUT(1, dy, xShape, 1.0f);
    ADD_INPUT(2, x, xShape, 1.0f);
    ADD_INPUT(3, rstd, rstdShape, 0.5f);
    ADD_INPUT(4, gamma, gammaShape, 2.0f);

    SET_OUTPUT(dx, xShape);
    SET_OUTPUT(dgamma, gammaShape);
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
    ok = CheckOutput(outputs[0], "dx", {2, 16}, {0.75f}) && ok;
    ok = CheckOutput(outputs[1], "dgamma", {16}, {1.0f}) && ok;
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
    LOG_PRINT("RmsNormGrad graph run success, output count: %zu\n", outputTensors.size());
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

    Graph graph("rms_norm_grad_graph");
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
