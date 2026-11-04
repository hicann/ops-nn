# RmsNorm

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

  Performs root mean square normalization on several trailing dimensions of the input tensor corresponding to `gamma`, and scales the normalization result with `gamma`. Compared with LayerNorm, RmsNorm does not perform the mean-subtraction operation and is commonly used for normalization computation of large models. The operator also outputs the reciprocal of the root mean square `rstd`, which can be used for backward computation.

- Formula:

  For each non-normalized position $p$, flatten the corresponding normalized suffix into $n$ elements. Then:

  $$
  rstd_p=\frac{1}{\sqrt{\frac{1}{n}\sum_{q=1}^{n}x_{p,q}^2+\epsilon}}
  $$

  $$
  y_{p,q}=x_{p,q}\cdot rstd_p\cdot \gamma_q
  $$

  where $n$ denotes the number of elements in `gamma`, $q$ denotes the flattened normalization index, $p$ denotes the non-normalized index outside the dimensions corresponding to `gamma`, and $\epsilon$ denotes the number added to the mean square value to improve numerical stability.

  In the following text, let $j=\operatorname{rank}(gamma)$ and $i=\operatorname{rank}(x)-j$, where $j\geq 1$ and $i\geq 0$. The shape of `x` is denoted as [A1,...,Ai,R1,...,Rj], and the shape of `gamma` is denoted as [R1,...,Rj]. When $i=0$, [A1,...,Ai] is an empty sequence.

## Ascend IR Definition

The header file containing the Ascend IR definition is located at: [rms_norm_proto.h](../op_graph/rms_norm_proto.h)

```c++
REG_OP(RmsNorm)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(RmsNorm)
```

## Parameters

| Parameter name | Input/Attribute/Output | Description | Usage instructions | Data type | Data format | Dimensions (shape) |
| --- | --- | --- | --- | --- | --- | --- |
| x (Tensor) | Mandatory input | Input tensor to be normalized, corresponding to $x$ in the formula. | For valid data type combinations, see "Constraints". | float32, float16, bfloat16 | ND | 1-8 dimensions, with the shape [A1,...,Ai,R1,...,Rj] |
| gamma (Tensor) | Mandatory input | Scaling factor of the normalization computation, corresponding to $\gamma$ in the formula. | For the shape relationship with `x`, see "Constraints"; for valid data type combinations, see "Constraints". | float32, float16, bfloat16 | ND | 1-8 dimensions, with the shape [R1,...,Rj] |
| epsilon (float) | Optional attribute | Number added to the mean square value, corresponding to $\epsilon$ in the formula. | The value cannot be less than 0. The default value is 1e-6. | - | - | - |
| y (Tensor) | Mandatory output | Result after normalization and scaling, corresponding to $y$ in the formula. | The shape and data type are the same as those of `x`. | float32, float16, bfloat16 | ND | Same as `x` |
| rstd (Tensor) | Mandatory output | Reciprocal of the root mean square, corresponding to $rstd$ in the formula. | The data type is fixed to float32; the sizes of all trailing dimensions corresponding to `gamma` are 1, and the other dimensions are the same as those of `x`. | float32 | ND | Same number of dimensions as `x`, with the shape [A1,...,Ai,1,...,1] and a total of `j` trailing 1s |

The parameter differences among products are described as follows:

- <term>Ascend 950PR&950DT products</term>: `x` and `gamma` allow dimension sizes of 0. When the product of the sizes of the normalized suffix dimensions is 0 and the product of the sizes of the non-normalized dimensions is greater than 0, `y` is an empty tensor and `rstd` is non-empty with all elements being NaN. When the product of the sizes of the non-normalized dimensions is 0, both `y` and `rstd` are empty tensors. If non-normalized dimensions do not exist, the product of their dimension sizes is counted as 1. Empty dimensions do not change the suffix matching requirement between `gamma` and `x`.
- Except for the preceding <term>Ascend 950PR&950DT products</term>, all dimension sizes of `x` and `gamma` must be greater than 0, and empty tensors are not supported.
- <term>Atlas inference products</term>, <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: `x`, `gamma`, and `y` do not support bfloat16.

The preceding constraints apply to the Ascend IR interface. For the parameter constraints of the aclnn interface, see [aclnnRmsNorm](aclnnRmsNorm_en.md).

## Constraints

- The number of dimensions of `gamma` cannot be greater than the number of dimensions of `x`. If `gamma` has `j` dimensions, the shape of `gamma` must be exactly the same as the trailing `j` dimensions of `x`. That is, when the shape of `x` is [A1,...,Ai,R1,...,Rj], the shape of `gamma` must be [R1,...,Rj].
- <term>Atlas inference products</term>: the data volume of the normalized dimensions [R1,...,Rj] (R1×...×Rj×the number of bytes of a single element) must be greater than or equal to 32 Bytes.
- The data type combinations supported by each product are as follows:

  - <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:

    | `x` data type | `gamma` data type | `y` data type | `rstd` data type |
    | --- | --- | --- | --- |
    | float16 | float32 | float16 | float32 |
    | bfloat16 | float32 | bfloat16 | float32 |
    | float16 | float16 | float16 | float32 |
    | bfloat16 | bfloat16 | bfloat16 | float32 |
    | float32 | float32 | float32 | float32 |

  - <term>Atlas inference products</term>, <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>:

    | `x` data type | `gamma` data type | `y` data type | `rstd` data type |
    | --- | --- | --- | --- |
    | float16 | float16 | float16 | float32 |
    | float32 | float32 | float32 | float32 |

- Default deterministic implementation.

## Invocation Sample

The sample code is for reference only. For the compilation and execution process in GE graph mode, refer to [Operator Invocation](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式).

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
