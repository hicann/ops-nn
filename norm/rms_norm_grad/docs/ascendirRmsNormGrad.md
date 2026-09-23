# RmsNormGrad

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

## 功能说明

- 算子功能：

  计算RmsNorm的反向传播结果。根据反向传入梯度`dy`、正向输入`x`、正向中间结果`rstd`和缩放因子`gamma`，计算输入梯度`dx`以及缩放因子梯度`dgamma`。

- 计算公式：

  对每个非归一化位置$p$，将对应的归一化后缀展平为$n$个元素，并令$rstd_p=1/\operatorname{Rms}(\mathbf{x}_p)$，则：

  $$
  m_p=\frac{1}{n}\sum_{q=1}^{n}(dy_{p,q}\cdot \gamma_q\cdot x_{p,q}\cdot rstd_p)
  $$

  $$
  dx_{p,q}=(dy_{p,q}\cdot \gamma_q-rstd_p\cdot x_{p,q}\cdot m_p)\cdot rstd_p
  $$

  $$
  dgamma_q=\sum_p(dy_{p,q}\cdot x_{p,q}\cdot rstd_p)
  $$

  其中，$n$表示`gamma`的元素个数，$q$表示展平后的归一化索引，$p$表示`gamma`所对应维度之外参与梯度累加的非归一化索引。

  下文令$j=\operatorname{rank}(gamma)$，$i=\operatorname{rank}(x)-j$，其中$j\geq 1$、$i\geq 0$。`x`和`dy`的shape记为[A1,...,Ai,R1,...,Rj]，`gamma`的shape记为[R1,...,Rj]；当$i=0$时，[A1,...,Ai]为空序列。

## Ascend IR定义

Ascend IR定义所在头文件路径：[rms_norm_grad_proto.h](../op_graph/rms_norm_grad_proto.h)

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

## 参数说明

| 参数名 | 输入/属性/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| dy (Tensor) | 必选输入 | 反向传入的梯度，对应公式中的$dy$。 | shape和数据类型必须与`x`相同；有效数据类型组合见“约束说明”。 | float32、float16、bfloat16 | ND | 1-8维，形状为[A1,...,Ai,R1,...,Rj] |
| x (Tensor) | 必选输入 | 正向算子的输入，对应公式中的$x$。 | shape和数据类型必须与`dy`相同。 | float32、float16、bfloat16 | ND | 与`dy`一致 |
| rstd (Tensor) | 必选输入 | 正向计算得到的均方根倒数，对应公式中的$rstd$。 | 数据类型必须为float32；shape关系见“约束说明”。 | float32 | ND | 以[A1,...,Ai]为基准，可在末尾删除或补充大小为1的维度 |
| gamma (Tensor) | 必选输入 | 正向归一化计算的缩放因子，对应公式中的$\gamma$。 | 与`x`的shape关系见“约束说明”；有效数据类型组合见“约束说明”。 | float32、float16、bfloat16 | ND | 1-8维，形状为[R1,...,Rj] |
| dx (Tensor) | 必选输出 | 输入`x`的梯度，对应公式中的$dx$。 | shape与`x`相同，数据类型与`dy`相同。 | float32、float16、bfloat16 | ND | 与`x`一致 |
| dgamma (Tensor) | 必选输出 | 缩放因子`gamma`的梯度，对应公式中的$dgamma$。 | shape与`gamma`相同，数据类型固定为float32。 | float32 | ND | 与`gamma`一致 |

- <term>Atlas推理系列产品</term>：`dy`、`x`、`gamma`和`dx`不支持bfloat16。

## 约束说明

- `dy`与`x`的shape必须相同。`gamma`的维度数不能大于`x`的维度数；若`gamma`为`j`维，则`gamma`的shape必须与`x`的末尾`j`个维度完全相同，即`x`的shape为[A1,...,Ai,R1,...,Rj]时，`gamma`的shape必须为[R1,...,Rj]。
- `rstd`的shape以[A1,...,Ai]为基准：可从末尾删除零个或多个大小为1的维度，或在末尾补充零个或多个大小为1的维度；其元素个数必须等于A1×...×Ai，当$i=0$时该乘积按1计算。非空场景下，RmsNorm输出的`rstd`满足该约束，可直接作为输入。
- <term>Ascend 950PR&950DT系列产品</term>：支持非归一化维[A1,...,Ai]中存在大小为0的维度。此时`dy`、`x`、`rstd`和`dx`为空Tensor，`dgamma`为全0；归一化维[R1,...,Rj]和`gamma`的各维大小必须大于0。
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas推理系列产品</term>：不支持空Tensor，输入Tensor的各维大小必须大于0。
- <term>Atlas推理系列产品</term>：归一化维[R1,...,Rj]的数据量（R1×...×Rj×单个元素字节数）必须大于等于32 Bytes。
- 各产品支持的数据类型组合如下：

  - <term>Ascend 950PR&950DT系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：

    | `dy`数据类型 | `x`数据类型 | `rstd`数据类型 | `gamma`数据类型 | `dx`数据类型 | `dgamma`数据类型 |
    | --- | --- | --- | --- | --- | --- |
    | float16 | float16 | float32 | float32 | float16 | float32 |
    | bfloat16 | bfloat16 | float32 | float32 | bfloat16 | float32 |
    | float16 | float16 | float32 | float16 | float16 | float32 |
    | float32 | float32 | float32 | float32 | float32 | float32 |
    | bfloat16 | bfloat16 | float32 | bfloat16 | bfloat16 | float32 |

  - <term>Atlas推理系列产品</term>：

    | `dy`数据类型 | `x`数据类型 | `rstd`数据类型 | `gamma`数据类型 | `dx`数据类型 | `dgamma`数据类型 |
    | --- | --- | --- | --- | --- | --- |
    | float16 | float16 | float32 | float16 | float16 | float32 |
    | float32 | float32 | float32 | float32 | float32 | float32 |

- <term>Ascend 950PR&950DT系列产品</term>：默认确定性实现。
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas推理系列产品</term>：默认非确定性实现。

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
