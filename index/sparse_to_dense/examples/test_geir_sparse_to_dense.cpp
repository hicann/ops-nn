/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <ctime>
#include <map>
#include <new>
#include <stdint.h>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "nn_other.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/sparse_to_dense_proto.h"

#define FAILED -1
#define SUCCESS 0

namespace ge {
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data) REG_OP(Const)
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(value, Tensor)
    .OP_END_FACTORY_REG(Const)
} // namespace ge

using namespace ge;
using std::string;
using std::vector;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

template <typename T>
int32_t GenTensorData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, const vector<T>& values)
{
    desc.SetRealDimCnt(shape.size());
    size_t elemNum = shape.empty() ? 1 : 1;
    for (auto dim : shape) {
        elemNum *= static_cast<size_t>(dim);
    }
    if (elemNum != values.size()) {
        return FAILED;
    }
    auto* data = new (std::nothrow) T[elemNum];
    if (data == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < elemNum; ++i) {
        data[i] = values[i];
    }
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(data), elemNum * sizeof(T));
    return SUCCESS;
}

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape, inputData, inputType)                            \
    do {                                                                                                          \
        auto placeholder = op::Data("placeholder" + std::to_string(inputIndex)).set_attr_index((inputIndex) - 1); \
        TensorDesc desc(ge::Shape(inputShape), FORMAT_ND, inputDtype);                                            \
        desc.SetPlacement(ge::kPlacementHost);                                                                    \
        desc.SetFormat(FORMAT_ND);                                                                                \
        Tensor tensor;                                                                                            \
        ret = GenTensorData<inputType>(inputShape, tensor, desc, inputData);                                      \
        if (ret != SUCCESS) {                                                                                     \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                        \
            return FAILED;                                                                                        \
        }                                                                                                         \
        placeholder.update_input_desc_x(desc);                                                                    \
        placeholder.update_output_desc_y(desc);                                                                   \
        input.push_back(tensor);                                                                                  \
        graph.AddOp(placeholder);                                                                                 \
        sparseToDense.set_input_##inputName(placeholder);                                                         \
        inputs.push_back(placeholder);                                                                            \
    } while (0)

#define ADD_CONST_INPUT(inputIndex, inputName, inputDtype, inputShape, inputData, inputType) \
    do {                                                                                     \
        auto constOp = op::Const("const" + std::to_string(inputIndex));                      \
        TensorDesc desc(ge::Shape(inputShape), FORMAT_ND, inputDtype);                       \
        desc.SetPlacement(ge::kPlacementHost);                                               \
        desc.SetFormat(FORMAT_ND);                                                           \
        Tensor tensor;                                                                       \
        ret = GenTensorData<inputType>(inputShape, tensor, desc, inputData);                 \
        if (ret != SUCCESS) {                                                                \
            printf("%s - ERROR - [XIR]: Generate const data failed\n", GetTime().c_str());   \
            return FAILED;                                                                   \
        }                                                                                    \
        constOp.SetAttr("value", tensor);                                                    \
        constOp.update_output_desc_y(desc);                                                  \
        graph.AddOp(constOp);                                                                \
        sparseToDense.set_input_##inputName(constOp);                                        \
        sparseToDense.update_input_desc_##inputName(desc);                                   \
    } while (0)

int32_t CreateOppInGraph(vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto sparseToDense = op::SparseToDense("SparseToDense");

    vector<int64_t> indicesShape = {3, 2};
    vector<int64_t> outputShapeShape = {2};
    vector<int64_t> valuesShape = {3};
    vector<int64_t> defaultValueShape = {};
    vector<int64_t> yShape = {3, 2};

    vector<int64_t> indicesData = {0, 1, 1, 0, 2, 1};
    vector<int64_t> outputShapeData = {3, 2};
    vector<double> valuesData = {1.0, 2.0, 3.0};
    vector<double> defaultValueData = {9.0};

    ADD_INPUT(1, indices, DT_INT64, indicesShape, indicesData, int64_t);
    ADD_CONST_INPUT(2, output_shape, DT_INT64, outputShapeShape, outputShapeData, int64_t);
    ADD_INPUT(3, values, DT_DOUBLE, valuesShape, valuesData, double);
    ADD_INPUT(4, default_value, DT_DOUBLE, defaultValueShape, defaultValueData, double);
    sparseToDense.set_attr_validate_indices(true);
    sparseToDense.update_output_desc_y(TensorDesc(ge::Shape(yShape), FORMAT_ND, DT_DOUBLE));
    outputs.push_back(sparseToDense);
    return SUCCESS;
}

int32_t CheckOutput(const vector<ge::Tensor>& output)
{
    if (output.size() != 1) {
        printf("%s - ERROR - [XIR]: Output size is invalid, output num: %zu\n", GetTime().c_str(), output.size());
        return FAILED;
    }
    const double expect[] = {9.0, 1.0, 2.0, 9.0, 9.0, 3.0};
    const size_t expectSize = sizeof(expect);
    if (output[0].GetSize() != expectSize) {
        printf("%s - ERROR - [XIR]: Output data size is invalid, data size: %zu\n", GetTime().c_str(),
               output[0].GetSize());
        return FAILED;
    }
    const auto* actual = reinterpret_cast<const double*>(output[0].GetData());
    for (size_t i = 0; i < sizeof(expect) / sizeof(expect[0]); ++i) {
        if (actual[i] != expect[i]) {
            printf("%s - ERROR - [XIR]: Output data is invalid, index: %zu, actual: %lf, expect: %lf\n",
                   GetTime().c_str(), i, actual[i], expect[i]);
            return FAILED;
        }
    }
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    Graph graph("tc_ge_irrun_test_sparse_to_dense");
    vector<ge::Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;

    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }

    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        GEFinalize();
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> sessionOptions = {};
    Session session(sessionOptions);
    std::map<AscendString, AscendString> graphOptions = {};
    ret = session.AddGraph(0, graph, graphOptions);
    if (ret != SUCCESS) {
        ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
        printf("%s - ERROR - [XIR]: Add graph failed: %s\n", GetTime().c_str(), errorMsg.GetString());
        GEFinalize();
        return FAILED;
    }

    vector<ge::Tensor> output;
    ret = session.RunGraph(0, input, output);
    if (ret != SUCCESS) {
        ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
        printf("%s - ERROR - [XIR]: Run graph failed: %s\n", GetTime().c_str(), errorMsg.GetString());
        GEFinalize();
        return FAILED;
    }
    ret = CheckOutput(output);
    if (ret != SUCCESS) {
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run SparseToDense graph success, output num: %zu\n", GetTime().c_str(), output.size());
    return GEFinalize() == SUCCESS ? SUCCESS : FAILED;
}
