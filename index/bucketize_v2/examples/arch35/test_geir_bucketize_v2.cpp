/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"

#include "nn_other.h"
#include "../../op_graph/bucketize_v2_proto.h"

#define FAILED -1
#define SUCCESS 0

#include "graph/operator.h"
#include "graph/operator_reg.h"
namespace ge {

REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
}

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT_TENSOR(inputIndex, inputName, inputDtype, inputShape, value)                                    \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                 \
    auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, \
                                                           inputDtype);                                           \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                              \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                          \
    Tensor tensor_placeholder##inputIndex;                                                                        \
    ret = GenOnesDataFloat(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                       \
                           placeholder##inputIndex##_desc, value);                                                \
    if (ret != SUCCESS) {                                                                                         \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                            \
        return FAILED;                                                                                            \
    }                                                                                                             \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##inputIndex);                                                              \
    graph.AddOp(placeholder##inputIndex);                                                                         \
    add1.set_input_##inputName(placeholder##inputIndex);                                                          \
    inputs.push_back(placeholder##inputIndex)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define ADD_INPUT_ATTR(attrName, attrValue) add1.set_attr_##attrName(attrValue)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

int32_t GenOnesDataFloat(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float* value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }

    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        LOG_PRINT("ERROR: Failed to allocate memory.\n");
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value[i];
    }

    uint32_t data_len = size * sizeof(float);
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto add1 = op::BucketizeV2("BucketizeV2");

    std::vector<int64_t> xShape = {6};
    std::vector<int64_t> boundariesShape = {4};

    float x_data[6] = {-5.0f, 50.0f, 150.0f, 10.0f, 5.0f, 100.0f};
    float boundaries_data[4] = {0.0f, 10.0f, 100.0f, 200.0f};

    ADD_INPUT_TENSOR(1, x, inDtype, xShape, x_data);
    ADD_INPUT_TENSOR(2, boundaries, inDtype, boundariesShape, boundaries_data);
    ADD_OUTPUT(3, y, DT_INT64, xShape);

    ADD_INPUT_ATTR(out_int32, false);
    ADD_INPUT_ATTR(right, false);

    outputs.push_back(add1);
    return SUCCESS;
}

bool InitEnv()
{
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return false;
    }
    return true;
}

bool CreateAndConfigGraph(Graph& graph, std::vector<ge::Tensor>& input)
{
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    Status ret = CreateOppInGraph(DT_FLOAT, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return false;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }
    return true;
}

bool AddGraphToSession(ge::Session* session, Graph& graph, uint32_t graph_id)
{
    std::map<AscendString, AscendString> graph_options = {};

    Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        std::string error_str(error_msg.GetString());
        std::cout << "Error message: " << error_str << std::endl;
        ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
        std::string warning_str(warning_msg.GetString());
        std::cout << "Warning message: " << warning_str << std::endl;
        LOG_PRINT("%s - INFO - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }
    return true;
}

bool DumpAndRunGraph(ge::Session* session, Graph& graph, std::vector<ge::Tensor>& input,
                     std::vector<ge::Tensor>& output, uint32_t graph_id)
{
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    Status ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }
    return true;
}

bool ProcessOutputData(std::vector<ge::Tensor>& output)
{
    int output_num = output.size();
    bool check_passed = true;
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        int64_t* output_data_i = (int64_t*)output[i].GetData();
        int64_t output_size = output[i].GetTensorDesc().GetShape().GetShapeSize();

        int64_t expect_out[6] = {0, 2, 3, 1, 1, 2};
        LOG_PRINT("output[%d] results:\n", i);
        for (int64_t j = 0; j < output_size; j++) {
            LOG_PRINT("  result[%ld]: %ld (expected: %ld)\n", j, output_data_i[j], expect_out[j]);
            if (output_data_i[j] != expect_out[j]) {
                LOG_PRINT("ERROR - [XIR]: output[%d] precision check failed at index %ld\n", i, j);
                check_passed = false;
            }
        }
    }
    if (check_passed) {
        LOG_PRINT("INFO - [XIR]: Precision check passed.\n");
    }
    return check_passed;
}

bool FinalizeRes()
{
    Status ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return false;
    }

    LOG_PRINT("=== Test completed successfully ===\n");
    return true;
}

int main(int argc, char* argv[])
{
    LOG_PRINT("=== BucketizeV2 GEIR Test Start ===\n");
    LOG_PRINT("Input x: [-5, 50, 150, 10, 5, 100]\n");
    LOG_PRINT("Input boundaries: [0, 10, 100, 200] (ascending, non-repeating)\n");
    LOG_PRINT("right=false (default): left-open, right-closed intervals\n");
    LOG_PRINT("  - x <= 0: index 0\n");
    LOG_PRINT("  - (0, 10]: index 1\n");
    LOG_PRINT("  - (10, 100]: index 2\n");
    LOG_PRINT("  - (100, 200]: index 3\n");
    LOG_PRINT("  - x > 200: index 4\n");
    LOG_PRINT("Expected output: [0, 2, 3, 1, 1, 2]\n");
    LOG_PRINT("  -5 <= 0 -> index 0\n");
    LOG_PRINT("  50 in (10, 100] -> index 2\n");
    LOG_PRINT("  150 in (100, 200] -> index 3\n");
    LOG_PRINT("  10 in (0, 10] -> index 1 (boundary value: 0 < 10 <= 10)\n");
    LOG_PRINT("  5 in (0, 10] -> index 1\n");
    LOG_PRINT("  100 in (10, 100] -> index 2 (boundary value: 10 < 100 <= 100)\n");

    if (!InitEnv()) {
        return FAILED;
    }

    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    if (!CreateAndConfigGraph(graph, input)) {
        LOG_PRINT("ERROR: CreateAndConfigGraph failed\n");
        return FAILED;
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        LOG_PRINT("ERROR: Failed to create session\n");
        ge::GEFinalize();
        return FAILED;
    }

    uint32_t graph_id = 0;
    if (!AddGraphToSession(session, graph, graph_id)) {
        LOG_PRINT("ERROR: AddGraphToSession failed\n");
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    if (!DumpAndRunGraph(session, graph, input, output, graph_id)) {
        LOG_PRINT("ERROR: DumpAndRunGraph failed\n");
        return FAILED;
    }

    if (!ProcessOutputData(output)) {
        LOG_PRINT("ERROR: ProcessOutputData failed\n");
        return FAILED;
    }

    if (!FinalizeRes()) {
        LOG_PRINT("ERROR: FinalizeRes failed\n");
        return FAILED;
    }
    return SUCCESS;
}