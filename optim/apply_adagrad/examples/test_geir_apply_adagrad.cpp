/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include <string>
#include <map>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/apply_adagrad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape)                                                 \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, 2.0f);                                          \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                  \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    applyAdagrad1.set_input_##intputName(placeholder##intputIndex);                                                 \
    inputs.push_back(placeholder##intputIndex)

#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape)                                           \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Const("placeholder" + intputIndex);                                         \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, 2.0f);                                          \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);                                     \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    applyAdagrad1.set_input_##intputName(placeholder##intputIndex);                                                 \
    applyAdagrad1.update_input_desc_##intputName(placeholder##intputIndex##_desc);                                  \
    inputs.push_back(placeholder##intputIndex);

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {
std::vector<std::unique_ptr<uint8_t[]>> g_input_holder;
}

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT) {
        return sizeof(float);
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return sizeof(uint16_t);
    }
    if (dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return sizeof(uint32_t);
    }
    if (dt == ge::DT_INT64 || dt == ge::DT_UINT64) {
        return sizeof(uint64_t);
    }
    if (dt == ge::DT_INT8) {
        return sizeof(uint8_t);
    }
    return 0;
}

bool GetShapeElementNum(const vector<int64_t>& shapes, size_t& element_num)
{
    element_num = 1;
    for (auto dim : shapes) {
        if (dim < 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / element_num) {
            return false;
        }
        element_num *= static_cast<size_t>(dim);
    }
    return true;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t element_num = 0;
    uint32_t dtype_size = GetDataTypeSize(data_type);
    if (dtype_size == 0 || !GetShapeElementNum(shapes, element_num) ||
        element_num > std::numeric_limits<size_t>::max() / dtype_size) {
        return FAILED;
    }

    size_t data_len = element_num * dtype_size;
    std::unique_ptr<uint8_t[]> data(new (std::nothrow) uint8_t[data_len]);
    if (data == nullptr) {
        return FAILED;
    }

    if (data_type == ge::DT_FLOAT) {
        for (size_t i = 0; i < element_num; ++i) {
            std::memcpy(data.get() + i * dtype_size, &value, dtype_size);
        }
    } else if (data_type == ge::DT_FLOAT16 || data_type == ge::DT_BF16) {
        constexpr uint16_t two_in_fp16_or_bf16 = 0x4000;
        for (size_t i = 0; i < element_num; ++i) {
            std::memcpy(data.get() + i * dtype_size, &two_in_fp16_or_bf16, dtype_size);
        }
    } else {
        std::memset(data.get(), static_cast<int>(value), data_len);
    }

    input_tensor = Tensor(input_tensor_desc, data.get(), data_len);
    g_input_holder.emplace_back(std::move(data));
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    if (inputData == nullptr) {
        return FAILED;
    }
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("WriteDataToFile: fopen failed for %s\n", bin_file.c_str());
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    if (written != data_size) {
        printf("WriteDataToFile: short write %zu/%lu\n", written, data_size);
        fclose(fp);
        return FAILED;
    }
    if (fclose(fp) != 0) {
        printf("WriteDataToFile: fclose failed for %s\n", bin_file.c_str());
        return FAILED;
    }
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto applyAdagrad1 = op::ApplyAdagrad("applyAdagrad1");
    std::vector<int64_t> xShape = {1};
    ADD_INPUT(1, var, inDtype, xShape);
    ADD_INPUT(2, accum, inDtype, xShape);
    ADD_INPUT(3, lr, inDtype, xShape);
    ADD_INPUT(4, grad, inDtype, xShape);

    outputs.push_back(applyAdagrad1);
    // 添加完毕
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_BF16;

    std::cout << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {

    };
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    std::unique_ptr<ge::Session> session(new (std::nothrow) Session(build_options));

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t dtype_size = GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (input_shape < 0 || dtype_size == 0 ||
            static_cast<uint64_t>(input_shape) > std::numeric_limits<uint32_t>::max() / dtype_size) {
            GEFinalize();
            return FAILED;
        }
        uint32_t data_size = static_cast<uint32_t>(input_shape) * dtype_size;
        if (WriteDataToFile(input_file, data_size, input_data_i) != SUCCESS) {
            GEFinalize();
            return FAILED;
        }
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t dtype_size = GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (output_shape < 0 || dtype_size == 0 ||
            static_cast<uint64_t>(output_shape) > std::numeric_limits<uint32_t>::max() / dtype_size) {
            GEFinalize();
            return FAILED;
        }
        uint32_t data_size = static_cast<uint32_t>(output_shape) * dtype_size;
        if (WriteDataToFile(output_file, data_size, output_data_i) != SUCCESS) {
            GEFinalize();
            return FAILED;
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    session.reset();
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
