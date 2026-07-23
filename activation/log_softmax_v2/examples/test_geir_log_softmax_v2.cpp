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
 * \file test_geir_log_softmax_v2.cpp
 * \brief
 */

#include <cmath>
#include <ctime>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <stdint.h>
#include <string>
#include <string.h>
#include <vector>

#include "assert.h"

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/log_softmax_v2_proto.cpp"

#define FAILED -1
#define SUCCESS 0

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge

using namespace ge;
using std::map;
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

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT) {
        return 4;
    } else if (dt == ge::DT_FLOAT16) {
        return 2;
    } else if (dt == ge::DT_DOUBLE) {
        return 8;
    } else if (dt == ge::DT_BF16) {
        return 2;
    } else if (dt == ge::DT_INT32) {
        return 4;
    } else if (dt == ge::DT_INT64) {
        return 8;
    }
    return 0;
}

const char* DataTypeToString(DataType dt)
{
    switch (dt) {
        case DT_FLOAT:
            return "DT_FLOAT";
        case DT_FLOAT16:
            return "DT_FLOAT16";
        case DT_DOUBLE:
            return "DT_DOUBLE";
        case DT_BF16:
            return "DT_BF16";
        case DT_INT32:
            return "DT_INT32";
        case DT_INT64:
            return "DT_INT64";
        default:
            return "DTYPE(unknown)";
    }
}

template <typename T>
int32_t GenTensorData(const vector<int64_t>& shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                      const vector<T>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    if (size != values.size()) {
        return FAILED;
    }

    uint32_t data_len = size * sizeof(T);
    T* p_data = new (std::nothrow) T[size];
    if (p_data == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        p_data[i] = values[i];
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(p_data), data_len);
    delete[] p_data;
    return SUCCESS;
}

int CreateOppInGraph(DataType input_dtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto log_softmax_v2 = op::LogSoftmaxV2("log_softmax_v2");

    std::vector<int64_t> logits_shape = {2, 4};
    std::vector<double> logits_data = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};

    vector<int64_t> placeholder1_shape = logits_shape;
    auto placeholder1 = op::Data("placeholder1").set_attr_index(0);
    TensorDesc placeholder1_desc = TensorDesc(ge::Shape(placeholder1_shape), FORMAT_ND, input_dtype);
    placeholder1_desc.SetPlacement(ge::kPlacementHost);
    placeholder1_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder1;
    ret = GenTensorData(placeholder1_shape, tensor_placeholder1, placeholder1_desc, logits_data);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder1.update_input_desc_x(placeholder1_desc);
    placeholder1.update_output_desc_y(placeholder1_desc);
    input.push_back(tensor_placeholder1);
    graph.AddOp(placeholder1);
    log_softmax_v2.set_input_logits(placeholder1);
    inputs.push_back(placeholder1);

    TensorDesc output_desc = TensorDesc(ge::Shape(logits_shape), FORMAT_ND, input_dtype);
    output_desc.SetPlacement(ge::kPlacementHost);
    output_desc.SetFormat(FORMAT_ND);
    log_softmax_v2.update_output_desc_logsoftmax(output_desc);

    outputs.push_back(log_softmax_v2);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed.ret = %d\n", GetTime().c_str(), ret);
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    DataType input_dtype = DT_DOUBLE;

    ret = CreateOppInGraph(input_dtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph failed\n", GetTime().c_str());
        delete session;
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
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    if (!output.empty()) {
        uint8_t* output_data = output[0].GetData();
        int64_t output_shape_size = output[0].GetTensorDesc().GetShape().GetShapeSize();
        DataType output_dtype = output[0].GetTensorDesc().GetDataType();
        printf("%s - INFO - [XIR]: Output dtype: %s, shape size: %ld\n", GetTime().c_str(),
               DataTypeToString(output_dtype), output_shape_size);
        double* result = reinterpret_cast<double*>(output_data);
        for (int64_t i = 0; i < output_shape_size; ++i) {
            printf("result[%ld] = %lf\n", i, result[i]);
        }

        double expected[8];
        double input_vals[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
        for (int row = 0; row < 2; ++row) {
            double max_val = input_vals[row * 4];
            for (int j = 1; j < 4; ++j) {
                max_val = std::max(max_val, input_vals[row * 4 + j]);
            }
            double sum = 0.0;
            for (int j = 0; j < 4; ++j) {
                sum += std::exp(input_vals[row * 4 + j] - max_val);
            }
            double log_sum = std::log(sum);
            for (int j = 0; j < 4; ++j) {
                expected[row * 4 + j] = input_vals[row * 4 + j] - max_val - log_sum;
            }
        }
        bool match = true;
        for (int64_t i = 0; i < output_shape_size; ++i) {
            if (std::fabs(result[i] - expected[i]) > 1e-6) {
                printf("MISMATCH at[%ld]: got=%lf, expected=%lf\n", i, result[i], expected[i]);
                match = false;
            }
        }
        if (match) {
            printf("%s - INFO - [XIR]: Output verification PASSED\n", GetTime().c_str());
        } else {
            printf("%s - ERROR - [XIR]: Output verification FAILED\n", GetTime().c_str());
        }
    }

    delete session;

    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());

    ge::AscendString err_msg_asc = ge::GEGetErrorMsgV2();
    std::string err_msg = err_msg_asc.GetString();
    if (!err_msg.empty()) {
        printf("Error message: %s\n", err_msg.c_str());
    }
    ge::AscendString warn_msg_asc = ge::GEGetWarningMsgV2();
    std::string warn_msg = warn_msg_asc.GetString();
    if (!warn_msg.empty()) {
        printf("Warning message: %s\n", warn_msg.c_str());
    }

    return SUCCESS;
}
