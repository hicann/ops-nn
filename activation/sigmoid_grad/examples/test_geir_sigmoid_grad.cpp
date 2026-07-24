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
 * \file test_geir_sigmoid_grad.cpp
 * \brief
 */

#include <complex>
#include <ctime>
#include <cstdio>
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

#include "../op_graph/sigmoid_grad_proto.h"

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
    } else if (dt == ge::DT_COMPLEX64) {
        return 8;
    } else if (dt == ge::DT_COMPLEX128) {
        return 16;
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
        case DT_COMPLEX64:
            return "DT_COMPLEX64";
        case DT_COMPLEX128:
            return "DT_COMPLEX128";
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

    size_t data_len = size * sizeof(T);
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
    auto sigmoid_grad = op::SigmoidGrad("sigmoid_grad");

    std::vector<int64_t> y_shape = {2, 3};
    std::vector<std::complex<double>> y_data = {{1.0, 2.0}, {0.5, -1.0}, {2.0, 0.0},
                                                {0.3, 0.4}, {1.5, -0.5}, {0.8, 1.2}};
    std::vector<std::complex<double>> dy_data = {{0.5, 0.5}, {1.0, -1.0}, {2.0, 1.0},
                                                 {0.3, 0.1}, {1.0, 0.5},  {0.2, -0.3}};

    auto placeholder1 = op::Data("placeholder1").set_attr_index(0);
    TensorDesc desc1 = TensorDesc(ge::Shape(y_shape), FORMAT_ND, input_dtype);
    desc1.SetPlacement(ge::kPlacementHost);
    desc1.SetFormat(FORMAT_ND);
    Tensor tensor1;
    ret = GenTensorData(y_shape, tensor1, desc1, y_data);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate y data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder1.update_input_desc_x(desc1);
    placeholder1.update_output_desc_y(desc1);
    input.push_back(tensor1);
    graph.AddOp(placeholder1);
    sigmoid_grad.set_input_y(placeholder1);
    inputs.push_back(placeholder1);

    auto placeholder2 = op::Data("placeholder2").set_attr_index(1);
    TensorDesc desc2 = TensorDesc(ge::Shape(y_shape), FORMAT_ND, input_dtype);
    desc2.SetPlacement(ge::kPlacementHost);
    desc2.SetFormat(FORMAT_ND);
    Tensor tensor2;
    ret = GenTensorData(y_shape, tensor2, desc2, dy_data);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate dy data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder2.update_input_desc_x(desc2);
    placeholder2.update_output_desc_y(desc2);
    input.push_back(tensor2);
    graph.AddOp(placeholder2);
    sigmoid_grad.set_input_dy(placeholder2);
    inputs.push_back(placeholder2);

    TensorDesc output_desc = TensorDesc(ge::Shape(y_shape), FORMAT_ND, input_dtype);
    output_desc.SetPlacement(ge::kPlacementHost);
    output_desc.SetFormat(FORMAT_ND);
    sigmoid_grad.update_output_desc_z(output_desc);

    outputs.push_back(sigmoid_grad);
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
        printf("%s - INFO - [XIR]: Initialize ge failed.ret = %d\n", GetTime().c_str(), ret);
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    DataType input_dtype = DT_COMPLEX128;

    ret = CreateOppInGraph(input_dtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Start to run graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: RunGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: RunGraph success\n", GetTime().c_str());

    if (!output.empty()) {
        uint8_t* output_data = output[0].GetData();
        int64_t output_shape_size = output[0].GetTensorDesc().GetShape().GetShapeSize();
        DataType output_dtype = output[0].GetTensorDesc().GetDataType();
        printf("%s - INFO - [XIR]: Output dtype: %s, shape size: %ld\n", GetTime().c_str(),
               DataTypeToString(output_dtype), output_shape_size);
        std::complex<double>* result = reinterpret_cast<std::complex<double>*>(output_data);

        std::complex<double> y_vals[6] = {{1.0, 2.0}, {0.5, -1.0}, {2.0, 0.0}, {0.3, 0.4}, {1.5, -0.5}, {0.8, 1.2}};
        std::complex<double> dy_vals[6] = {{0.5, 0.5}, {1.0, -1.0}, {2.0, 1.0}, {0.3, 0.1}, {1.0, 0.5}, {0.2, -0.3}};
        bool match = true;
        for (int64_t i = 0; i < output_shape_size; ++i) {
            std::complex<double> expected = dy_vals[i] * (std::complex<double>(1.0) - y_vals[i]) * y_vals[i];
            printf("result[%ld] = (%.6f, %.6f), expected = (%.6f, %.6f)\n", i, result[i].real(), result[i].imag(),
                   expected.real(), expected.imag());
            if (std::abs(result[i] - expected) > 1e-6) {
                printf("MISMATCH at[%ld]\n", i);
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

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());

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
