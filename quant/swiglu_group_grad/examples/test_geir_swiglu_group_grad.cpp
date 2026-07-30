/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_swiglu_group_grad.cpp
 * @brief GE IR graph-mode call example — SwigluGroupGrad
 *
 * Gradient of ClampedSwiglu activation:
 *   dg = grad_y · silu'(g̃) · ũ · w_t · I(g<c) · m_r
 *   du = grad_y · f · w_t · I(-c<u<c) · m_r
 *   grad_weight = Σ(grad_y · y_origin) along hidden dim
 *
 * Test case: T=8, H=128
 *   grad_y:  (8, 128)  BF16  — required
 *   x:            (8, 256)  BF16  — required
 *   weight:       (8, 1)    FP32  — optional
 *   y_origin:     (8, 128)  BF16  — optional
 *   group_index:  (4,)      INT64 — optional
 *   clamp_limit:  7.0             — attribute
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/swiglu_group_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

const int T = 8;
const int H = 128;

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
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return 2;
    } else if (dt == ge::DT_INT64) {
        return 8;
    } else if (dt == ge::DT_INT32) {
        return 4;
    }
    return 4;
}

int32_t GenInputData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                     float fill_value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t dtypeSize = GetDataTypeSize(data_type);
    uint32_t data_len = size * dtypeSize;
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        return FAILED;
    }
    if (data_type == ge::DT_FLOAT) {
        float val = fill_value;
        for (size_t i = 0; i < size; ++i) {
            memcpy(pData + i * dtypeSize, &val, dtypeSize);
        }
    } else if (data_type == ge::DT_FLOAT16) {
        uint16_t fp16 = 0x3C00; // 1.0 in FP16
        if (fill_value == 0.5f) {
            fp16 = 0x3800; // 0.5 in FP16
        } else if (fill_value == 0.3f) {
            fp16 = 0x34CC; // ~0.3 in FP16
        }
        for (size_t i = 0; i < size; ++i) {
            memcpy(pData + i * dtypeSize, &fp16, dtypeSize);
        }
    } else if (data_type == ge::DT_BF16) {
        uint16_t bf16 = 0x3F80; // 1.0 in BF16
        if (fill_value == 0.5f) {
            bf16 = 0x3F00; // 0.5 in BF16
        } else if (fill_value == 0.3f) {
            bf16 = 0x3E99; // ~0.3 in BF16
        }
        for (size_t i = 0; i < size; ++i) {
            memcpy(pData + i * dtypeSize, &bf16, dtypeSize);
        }
    } else if (data_type == ge::DT_INT64) {
        int64_t val = static_cast<int64_t>(fill_value);
        for (size_t i = 0; i < size; ++i) {
            memcpy(pData + i * dtypeSize, &val, dtypeSize);
        }
    } else if (data_type == ge::DT_INT32) {
        int32_t val = static_cast<int32_t>(fill_value);
        for (size_t i = 0; i < size; ++i) {
            memcpy(pData + i * dtypeSize, &val, dtypeSize);
        }
    } else {
        memset(pData, 0, data_len);
    }
    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;

    auto swigluGroupGradOp = op::SwigluGroupGrad("swiglu_group_grad_1");

    // ── Required input: grad_y (dy), shape (T, H) ──────────────────────
    vector<int64_t> grad_y_shape = {T, H};
    auto data_grad_y = op::Data("data_grad_y").set_attr_index(0);
    TensorDesc grad_y_desc = TensorDesc(ge::Shape(grad_y_shape), FORMAT_ND, inDtype);
    grad_y_desc.SetPlacement(ge::kPlacementHost);
    Tensor tensor_grad_y;
    ret = GenInputData(grad_y_shape, tensor_grad_y, grad_y_desc, inDtype, 1.0f);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate grad_y data failed\n", GetTime().c_str());
        return FAILED;
    }
    data_grad_y.update_input_desc_x(grad_y_desc);
    input.push_back(tensor_grad_y);
    graph.AddOp(data_grad_y);
    swigluGroupGradOp.set_input_grad_y(data_grad_y);
    inputs.push_back(data_grad_y);

    // ── Required input: x, shape (T, 2H) ────────────────────────────────────
    vector<int64_t> x_shape = {T, H * 2};
    auto data_x = op::Data("data_x").set_attr_index(1);
    TensorDesc x_desc = TensorDesc(ge::Shape(x_shape), FORMAT_ND, inDtype);
    x_desc.SetPlacement(ge::kPlacementHost);
    Tensor tensor_x;
    ret = GenInputData(x_shape, tensor_x, x_desc, inDtype, 0.5f);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate x data failed\n", GetTime().c_str());
        return FAILED;
    }
    data_x.update_input_desc_x(x_desc);
    input.push_back(tensor_x);
    graph.AddOp(data_x);
    swigluGroupGradOp.set_input_x(data_x);
    inputs.push_back(data_x);

    // ── Optional input: weight, shape (T, 1), FP32 ──────────────────────────
    vector<int64_t> weight_shape = {T, 1};
    auto data_weight = op::Data("data_weight").set_attr_index(2);
    TensorDesc weight_desc = TensorDesc(ge::Shape(weight_shape), FORMAT_ND, DT_FLOAT);
    weight_desc.SetPlacement(ge::kPlacementHost);
    Tensor tensor_weight;
    ret = GenInputData(weight_shape, tensor_weight, weight_desc, DT_FLOAT, 1.0f);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate weight data failed\n", GetTime().c_str());
        return FAILED;
    }
    data_weight.update_input_desc_x(weight_desc);
    input.push_back(tensor_weight);
    graph.AddOp(data_weight);
    swigluGroupGradOp.set_input_weight(data_weight);
    inputs.push_back(data_weight);

    // ── Optional input: y_origin (yOrigin), shape (T, H), same dtype ────────
    vector<int64_t> y_origin_shape = {T, H};
    auto data_y_origin = op::Data("data_y_origin").set_attr_index(3);
    TensorDesc y_origin_desc = TensorDesc(ge::Shape(y_origin_shape), FORMAT_ND, inDtype);
    y_origin_desc.SetPlacement(ge::kPlacementHost);
    Tensor tensor_y_origin;
    ret = GenInputData(y_origin_shape, tensor_y_origin, y_origin_desc, inDtype, 0.3f);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate y_origin data failed\n", GetTime().c_str());
        return FAILED;
    }
    data_y_origin.update_input_desc_x(y_origin_desc);
    input.push_back(tensor_y_origin);
    graph.AddOp(data_y_origin);
    swigluGroupGradOp.set_input_y_origin(data_y_origin);
    inputs.push_back(data_y_origin);

    // ── Optional input: group_index, shape (4,), INT64 ──────────────────────
    vector<int64_t> group_index_shape = {4};
    auto data_group_index = op::Data("data_group_index").set_attr_index(4);
    TensorDesc group_index_desc = TensorDesc(ge::Shape(group_index_shape), FORMAT_ND, DT_INT64);
    group_index_desc.SetPlacement(ge::kPlacementHost);
    Tensor tensor_group_index;
    ret = GenInputData(group_index_shape, tensor_group_index, group_index_desc, DT_INT64, 2.0f);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate group_index data failed\n", GetTime().c_str());
        return FAILED;
    }
    data_group_index.update_input_desc_x(group_index_desc);
    input.push_back(tensor_group_index);
    graph.AddOp(data_group_index);
    swigluGroupGradOp.set_input_group_index(data_group_index);
    inputs.push_back(data_group_index);

    // ── Required output: grad_x (dx), shape (T, 2H) ─────────────────────────
    TensorDesc grad_x_desc = TensorDesc(ge::Shape(x_shape), FORMAT_ND, inDtype);
    swigluGroupGradOp.update_output_desc_grad_x(grad_x_desc);

    // ── Optional output: grad_weight, shape (T, 1), FP32 ────────────────────
    TensorDesc grad_weight_desc = TensorDesc(ge::Shape(weight_shape), FORMAT_ND, DT_FLOAT);
    swigluGroupGradOp.update_output_desc_grad_weight(grad_weight_desc);

    // ── Attribute: clamp_limit = 7.0 ─────────────────────────────────────────
    swigluGroupGradOp.set_attr_clamp_limit(7.0f);

    outputs.push_back(swigluGroupGradOp);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "swiglu_group_grad_ge_ir_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_BF16;
    if (argc > 1) {
        std::string dtypeArg(argv[1]);
        if (dtypeArg == "fp16") {
            inDtype = DT_FLOAT16;
        } else if (dtypeArg == "fp32") {
            inDtype = DT_FLOAT;
        }
    }
    printf("%s - INFO - [XIR]: Using dtype = %d\n", GetTime().c_str(), static_cast<int>(inDtype));

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create op in graph failed\n", GetTime().c_str());
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
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph to session failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int output_num = output.size();
    printf("%s - INFO - [XIR]: Number of outputs: %d\n", GetTime().c_str(), output_num);
    for (int i = 0; i < output_num; i++) {
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("%s - INFO - [XIR]: output %d shape size = %ld, dtype = %d\n", GetTime().c_str(), i, output_shape,
               static_cast<int>(output[i].GetTensorDesc().GetDataType()));
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    if (!error_str.empty()) {
        std::cout << "Error message: " << error_str << std::endl;
    }

    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
