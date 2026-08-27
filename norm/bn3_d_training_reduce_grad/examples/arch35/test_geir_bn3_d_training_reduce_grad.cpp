/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* GEIR (Graph Engine IR) example for BN3DTrainingReduceGrad operator */

#include <iostream>
#include <fstream>
#include <memory>
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
#include "array_ops.h"
#include "ge_ir_build.h"
#include "../../op_graph/bn3_d_training_reduce_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::unique_ptr;
using std::vector;

// GEIR 专用宏：添加输入placeholder到图中
#define ADD_INPUT(inputIndex, inputName, inputShape, inputFormat)                                          \
    do {                                                                                                   \
        vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                      \
        auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0);             \
        TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), \
                                                               inputFormat, DT_FLOAT);                     \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                   \
        placeholder##inputIndex##_desc.SetFormat(inputFormat);                                             \
        placeholder##inputIndex##_desc.SetOriginFormat(inputFormat);                                       \
        Tensor tensor_placeholder##inputIndex;                                                             \
        ret = GenOnesDataFloat32(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,          \
                                 placeholder##inputIndex##_desc, inputDataHolder);                         \
        if (ret != SUCCESS) {                                                                              \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                 \
            return FAILED;                                                                                 \
        }                                                                                                  \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                       \
        placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                      \
        input.push_back(tensor_placeholder##inputIndex);                                                   \
        graph.AddOp(placeholder##inputIndex);                                                              \
        bn3_d_training_reduce_grad_op.set_input_##inputName(placeholder##inputIndex);                      \
        inputs.push_back(placeholder##inputIndex);                                                         \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                           vector<unique_ptr<float[]>>& inputDataHolder)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t byteSizeFloat32 = 4;
    uint32_t data_len = size * byteSizeFloat32;
    unique_ptr<float[]> pData(new (std::nothrow) float[size]);
    if (pData == nullptr) {
        return FAILED;
    }

    for (size_t i = 0; i < size; ++i) {
        pData[i] = 1.0;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData.get()), data_len);
    inputDataHolder.push_back(std::move(pData));
    return SUCCESS;
}

int CreateOppInGraph(std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                     Graph& graph, vector<unique_ptr<float[]>>& inputDataHolder)
{
    Status ret = SUCCESS;

    // 创建 BN3DTrainingReduceGrad 算子实例
    auto bn3_d_training_reduce_grad_op = op::BN3DTrainingReduceGrad("bn3_d_training_reduce_grad");

    // 设置属性：epsilon
    bn3_d_training_reduce_grad_op.set_attr_epsilon(0.0001);

    // 必选输入 grads / x：5D 张量 (N, C, D, H, W)
    std::vector<int64_t> shape5 = {2, 3, 4, 5, 6};
    ADD_INPUT(1, grads, shape5, FORMAT_NCDHW);
    ADD_INPUT(2, x, shape5, FORMAT_NCDHW);

    // 必选输入 diff_scale / diff_offset / scale / batch_mean / batch_variance：1D 张量 (C,)
    std::vector<int64_t> shape1 = {3};
    ADD_INPUT(3, diff_scale, shape1, FORMAT_ND);
    ADD_INPUT(4, diff_offset, shape1, FORMAT_ND);
    ADD_INPUT(5, scale, shape1, FORMAT_ND);
    ADD_INPUT(6, batch_mean, shape1, FORMAT_ND);
    ADD_INPUT(7, batch_variance, shape1, FORMAT_ND);

    outputs.push_back(bn3_d_training_reduce_grad_op);
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
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    vector<unique_ptr<float[]>> inputDataHolder;

    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    ret = CreateOppInGraph(input, inputs, outputs, graph, inputDataHolder);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir graph failed\n", GetTime().c_str());
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

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
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

    // 打印输出（y: FLOAT32, shape 与 grads 一致）
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        float* output_data_i = reinterpret_cast<float*>(output[i].GetData());
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        LOG_PRINT("result[0] is: %f\n", output_data_i[0]);
    }

    delete session;
    session = nullptr;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
