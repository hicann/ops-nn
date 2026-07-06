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
 * @file test_geir_apply_proximal_adagrad.cpp
 * @brief GE IR 图模式调用示例 - ApplyProximalAdagrad
 *
 * 数学公式 (l1 = 0 时):
 *   accum_t = accum_{t-1} + grad_t^2
 *   eta_t   = lr / sqrt(accum_t)
 *   prox_t  = var_{t-1} - eta_t * grad_t
 *   var_t   = prox_t / (1 + eta_t * l2)
 *
 * 通过 GE Session 构建计算图，编译并执行 ApplyProximalAdagrad 算子。
 * 算子有 6 个输入 (var, accum, lr, l1, l2, grad) 和 2 个 inplace 输出 (var, accum)。
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

#include "../../op_graph/apply_proximal_adagrad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                  \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                 \
    auto placeholder##inputIndex = op::Data("placeholder" #inputIndex).set_attr_index(inputIndex - 1);            \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, \
                                                           inputDtype);                                           \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                              \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                          \
    Tensor tensor_placeholder##inputIndex;                                                                        \
    ret = GenInputData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                           \
                       placeholder##inputIndex##_desc, inputDtype, inputIndex);                                   \
    if (ret != SUCCESS) {                                                                                         \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                            \
        return FAILED;                                                                                            \
    }                                                                                                             \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##inputIndex);                                                              \
    graph.AddOp(placeholder##inputIndex);                                                                         \
    applyProximalAdagradOp.set_input_##inputName(placeholder##inputIndex);                                        \
    inputs.push_back(placeholder##inputIndex);

#define ADD_OUTPUT(outputName, outputDtype, outputShape)                                           \
    TensorDesc outputName##_out_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    applyProximalAdagradOp.update_output_desc_##outputName(outputName##_out_desc);

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
    }
    return 4;
}

int32_t GenInputData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                     int inputIndex)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return FAILED;
    }

    // inputIndex 与 ApplyProximalAdagrad 的输入顺序一一对应:
    //   1=var, 2=accum, 3=lr, 4=l1, 5=l2, 6=grad
    switch (inputIndex) {
        case 1: // var: 全 0 起步
            for (size_t i = 0; i < size; ++i) {
                pData[i] = 0.0f;
            }
            break;
        case 2: // accum: 全 1, 避开 rsqrt(0) 的奇点
            for (size_t i = 0; i < size; ++i) {
                pData[i] = 1.0f;
            }
            break;
        case 3: // lr: 0.1
            for (size_t i = 0; i < size; ++i) {
                pData[i] = 0.1f;
            }
            break;
        case 4: // l1: 0.0 (走 fast-path)
            for (size_t i = 0; i < size; ++i) {
                pData[i] = 0.0f;
            }
            break;
        case 5: // l2: 0.0
            for (size_t i = 0; i < size; ++i) {
                pData[i] = 0.0f;
            }
            break;
        case 6: // grad: 0.1 常数, 期望输出 = -0.1 * 0.1 / sqrt(1.01) ≈ -0.00995037
        default:
            for (size_t i = 0; i < size; ++i) {
                pData[i] = 0.1f;
            }
            break;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
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
    auto applyProximalAdagradOp = op::ApplyProximalAdagrad("apply_proximal_adagrad_1");
    std::vector<int64_t> mainShape = {4, 4, 4};
    std::vector<int64_t> scalarShape = {1};

    // 添加输入: var (1), accum (2), lr (3), l1 (4), l2 (5), grad (6)
    ADD_INPUT(1, var, inDtype, mainShape);
    ADD_INPUT(2, accum, inDtype, mainShape);
    ADD_INPUT(3, lr, inDtype, scalarShape);
    ADD_INPUT(4, l1, inDtype, scalarShape);
    ADD_INPUT(5, l2, inDtype, scalarShape);
    ADD_INPUT(6, grad, inDtype, mainShape);

    // 添加输出: 仅 var (与 proto.h REG_OP 一致)。accum 通过 input ref 原地更新,
    // 不在输出端口暴露。
    ADD_OUTPUT(var, inDtype, mainShape);

    outputs.push_back(applyProximalAdagradOp);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "apply_proximal_adagrad_ge_ir_test";
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

    DataType inDtype = DT_FLOAT;
    if (argc > 1) {
        std::string dtypeArg(argv[1]);
        if (dtypeArg == "fp16") {
            inDtype = DT_FLOAT16;
        } else if (dtypeArg == "bf16") {
            inDtype = DT_BF16;
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

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
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

    // 验证输出: 公式期望 (lr=0.1, accum0=1.0, grad=0.1, l1=l2=0)
    //   accum1 = 1.0 + 0.01 = 1.01
    //   eta    = 0.1 / sqrt(1.01)            ≈ 0.099504
    //   prox   = 0 - eta * 0.1               ≈ -0.0099504
    //   var1   = prox / (1 + eta*0)          ≈ -0.0099504
    int output_num = output.size();
    printf("%s - INFO - [XIR]: Number of outputs: %d\n", GetTime().c_str(), output_num);
    const float expectedVar = -0.1f * 0.1f / std::sqrt(1.01f);
    for (int i = 0; i < output_num; i++) {
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("%s - INFO - [XIR]: output %d shape size = %ld, dtype = %d\n", GetTime().c_str(), i, output_shape,
               static_cast<int>(output[i].GetTensorDesc().GetDataType()));

        if (output[i].GetTensorDesc().GetDataType() == DT_FLOAT && output_shape > 0) {
            float* outData = reinterpret_cast<float*>(output[i].GetData());
            int64_t printNum = (output_shape < 10) ? output_shape : 10;
            for (int64_t j = 0; j < printNum; j++) {
                printf("  var[%ld] = %.6f (expected = %.6f)\n", j, outData[j], expectedVar);
            }
        }
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
