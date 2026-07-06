/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* GEIR (Graph Engine IR) example for RmsNormDynamicQuant operator */

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
#include "../op_graph/rms_norm_dynamic_quant_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
using std::unique_ptr;

// GEIR 专用宏：添加输入placeholder到图中
#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                      \
    do {                                                                                                              \
        vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                 \
        auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0);                        \
        TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, \
                                                               inputDtype);                                           \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                              \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                          \
        Tensor tensor_placeholder##inputIndex;                                                                        \
        ret = GenOnesDataFloat32(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                     \
                                 placeholder##inputIndex##_desc, 2, inputDataHolder);                                \
        if (ret != SUCCESS) {                                                                                         \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                            \
            return FAILED;                                                                                            \
        }                                                                                                             \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                  \
        placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                                 \
        input.push_back(tensor_placeholder##inputIndex);                                                              \
        graph.AddOp(placeholder##inputIndex);                                                                         \
        rms_norm_dynamic_quant_op.set_input_##inputName(placeholder##inputIndex);                                     \
        inputs.push_back(placeholder##inputIndex);                                                                    \
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

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        return fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        return twoByte;
    } else if (dt == ge::DT_BF16) {
        return twoByte;
    } else if (dt == ge::DT_INT32) {
        return fourByte;
    } else if (dt == ge::DT_INT64) {
        return eightByte;
    } else if (dt == ge::DT_INT8) {
        return oneByte;
    }
    return fourByte;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value,
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
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData.get()), data_len);
    inputDataHolder.push_back(std::move(pData));
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    if (written != data_size) {
        return FAILED;
    }
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph, vector<unique_ptr<float[]>>& inputDataHolder)
{
    Status ret = SUCCESS;

    // 创建 RmsNormDynamicQuant 算子实例
    auto rms_norm_dynamic_quant_op = op::RmsNormDynamicQuant("rms_norm_dynamic_quant");

    // 设置属性：epsilon 和 dst_type
    rms_norm_dynamic_quant_op.set_attr_epsilon(1e-6);
    rms_norm_dynamic_quant_op.set_attr_dst_type(DT_INT8);

    // 必选输入 x: 2D 张量 (rows, hidden)，例如 (4, 64)
    std::vector<int64_t> xShape = {4, 64};
    ADD_INPUT(1, x, inDtype, xShape);

    // 必选输入 gamma: 1D 张量，大小等于 x 最后一维
    std::vector<int64_t> gammaShape = {64};
    ADD_INPUT(2, gamma, inDtype, gammaShape);

    // 可选输入 smooth_scales: 1D 张量，大小等于 x 最后一维
    std::vector<int64_t> smoothScalesShape = {64};
    ADD_INPUT(3, smooth_scales, inDtype, smoothScalesShape);

    // 可选输入 beta: 1D 张量，大小等于 x 最后一维
    std::vector<int64_t> betaShape = {64};
    ADD_INPUT(4, beta, inDtype, betaShape);

    outputs.push_back(rms_norm_dynamic_quant_op);
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

    // x, gamma 等输入数据类型必须为 FLOAT16 或 BF16
    DataType inDtype = DT_FLOAT16;
    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }
    std::cout << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph, inputDataHolder);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
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

    // 保存和打印输入数据
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    // 保存和打印输出数据（y: INT8, scale: FLOAT32）
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);

        // y 输出是 INT8, scale 输出是 FLOAT32
        if (output[i].GetTensorDesc().GetDataType() == DT_FLOAT) {
            float* resultData = (float*)output_data_i;
            for (int64_t j = 0; j < output_shape; j++) {
                LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
            }
        } else if (output[i].GetTensorDesc().GetDataType() == DT_INT8) {
            int8_t* resultData = (int8_t*)output_data_i;
            for (int64_t j = 0; j < output_shape; j++) {
                LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
            }
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
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
