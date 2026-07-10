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
 * \file test_geir_sparse_segment_sum_grad.cpp
 * \brief GEIR graph-mode example for SparseSegmentSumGrad.
 *        Computes: output[indices[j]] += grad[segment_ids[j]]
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <ctime>
#include <new>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/sparse_segment_sum_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

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
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;
    uint32_t size = 1;
    if (dt == ge::DT_FLOAT) {
        size = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        size = twoByte;
    } else if (dt == ge::DT_BF16) {
        size = twoByte;
    } else if (dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        size = twoByte;
    } else if (dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        size = fourByte;
    } else if (dt == ge::DT_INT64 || dt == ge::DT_UINT64) {
        size = eightByte;
    } else if (dt == ge::DT_INT8 || dt == ge::DT_UINT8) {
        size = oneByte;
    }
    return size;
}

// Generate sequential float data 0, 1, 2, ... for grad.
int32_t GenFloatSeq(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * sizeof(float);
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = static_cast<float>(i);
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    return SUCCESS;
}

// Generate int32 data from a given value array (for indices / segment_ids).
int32_t GenInt32Arr(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, const int32_t* vals,
                    size_t valsCnt)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * sizeof(int32_t);
    int32_t* pData = new (std::nothrow) int32_t[size];
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = (i < valsCnt) ? vals[i] : 0;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp;
    fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto sparseSegmentSumGrad = op::SparseSegmentSumGrad("sparseSegmentSumGrad");

    // grad: Data input, float32, shape {3, 2} -> [[0,1],[2,3],[4,5]]
    vector<int64_t> gradShape = {3, 2};
    auto placeholderGrad = op::Data("placeholderGrad").set_attr_index(0);
    TensorDesc gradDesc = TensorDesc(ge::Shape(gradShape), FORMAT_ND, inDtype);
    gradDesc.SetPlacement(ge::kPlacementHost);
    gradDesc.SetFormat(FORMAT_ND);
    Tensor tensorGrad;
    ret = GenFloatSeq(gradShape, tensorGrad, gradDesc);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate grad data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholderGrad.update_input_desc_x(gradDesc);
    placeholderGrad.update_output_desc_y(gradDesc);
    input.push_back(tensorGrad);
    graph.AddOp(placeholderGrad);
    sparseSegmentSumGrad.set_input_grad(placeholderGrad);
    inputs.push_back(placeholderGrad);

    // indices: Const input, int32, shape {4}, values [0, 1, 2, 0]
    vector<int64_t> indicesShape = {4};
    auto placeholderIndices = op::Const("placeholderIndices");
    TensorDesc indicesDesc = TensorDesc(ge::Shape(indicesShape), FORMAT_ND, DT_INT32);
    indicesDesc.SetPlacement(ge::kPlacementHost);
    indicesDesc.SetFormat(FORMAT_ND);
    Tensor tensorIndices;
    const int32_t indicesVals[] = {0, 1, 2, 0};
    ret = GenInt32Arr(indicesShape, tensorIndices, indicesDesc, indicesVals, sizeof(indicesVals) / sizeof(int32_t));
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate indices data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholderIndices.SetAttr("value", tensorIndices);
    placeholderIndices.update_output_desc_y(indicesDesc);
    graph.AddOp(placeholderIndices);
    sparseSegmentSumGrad.set_input_indices(placeholderIndices);
    sparseSegmentSumGrad.update_input_desc_indices(indicesDesc);
    inputs.push_back(placeholderIndices);

    // segment_ids: Const input, int32, shape {4}, values [0, 1, 2, 0]
    vector<int64_t> segmentIdsShape = {4};
    auto placeholderSegmentIds = op::Const("placeholderSegmentIds");
    TensorDesc segmentIdsDesc = TensorDesc(ge::Shape(segmentIdsShape), FORMAT_ND, DT_INT32);
    segmentIdsDesc.SetPlacement(ge::kPlacementHost);
    segmentIdsDesc.SetFormat(FORMAT_ND);
    Tensor tensorSegmentIds;
    const int32_t segmentIdsVals[] = {0, 1, 2, 0};
    ret = GenInt32Arr(segmentIdsShape, tensorSegmentIds, segmentIdsDesc, segmentIdsVals,
                      sizeof(segmentIdsVals) / sizeof(int32_t));
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate segment_ids data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholderSegmentIds.SetAttr("value", tensorSegmentIds);
    placeholderSegmentIds.update_output_desc_y(segmentIdsDesc);
    graph.AddOp(placeholderSegmentIds);
    sparseSegmentSumGrad.set_input_segment_ids(placeholderSegmentIds);
    sparseSegmentSumGrad.update_input_desc_segment_ids(segmentIdsDesc);
    inputs.push_back(placeholderSegmentIds);

    // output_dim0: Const input, int32 scalar, value 3 (output first dim size)
    vector<int64_t> outputDim0Shape = {1};
    auto placeholderOutputDim0 = op::Const("placeholderOutputDim0");
    TensorDesc outputDim0Desc = TensorDesc(ge::Shape(outputDim0Shape), FORMAT_ND, DT_INT32);
    outputDim0Desc.SetPlacement(ge::kPlacementHost);
    outputDim0Desc.SetFormat(FORMAT_ND);
    Tensor tensorOutputDim0;
    const int32_t outputDim0Val = 3;
    ret = GenInt32Arr(outputDim0Shape, tensorOutputDim0, outputDim0Desc, &outputDim0Val, 1);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate output_dim0 data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholderOutputDim0.SetAttr("value", tensorOutputDim0);
    placeholderOutputDim0.update_output_desc_y(outputDim0Desc);
    graph.AddOp(placeholderOutputDim0);
    sparseSegmentSumGrad.set_input_output_dim0(placeholderOutputDim0);
    sparseSegmentSumGrad.update_input_desc_output_dim0(outputDim0Desc);
    inputs.push_back(placeholderOutputDim0);

    outputs.push_back(sparseSegmentSumGrad);
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

    DataType inDtype = DT_FLOAT;
    std::cout << "inDtype: " << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
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

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed, ret = %d\n", GetTime().c_str(), ret);
        delete session;
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
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        float* resultData = (float*)output_data_i;
        printf("output result (shape [%ld]):\n", output_shape);
        for (int64_t j = 0; j < output_shape; j++) {
            printf("result[%ld] is: %f\n", j, resultData[j]);
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
