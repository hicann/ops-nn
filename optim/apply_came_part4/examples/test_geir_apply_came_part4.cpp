/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_apply_came_part4.cpp
 * @brief ApplyCamePart4 算子 GE IR 图模式调用示例
 *
 * 构图并运行单算子子图；ApplyCamePart4 有 12 个输入 / 3 个输出：
 *   - 输入：param_in, m, r_in, c_in, weight_decay, lr, beta3,
 *           sum_r(optional), sum_u_r, sum_u_c, sum_u_rc, global_shape(optional)
 *   - 输出：param_out, r_out, c_out
 *
 * 目标平台：Ascend950（arch35 / DAV_3510）
 *
 * 用法：默认不给 sum_r / global_shape（走 kernel 内归约路径）；
 *       传入参数 "with_optional" 时连接两个可选输入。
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/apply_came_part4_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
std::vector<std::vector<float>> gInputBuffers;
std::vector<std::vector<uint16_t>> gInputFp16Buffers;
std::vector<std::vector<int64_t>> gInputInt64Buffers;

// 全局形状配置（示例固定 shape，可自行调整）
std::vector<int64_t> kParamShape = {128, 64};
std::vector<int64_t> kRShape = {128};
std::vector<int64_t> kCShape = {64};
const std::vector<int64_t> kScalarShape = {1};
const std::vector<int64_t> kGlobalShape = {2};

// float -> fp16 bits (round to nearest even)
uint16_t FloatToFp16Bits(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign); // underflow to zero (demo data is well-scaled)
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00); // overflow to inf
    }
    uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    if ((mant & 0x1fff) > 0x1000 || ((mant & 0x1fff) == 0x1000 && (half & 1))) {
        half += 1;
    }
    return static_cast<uint16_t>(half);
}

// float -> bf16 bits (round to nearest even)
uint16_t FloatToBf16Bits(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t lsb = (x >> 16) & 1;
    x += 0x7fff + lsb;
    return static_cast<uint16_t>(x >> 16);
}
} // namespace

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

size_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return 4;
    }
    if (dt == ge::DT_INT64 || dt == ge::DT_UINT64) {
        return 8;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return 2;
    }
    return 1;
}

int64_t ShapeSize(const vector<int64_t>& shapes)
{
    int64_t size = 1;
    for (int64_t dim : shapes) {
        size *= dim;
    }
    return size;
}

// 生成确定性的变化数据：value_i = base + i * step，便于 golden 比对
int32_t GenFloatData(const vector<int64_t>& shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc, float base,
                     float step)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    int64_t size = ShapeSize(shapes);
    gInputBuffers.emplace_back(size);
    auto& data = gInputBuffers.back();
    for (int64_t i = 0; i < size; ++i) {
        data[i] = base + static_cast<float>(i) * step;
    }
    inputTensor = Tensor(inputTensorDesc, reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(float));
    return SUCCESS;
}

// fp16/bf16 数据：同样以 base + i*step 生成再编码，保证与 golden 输入一致
int32_t GenFp16Data(const vector<int64_t>& shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc, float base,
                    float step, DataType dtype)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    int64_t size = ShapeSize(shapes);
    gInputFp16Buffers.emplace_back(size);
    auto& data = gInputFp16Buffers.back();
    for (int64_t i = 0; i < size; ++i) {
        float v = base + static_cast<float>(i) * step;
        data[i] = (dtype == DT_BF16) ? FloatToBf16Bits(v) : FloatToFp16Bits(v);
    }
    inputTensor = Tensor(inputTensorDesc, reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(uint16_t));
    return SUCCESS;
}

int32_t GenInt64Data(const vector<int64_t>& shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc,
                     const std::vector<int64_t>& values)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    gInputInt64Buffers.push_back(values);
    auto& data = gInputInt64Buffers.back();
    inputTensor = Tensor(inputTensorDesc, reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(int64_t));
    return SUCCESS;
}

int32_t WriteDataToFile(const string& binFile, uint64_t dataSize, const uint8_t* inputData)
{
    FILE* fp = fopen(binFile.c_str(), "wb");
    if (fp == nullptr) {
        printf("WriteDataToFile: fopen failed for %s\n", binFile.c_str());
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), dataSize, fp);
    fclose(fp);
    if (written != dataSize) {
        printf("WriteDataToFile: short write %zu/%lu\n", written, dataSize);
        return FAILED;
    }
    return SUCCESS;
}

template <typename SetterFn>
int32_t AddFloatInput(int placeholderIndex, SetterFn portSetter, const vector<int64_t>& shape, float base, float step,
                      Graph& graph, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs)
{
    std::string name = "placeholder" + std::to_string(placeholderIndex);
    auto data = op::Data(name.c_str()).set_attr_index(placeholderIndex);
    TensorDesc desc = TensorDesc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    desc.SetFormat(FORMAT_ND);
    desc.SetOriginFormat(FORMAT_ND);
    desc.SetOriginShape(ge::Shape(shape));
    Tensor tensor;
    if (GenFloatData(shape, tensor, desc, base, step) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed for %s\n", GetTime().c_str(), name.c_str());
        return FAILED;
    }
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    input.push_back(tensor);
    graph.AddOp(data);
    portSetter(data);
    inputs.push_back(data);
    return SUCCESS;
}

// param/m/r/c 输入，dtype 可变（fp32/fp16/bf16）
template <typename SetterFn>
int32_t AddTensorInput(int placeholderIndex, SetterFn portSetter, DataType dtype, const vector<int64_t>& shape,
                       float base, float step, Graph& graph, std::vector<ge::Tensor>& input,
                       std::vector<Operator>& inputs)
{
    if (dtype == DT_FLOAT) {
        return AddFloatInput(placeholderIndex, portSetter, shape, base, step, graph, input, inputs);
    }
    std::string name = "placeholder" + std::to_string(placeholderIndex);
    auto data = op::Data(name.c_str()).set_attr_index(placeholderIndex);
    TensorDesc desc = TensorDesc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetFormat(FORMAT_ND);
    desc.SetOriginFormat(FORMAT_ND);
    desc.SetOriginShape(ge::Shape(shape));
    Tensor tensor;
    if (GenFp16Data(shape, tensor, desc, base, step, dtype) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed for %s\n", GetTime().c_str(), name.c_str());
        return FAILED;
    }
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    input.push_back(tensor);
    graph.AddOp(data);
    portSetter(data);
    inputs.push_back(data);
    return SUCCESS;
}

int CreateOppInGraph(DataType tensorDtype, bool withOptional, std::vector<ge::Tensor>& input,
                     std::vector<Operator>& inputs, std::vector<Operator>& outputs, Graph& graph)
{
    auto came4 = op::ApplyCamePart4("apply_came_part4_1");

    int idx = 0;
    if (AddTensorInput(
            idx++, [&](Operator& d) { came4.set_input_param(d); }, tensorDtype, kParamShape, 0.5f, 0.001f, graph, input,
            inputs) != SUCCESS) {
        return FAILED;
    }
    if (AddTensorInput(
            idx++, [&](Operator& d) { came4.set_input_m(d); }, tensorDtype, kParamShape, 0.01f, 0.0001f, graph, input,
            inputs) != SUCCESS) {
        return FAILED;
    }
    if (AddTensorInput(
            idx++, [&](Operator& d) { came4.set_input_r(d); }, tensorDtype, kRShape, 0.2f, 0.002f, graph, input,
            inputs) != SUCCESS) {
        return FAILED;
    }
    if (AddTensorInput(
            idx++, [&](Operator& d) { came4.set_input_c(d); }, tensorDtype, kCShape, 0.3f, 0.003f, graph, input,
            inputs) != SUCCESS) {
        return FAILED;
    }
    if (AddFloatInput(
            idx++, [&](Operator& d) { came4.set_input_weight_decay(d); }, kScalarShape, 0.01f, 0.0f, graph, input,
            inputs) != SUCCESS) {
        return FAILED;
    }
    if (AddFloatInput(
            idx++, [&](Operator& d) { came4.set_input_lr(d); }, kScalarShape, 0.001f, 0.0f, graph, input, inputs) !=
        SUCCESS) {
        return FAILED;
    }
    if (AddFloatInput(
            idx++, [&](Operator& d) { came4.set_input_beta3(d); }, kScalarShape, 0.9f, 0.0f, graph, input, inputs) !=
        SUCCESS) {
        return FAILED;
    }
    if (withOptional) {
        // sum_r 输入值：与无可选输入路径的 kernel 内归约结果（fp32 累加 sum(r_in)）一致
        int64_t rLen = kRShape[0];
        float sumR = static_cast<float>(rLen) * 0.2f + 0.002f * (static_cast<float>(rLen - 1) * rLen / 2.0f);
        if (AddFloatInput(
                idx++, [&](Operator& d) { came4.set_input_sum_r(d); }, kScalarShape, sumR, 0.0f, graph, input,
                inputs) != SUCCESS) {
            return FAILED;
        }
    }
    if (AddFloatInput(
            idx++, [&](Operator& d) { came4.set_input_sum_u_r(d); }, kRShape, 0.4f, 0.001f, graph, input, inputs) !=
        SUCCESS) {
        return FAILED;
    }
    if (AddFloatInput(
            idx++, [&](Operator& d) { came4.set_input_sum_u_c(d); }, kCShape, 0.5f, 0.001f, graph, input, inputs) !=
        SUCCESS) {
        return FAILED;
    }
    if (AddFloatInput(
            idx++, [&](Operator& d) { came4.set_input_sum_u_rc(d); }, kScalarShape, 0.6f, 0.0f, graph, input, inputs) !=
        SUCCESS) {
        return FAILED;
    }
    if (withOptional) {
        std::string name = "placeholder" + std::to_string(idx++);
        auto data = op::Data(name.c_str()).set_attr_index(0);
        TensorDesc desc = TensorDesc(ge::Shape(kGlobalShape), FORMAT_ND, DT_INT64);
        desc.SetFormat(FORMAT_ND);
        desc.SetOriginFormat(FORMAT_ND);
        desc.SetOriginShape(ge::Shape(kGlobalShape));
        Tensor tensor;
        if (GenInt64Data(kGlobalShape, tensor, desc, {kParamShape[0], kParamShape[1]}) != SUCCESS) {
            return FAILED;
        }
        data.update_input_desc_x(desc);
        data.update_output_desc_y(desc);
        input.push_back(tensor);
        graph.AddOp(data);
        came4.set_input_global_shape(data);
        inputs.push_back(data);
    }

    TensorDesc paramOutDesc = TensorDesc(ge::Shape(kParamShape), FORMAT_ND, tensorDtype);
    TensorDesc rOutDesc = TensorDesc(ge::Shape(kRShape), FORMAT_ND, tensorDtype);
    TensorDesc cOutDesc = TensorDesc(ge::Shape(kCShape), FORMAT_ND, tensorDtype);
    came4.update_output_desc_param(paramOutDesc);
    came4.update_output_desc_r(rOutDesc);
    came4.update_output_desc_c(cOutDesc);

    outputs.push_back(came4);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    // 用法: test_geir_apply_came_part4 [fp32|fp16|bf16] [with_optional] [n m]
    std::string dtypeStr = (argc > 1) ? argv[1] : "fp32";
    bool withOptional = (argc > 2 && std::string(argv[2]) == "with_optional");
    DataType tensorDtype = DT_FLOAT;
    if (dtypeStr == "fp16") {
        tensorDtype = DT_FLOAT16;
    } else if (dtypeStr == "bf16") {
        tensorDtype = DT_BF16;
    } else if (dtypeStr == "with_optional") { // 兼容单参数用法
        tensorDtype = DT_FLOAT;
        dtypeStr = "fp32";
        withOptional = true;
    }
    if (argc > 4) {
        int64_t n = std::stoll(argv[3]);
        int64_t m = std::stoll(argv[4]);
        kParamShape = {n, m};
        kRShape = {n};
        kCShape = {m};
    }
    std::string graphNameStr = "tc_apply_came_part4_" + dtypeStr + (withOptional ? "_optional" : "");
    Graph graph(graphNameStr.c_str());
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

    ret = CreateOppInGraph(tensorDtype, withOptional, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    std::unique_ptr<ge::Session> session(new (std::nothrow) Session(build_options));
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        session.reset();
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());

    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        session.reset();
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    const string prefix = "./" + graphNameStr;
    for (size_t i = 0; i < input.size(); ++i) {
        string input_file = prefix + "_npu_input_" + std::to_string(i) + ".bin";
        int64_t shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        const uint64_t data_size = static_cast<uint64_t>(shape) *
                                   GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(input_file, data_size, input[i].GetData()) != SUCCESS) {
            session.reset();
            GEFinalize();
            return FAILED;
        }
    }

    const size_t output_num = output.size();
    if (output_num != 3) {
        printf("%s - ERROR - [XIR]: Expected 3 outputs, got %zu\n", GetTime().c_str(), output_num);
        session.reset();
        GEFinalize();
        return FAILED;
    }
    for (size_t i = 0; i < output_num; ++i) {
        string output_file = prefix + "_npu_output_" + std::to_string(i) + ".bin";
        int64_t shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        const uint64_t data_size = static_cast<uint64_t>(shape) *
                                   GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(output_file, data_size, output[i].GetData()) != SUCCESS) {
            session.reset();
            GEFinalize();
            return FAILED;
        }
    }

    printf("%s - INFO - [XIR]: Generated all graph outputs\n", GetTime().c_str());
    session.reset();
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
