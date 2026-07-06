/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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
#include <cmath>
#include <cstdlib>
#include <cassert>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "array_ops.h"

#include "../op_graph/acts_ulq_proto.h"

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
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16)
        return 2;
    if (dt == ge::DT_BF16)
        return 2;
    if (dt == ge::DT_INT16)
        return 2;
    if (dt == ge::DT_UINT16)
        return 2;
    if (dt == ge::DT_INT32)
        return 4;
    if (dt == ge::DT_UINT32)
        return 4;
    if (dt == ge::DT_INT64)
        return 8;
    if (dt == ge::DT_UINT64)
        return 8;
    if (dt == ge::DT_INT8)
        return 1;
    return 1;
}

int32_t GenFloatData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                     float low, float high, unsigned int seed)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    float* pData = new (std::nothrow) float[size];
    srand(seed);
    for (size_t i = 0; i < size; ++i) {
        pData[i] = low + (high - low) * ((float)rand() / (float)RAND_MAX);
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

// CPU Golden: ActsULQ computation (float32 precision)
// cmin and cmax are scalars (broadcast to all elements)
void ActsUlqGolden(const float* data, const float* cmin, const float* cmax, float* out, float* min_mask,
                   float* max_mask, float* loss, size_t size, bool fixed_min)
{
    const float step = 255.0f;
    const float eps = 1.192092896e-07f;
    const float inv_step = 1.0f / step;

    // cmin and cmax are scalars
    float cmin_val = cmin[0];
    float cmax_val = cmax[0];

    for (size_t i = 0; i < size; i++) {
        float ori_clip_min = fixed_min ? 0.0f : std::min(cmin_val, 0.0f);
        float ori_clip_max = std::max(cmax_val, step * eps);
        float scale = (ori_clip_max - ori_clip_min) * inv_step;
        float offset = std::round(ori_clip_min / scale);
        float clip_min = scale * offset;
        float clip_max = scale * (offset + step);

        float clamped_x = std::min(std::max(data[i], clip_min), clip_max);
        float cmask_min = (data[i] >= clip_min) ? 1.0f : 0.0f;
        float cmask_max = (data[i] <= clip_max) ? 1.0f : 0.0f;

        float raw_x = clamped_x / scale;
        float round_x = std::round(raw_x);
        float clamped_loss = (round_x - raw_x) * inv_step;

        float raw_m = ori_clip_min / scale;
        float round_m = std::round(raw_m);
        float loss_m = (round_m - raw_m) * inv_step;

        float x_clamped_loss = (cmask_min != 0.0f) ? clamped_loss : loss_m;
        x_clamped_loss = (cmask_max != 0.0f) ? x_clamped_loss : loss_m;

        out[i] = round_x * scale;
        min_mask[i] = cmask_min;
        max_mask[i] = cmask_max;
        loss[i] = x_clamped_loss;
    }
}

bool CompareFloat(const float* golden, const float* actual, size_t size, float rtol, float atol, const char* name)
{
    int fail_count = 0;
    int first_fail_idx = -1;
    float first_golden = 0, first_actual = 0;

    for (size_t i = 0; i < size; i++) {
        float diff = std::abs(actual[i] - golden[i]);
        float threshold = atol + rtol * std::abs(golden[i]);
        if (diff > threshold) {
            fail_count++;
            if (first_fail_idx < 0) {
                first_fail_idx = i;
                first_golden = golden[i];
                first_actual = actual[i];
            }
        }
    }

    if (fail_count == 0) {
        printf("    [PASS] %s (%zu elems)\n", name, size);
        return true;
    } else {
        float diff = std::abs(first_actual - first_golden);
        float threshold = atol + rtol * std::abs(first_golden);
        printf("    [FAIL] %s: %d/%zu mismatches\n", name, fail_count, size);
        printf("           first fail [%d]: golden=%.8g actual=%.8g diff=%.4e tol=%.4e\n", first_fail_idx, first_golden,
               first_actual, diff, threshold);
        return false;
    }
}

bool CompareMask(const float* golden, const float* actual, size_t size, const char* name)
{
    int fail_count = 0;
    int first_fail_idx = -1;

    for (size_t i = 0; i < size; i++) {
        if (golden[i] != actual[i]) {
            fail_count++;
            if (first_fail_idx < 0) {
                first_fail_idx = i;
            }
        }
    }

    if (fail_count == 0) {
        printf("    [PASS] %s (%zu elems, exact match)\n", name, size);
        return true;
    } else {
        printf("    [FAIL] %s: %d/%zu mismatches\n", name, fail_count, size);
        printf("           first fail [%d]: golden=%.1f actual=%.1f\n", first_fail_idx, golden[first_fail_idx],
               actual[first_fail_idx]);
        return false;
    }
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph, bool fixed_min, float*& data_host, float*& cmin_host,
                     float*& cmax_host, size_t& total_elems)
{
    Status ret = SUCCESS;
    auto acts_ulq1 = op::ActsULQ("acts_ulq1");

    acts_ulq1.set_attr_fixed_min(fixed_min);
    acts_ulq1.set_attr_num_bits(8);

    std::vector<int64_t> dataShape = {4, 4};
    std::vector<int64_t> clampShape = {1};
    total_elems = 16;
    size_t clamp_elems = 1;

    // Generate meaningful input data
    data_host = new float[total_elems];
    cmin_host = new float[clamp_elems];
    cmax_host = new float[clamp_elems];

    srand(42);
    for (size_t i = 0; i < total_elems; i++) {
        data_host[i] = -3.0f + 6.0f * ((float)rand() / (float)RAND_MAX);
    }
    cmin_host[0] = -4.0f + 2.0f * ((float)rand() / (float)RAND_MAX);
    cmax_host[0] = 1.0f + 3.0f * ((float)rand() / (float)RAND_MAX);

    // Input 1: data
    {
        auto ph = op::Data("placeholder1").set_attr_index(0);
        TensorDesc desc(ge::Shape(dataShape), FORMAT_ND, inDtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor(desc, (uint8_t*)data_host, total_elems * sizeof(float));
        ph.update_input_desc_x(desc);
        ph.update_output_desc_y(desc);
        input.push_back(tensor);
        graph.AddOp(ph);
        acts_ulq1.set_input_x(ph);
        inputs.push_back(ph);
    }

    // Input 2: clamp_min
    {
        auto ph = op::Data("placeholder2").set_attr_index(0);
        TensorDesc desc(ge::Shape(clampShape), FORMAT_ND, inDtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor(desc, (uint8_t*)cmin_host, clamp_elems * sizeof(float));
        ph.update_input_desc_x(desc);
        ph.update_output_desc_y(desc);
        input.push_back(tensor);
        graph.AddOp(ph);
        acts_ulq1.set_input_clamp_min(ph);
        inputs.push_back(ph);
    }

    // Input 3: clamp_max
    {
        auto ph = op::Data("placeholder3").set_attr_index(0);
        TensorDesc desc(ge::Shape(clampShape), FORMAT_ND, inDtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor(desc, (uint8_t*)cmax_host, clamp_elems * sizeof(float));
        ph.update_input_desc_x(desc);
        ph.update_output_desc_y(desc);
        input.push_back(tensor);
        graph.AddOp(ph);
        acts_ulq1.set_input_clamp_max(ph);
        inputs.push_back(ph);
    }

    // 4 outputs
    {
        TensorDesc desc(ge::Shape(dataShape), FORMAT_ND, inDtype);
        acts_ulq1.update_output_desc_y(desc);
    }
    {
        TensorDesc desc(ge::Shape(dataShape), FORMAT_ND, inDtype);
        acts_ulq1.update_output_desc_clamp_min_mask(desc);
    }
    {
        TensorDesc desc(ge::Shape(dataShape), FORMAT_ND, inDtype);
        acts_ulq1.update_output_desc_clamp_max_mask(desc);
    }
    {
        TensorDesc desc(ge::Shape(dataShape), FORMAT_ND, inDtype);
        acts_ulq1.update_output_desc_x_clamped_loss(desc);
    }

    outputs.push_back(acts_ulq1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [GEIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [GEIR]: Initialize ge success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    DataType inDtype = DT_FLOAT;
    bool fixed_min = false;
    float* data_host = nullptr;
    float* cmin_host = nullptr;
    float* cmax_host = nullptr;
    size_t total_elems = 0;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph, fixed_min, data_host, cmin_host, cmax_host,
                           total_elems);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: CreateOppInGraph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [GEIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [GEIR]: AddGraph success\n", GetTime().c_str());

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: Run graph failed with error code: %d\n", GetTime().c_str(), ret);
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [GEIR]: Run graph success\n", GetTime().c_str());

    // Compute CPU golden
    float* golden_out = new float[total_elems];
    float* golden_min_mask = new float[total_elems];
    float* golden_max_mask = new float[total_elems];
    float* golden_loss = new float[total_elems];
    ActsUlqGolden(data_host, cmin_host, cmax_host, golden_out, golden_min_mask, golden_max_mask, golden_loss,
                  total_elems, fixed_min);

    // Compare NPU output with golden
    const float rtol = 1.220703125e-04f;
    const float atol = 1.220703125e-03f;
    bool all_pass = true;

    printf("\n=== Precision Comparison (FP32: rtol=%.4e, atol=%.4e) ===\n", rtol, atol);

    if (output.size() >= 4) {
        float* npu_out = (float*)output[0].GetData();
        float* npu_min_mask = (float*)output[1].GetData();
        float* npu_max_mask = (float*)output[2].GetData();
        float* npu_loss = (float*)output[3].GetData();

        all_pass &= CompareFloat(golden_out, npu_out, total_elems, rtol, atol, "output");
        all_pass &= CompareMask(golden_min_mask, npu_min_mask, total_elems, "clamp_min_mask");
        all_pass &= CompareMask(golden_max_mask, npu_max_mask, total_elems, "clamp_max_mask");
        all_pass &= CompareFloat(golden_loss, npu_loss, total_elems, rtol, atol, "x_clamped_loss");
    } else {
        printf("    [FAIL] Expected 4 outputs, got %zu\n", output.size());
        all_pass = false;
    }

    printf("\n=== Result: %s ===\n", all_pass ? "ALL PASS" : "FAIL");

    // Cleanup
    delete[] golden_out;
    delete[] golden_min_mask;
    delete[] golden_max_mask;
    delete[] golden_loss;
    delete[] data_host;
    delete[] cmin_host;
    delete[] cmax_host;

    delete session;
    GEFinalize();

    return all_pass ? SUCCESS : FAILED;
}
