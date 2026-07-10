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
#include <sys/time.h>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/global_lp_pool_proto.h"

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
    return 4;
}

int32_t GenTestData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    const vector<float>& host_data)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++)
        size *= shapes[i];
    uint32_t data_len = size * GetDataTypeSize(data_type);
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];

    if (data_type == DT_FLOAT16) {
        uint16_t* ptr = reinterpret_cast<uint16_t*>(pData);
        for (size_t i = 0; i < size; i++) {
            // float -> fp16 via simple conversion
            float v = host_data[i];
            uint32_t raw = *(uint32_t*)&v;
            uint16_t sign = (raw >> 16) & 0x8000;
            uint16_t exp = ((raw >> 23) & 0xff);
            uint16_t mant = (raw >> 13) & 0x3ff;
            if (exp == 0) {
                ptr[i] = sign;
            } else if (exp < 112) {
                ptr[i] = sign;
            } else if (exp > 142) {
                ptr[i] = sign | 0x7c00;
            } else {
                ptr[i] = sign | ((exp - 112) << 10) | mant;
            }
        }
    } else if (data_type == DT_FLOAT) {
        float* ptr = reinterpret_cast<float*>(pData);
        for (size_t i = 0; i < size; i++)
            ptr[i] = host_data[i];
    }
    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    return SUCCESS;
}

float ComputeGolden(const vector<float>& input, float p)
{
    double sum = 0.0;
    for (auto v : input)
        sum += std::pow(std::abs((double)v), (double)p);
    return (float)std::pow(sum, 1.0 / (double)p);
}

struct TestCase {
    string name;
    vector<int64_t> shape; // 4D or 5D
    float p;
};

int main(int argc, char* argv[])
{
    // ── 5 cases from UT ──
    TestCase cases[] = {
        {"basic_4d_fp16_p2", {1, 64, 32, 32}, 2.0f},
        {"small_spatial_4d_fp16_p1", {2, 32, 4, 4}, 1.0f},
        {"large_channel_4d_fp16_p3", {1, 256, 16, 16}, 3.0f},
        {"basic_5d_fp16_p2", {1, 32, 4, 8, 8}, 2.0f}, // NCD0D1D2
        {"empty_a_axis", {0, 64, 32, 32}, 2.0f},
    };

    // ── Init GE ──
    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(global_options) != SUCCESS) {
        printf("%s - ERROR - GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEInitialize success\n", GetTime().c_str());

    std::map<ge::AscendString, ge::AscendString> session_options = {};
    ge::Session* session = new ge::Session(session_options);
    if (session == nullptr) {
        ge::GEFinalize();
        return FAILED;
    }

    int passed = 0;
    int failed = 0;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const auto& tc = cases[ci];
        int64_t N = tc.shape[0], C = tc.shape[1];
        int64_t spatial = 1;
        for (size_t d = 2; d < tc.shape.size(); d++)
            spatial *= tc.shape[d];
        int64_t input_size = N * C * spatial;
        int64_t out_size = N * C;

        printf("\n=== Case %zu: %s | shape=(", ci, tc.name.c_str());
        for (size_t d = 0; d < tc.shape.size(); d++)
            printf("%ld%s", tc.shape[d], d + 1 < tc.shape.size() ? "," : "");
        printf(") p=%.1f ===\n", tc.p);

        // ── Generate random input data ──
        vector<float> host_data(input_size);
        for (int64_t i = 0; i < input_size; i++)
            host_data[i] = -5.0f + (float)rand() / RAND_MAX * 10.0f;

        // ── Compute golden ──
        vector<float> golden(out_size);
        for (int64_t n = 0; n < N; n++) {
            for (int64_t c = 0; c < C; c++) {
                vector<float> spatial_data(spatial);
                for (int64_t s = 0; s < spatial; s++) {
                    int64_t idx = n * C * spatial + c * spatial + s;
                    spatial_data[s] = host_data[idx];
                }
                golden[n * C + c] = ComputeGolden(spatial_data, tc.p);
            }
        }

        // ── Build graph ──
        Graph graph("glp_graph");
        DataType inDtype = DT_FLOAT16;

        auto inputData = op::Data("input");
        TensorDesc inputDesc(ge::Shape(tc.shape), FORMAT_ND, inDtype);
        inputDesc.SetPlacement(ge::kPlacementHost);
        inputDesc.SetFormat(FORMAT_ND);
        inputData.update_input_desc_x(inputDesc);
        inputData.update_output_desc_y(inputDesc);

        auto glpOp = op::GlobalLpPool("global_lp_pool");
        glpOp.set_attr_p(tc.p);
        glpOp.set_input_x(inputData);
        glpOp.update_input_desc_x(inputDesc);

        vector<int64_t> outShapeVec;
        outShapeVec.push_back(N);
        outShapeVec.push_back(C);
        for (size_t d = 2; d < tc.shape.size(); d++)
            outShapeVec.push_back(1);

        TensorDesc outputDesc(ge::Shape(outShapeVec), FORMAT_ND, inDtype);
        outputDesc.SetPlacement(ge::kPlacementHost);
        outputDesc.SetFormat(FORMAT_ND);
        glpOp.update_output_desc_y(outputDesc);

        graph.AddOp(inputData);
        graph.AddOp(glpOp);
        vector<Operator> graphInputs{inputData};
        vector<Operator> graphOutputs{glpOp};
        graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

        // ── Set input data ──
        Tensor inputTensor;
        GenTestData(tc.shape, inputTensor, inputDesc, inDtype, host_data);
        vector<Tensor> inputs{inputTensor};

        // ── Add to session and run ──
        uint32_t graph_id = (uint32_t)ci;
        std::map<AscendString, AscendString> graph_options = {};
        if (session->AddGraph(graph_id, graph, graph_options) != SUCCESS) {
            printf("  FAIL: AddGraph failed\n");
            failed++;
            continue;
        }

        vector<Tensor> outputs;
        Status runRet = session->RunGraph(graph_id, inputs, outputs);
        if (runRet != SUCCESS || outputs.empty()) {
            printf("  FAIL: RunGraph failed (ret=%d)\n", runRet);
            failed++;
            continue;
        }

        // ── Compare with golden ──
        const float* outData_f32 = nullptr;
        vector<float> outF32(out_size);
        const uint8_t* rawData = outputs[0].GetData();
        if (inDtype == DT_FLOAT16) {
            const uint16_t* fp16Data = reinterpret_cast<const uint16_t*>(rawData);
            for (int64_t i = 0; i < out_size; i++) {
                // fp16 -> float
                uint16_t v = fp16Data[i];
                uint32_t sign = (v >> 15) & 1;
                uint32_t exp = (v >> 10) & 0x1f;
                uint32_t mant = v & 0x3ff;
                uint32_t f32;
                if (exp == 0) {
                    f32 = sign << 31;
                } else if (exp == 31) {
                    f32 = (sign << 31) | 0x7f800000 | (mant << 13);
                } else {
                    f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                }
                outF32[i] = *(float*)&f32;
            }
            outData_f32 = outF32.data();
        } else {
            outData_f32 = reinterpret_cast<const float*>(rawData);
        }

        float maxRelErr = 0.0f;
        int mismatches = 0;
        for (int64_t i = 0; i < out_size; i++) {
            float expected = golden[i];
            float actual = outData_f32[i];
            float absErr = std::abs(actual - expected);
            float relErr = (std::abs(expected) > 1e-6f) ? absErr / std::abs(expected) : absErr;
            if (relErr > maxRelErr)
                maxRelErr = relErr;
            if (relErr > 0.01f) {
                if (mismatches < 3)
                    printf("  mism[%ld]: exp=%.4f act=%.4f err=%.2e\n", i, expected, actual, relErr);
                mismatches++;
            }
        }

        if (mismatches == 0) {
            printf("  ✅ PASS (maxRelErr=%.2e)\n", maxRelErr);
            passed++;
        } else {
            printf("  ❌ FAIL: %d/%ld mismatches (maxRelErr=%.2e)\n", mismatches, out_size, maxRelErr);
            failed++;
        }
    }

    printf("\n=== Result: %d PASS, %d FAIL ===\n", passed, failed);

    if (session != nullptr)
        delete session;
    ge::GEFinalize();
    return (failed == 0) ? SUCCESS : FAILED;
}
