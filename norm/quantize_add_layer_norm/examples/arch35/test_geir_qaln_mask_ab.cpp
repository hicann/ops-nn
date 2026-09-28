/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_qaln_mask_ab.cpp
 * \brief ascend950 real-device driver for the arch35 full_load tail-mask issue (cols%64 != 0).
 *        Two modes:
 *        1. generate mode (default, A/B harness): deterministic integer-derived fp32 inputs,
 *           shape via QALN_ROWS/QALN_COLS/QALN_PREFIX; dumps every input tensor and y.
 *        2. replay mode (QALN_CASE_DIR=<caseDir>): loads the frozen representative-case inputs
 *           prepared by qaln_case_tool.py (bin/<name>.bin + manifest.txt, converted from the
 *           issue package npy files), replays exact dtype/shape/attrs (incl. zero_points and
 *           additional_output), dumps y (and x when additional_output=true).
 *
 * Usage: ./test_geir_qaln_mask_ab <rows> <cols> <zp 0|1> <epsilon> <outPrefix>
 */

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
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

#include "../../op_graph/quantize_add_layer_norm_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int64_t gRowsDefault = 672;
constexpr int64_t gColsDefault = 224;

int32_t WriteBin(const string& path, uint64_t bytes, const uint8_t* data)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        printf("open %s for write failed\n", path.c_str());
        return FAILED;
    }
    fwrite(data, sizeof(uint8_t), bytes, fp);
    fclose(fp);
    printf("[AB] dumped %s (%llu bytes)\n", path.c_str(), static_cast<unsigned long long>(bytes));
    return SUCCESS;
}

// All values are integer * power-of-two: exactly representable in fp32 and trivially
// reproducible by the off-device numpy oracle.
float GenX1(int64_t idx) { return static_cast<float>(((idx * 37) % 199) - 99) * 0.03125f; }
float GenX2(int64_t idx) { return static_cast<float>(((idx * 53) % 173) - 86) * 0.015625f; }
float GenGamma(int64_t c) { return 0.5f + static_cast<float>(c % 7) * 0.125f; }
float GenBeta(int64_t c) { return static_cast<float>(((c * 13) % 41) - 20) * 0.125f; }
float GenBias(int64_t c) { return static_cast<float>(((c * 29) % 61) - 30) * 0.0625f; }
// negative per-channel div scales, mirroring bb_l2_941: in [-2.125, -0.125]
float GenScale(int64_t c) { return -(0.125f * static_cast<float>(1 + (c % 16))); }

int32_t GenFp32Tensor(int64_t elemNum, float (*gen)(int64_t), TensorDesc& desc, Tensor& tensor, const string& dumpPath)
{
    desc.SetRealDimCnt(1);
    uint64_t dataLen = static_cast<uint64_t>(elemNum) * sizeof(float);
    float* pData = new (std::nothrow) float[elemNum];
    if (pData == nullptr) {
        return FAILED;
    }
    for (int64_t i = 0; i < elemNum; i++) {
        pData[i] = gen(i);
    }
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(pData), dataLen);
    if (!dumpPath.empty()) {
        WriteBin(dumpPath, dataLen, reinterpret_cast<uint8_t*>(pData));
    }
    delete[] pData;
    return SUCCESS;
}

// ---- replay mode helpers ----
struct CaseCfg {
    string caseName;
    vector<int64_t> xDims;
    int64_t cols = 0;
    string xDtype = "fp32";     // fp16 | bf16 | fp32 (x1/x2/gamma/beta/bias and x output)
    string scaleDtype = "fp32"; // scales / zero_points (fp32 or bf16)
    int hasZp = 0;
    int outX = 0;
    float epsilon = 1e-5f;
};

ge::DataType GeDt(const string& s)
{
    if (s == "fp16") {
        return DT_FLOAT16;
    }
    if (s == "bf16") {
        return DT_BF16;
    }
    return DT_FLOAT;
}

uint32_t DtItemBytes(ge::DataType dt)
{
    if (dt == DT_FLOAT || dt == DT_INT32) {
        return 4U;
    }
    return 2U; // fp16 / bf16
}

bool ParseCaseCfg(const string& path, CaseCfg& cfg)
{
    FILE* fp = fopen(path.c_str(), "r");
    if (fp == nullptr) {
        printf("[CASE] cannot open %s\n", path.c_str());
        return false;
    }
    char line[512];
    while (fgets(line, sizeof(line), fp) != nullptr) {
        string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        size_t eq = s.find('=');
        if (eq == string::npos) {
            continue;
        }
        string k = s.substr(0, eq);
        string v = s.substr(eq + 1);
        if (k == "case") {
            cfg.caseName = v;
        } else if (k == "dims") {
            cfg.xDims.clear();
            size_t pos = 0;
            while (pos < v.size()) {
                size_t comma = v.find(',', pos);
                string tok = (comma == string::npos) ? v.substr(pos) : v.substr(pos, comma - pos);
                if (!tok.empty()) {
                    cfg.xDims.push_back(strtoll(tok.c_str(), nullptr, 10));
                }
                if (comma == string::npos) {
                    break;
                }
                pos = comma + 1;
            }
        } else if (k == "xdtype") {
            cfg.xDtype = v;
        } else if (k == "scaledtype") {
            cfg.scaleDtype = v;
        } else if (k == "zp") {
            cfg.hasZp = atoi(v.c_str());
        } else if (k == "outx") {
            cfg.outX = atoi(v.c_str());
        } else if (k == "epsilon") {
            cfg.epsilon = strtof(v.c_str(), nullptr);
        }
    }
    fclose(fp);
    if (cfg.xDims.empty()) {
        printf("[CASE] manifest has no dims\n");
        return false;
    }
    cfg.cols = cfg.xDims.back();
    return true;
}

uint8_t* ReadBin(const string& path, uint64_t& bytes)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        printf("[CASE] cannot open %s\n", path.c_str());
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* buf = new (std::nothrow) uint8_t[sz];
    if (buf != nullptr && sz > 0) {
        if (fread(buf, 1, static_cast<size_t>(sz), fp) != static_cast<size_t>(sz)) {
            delete[] buf;
            buf = nullptr;
        }
    }
    fclose(fp);
    bytes = static_cast<uint64_t>(sz);
    return buf;
}

Status AddFileInput(Graph& graph, std::vector<ge::Tensor>& inputTensors, std::vector<Operator>& inputs,
                    op::QuantizeAddLayerNorm& add1, const string& name, const vector<int64_t>& shape,
                    const string& path, ge::DataType dt)
{
    uint64_t bytes = 0;
    uint8_t* data = ReadBin(path, bytes);
    if (data == nullptr) {
        return FAILED;
    }
    int64_t elem = 1;
    for (auto d : shape) {
        elem *= d;
    }
    uint64_t expect = static_cast<uint64_t>(elem) * DtItemBytes(dt);
    if (bytes != expect) {
        printf("[CASE] %s size mismatch: file %llu bytes, expect %llu (%lld elems)\n", name.c_str(),
               static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(expect),
               static_cast<long long>(elem));
        delete[] data;
        return FAILED;
    }
    auto dataOp = op::Data("file_" + name).set_attr_index(0);
    TensorDesc desc = TensorDesc(ge::Shape(shape), FORMAT_ND, dt);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    dataOp.update_input_desc_x(desc);
    dataOp.update_output_desc_y(desc);
    inputTensors.push_back(Tensor(desc, data, bytes));
    graph.AddOp(dataOp);
    if (name == "x1") {
        add1.set_input_x1(dataOp);
    } else if (name == "x2") {
        add1.set_input_x2(dataOp);
    } else if (name == "gamma") {
        add1.set_input_gamma(dataOp);
    } else if (name == "beta") {
        add1.set_input_beta(dataOp);
    } else if (name == "bias") {
        add1.set_input_bias(dataOp);
    } else if (name == "scales") {
        add1.set_input_scales(dataOp);
    } else if (name == "zero_points") {
        add1.set_input_zero_points(dataOp);
    }
    inputs.push_back(dataOp);
    printf("[CASE] input %-11s dtype=%d elems=%lld\n", name.c_str(), static_cast<int>(dt),
           static_cast<long long>(elem));
    delete[] data; // Tensor(desc, data, len) copies the buffer
    return SUCCESS;
}
} // namespace

static Status AddProbeInput(Graph& graph, std::vector<ge::Tensor>& inputTensors, std::vector<Operator>& inputs,
                            op::QuantizeAddLayerNorm& add1, const string& name, const vector<int64_t>& shape,
                            float (*gen)(int64_t), const string& dumpPath)
{
    auto data = op::Data("placeholder_" + name).set_attr_index(0);
    TensorDesc desc = TensorDesc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    Tensor tensor;
    int64_t elem = 1;
    for (auto d : shape) {
        elem *= d;
    }
    TensorDesc genDesc = desc;
    if (GenFp32Tensor(elem, gen, genDesc, tensor, dumpPath) != SUCCESS) {
        printf("gen tensor %s failed\n", name.c_str());
        return FAILED;
    }
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    inputTensors.push_back(tensor);
    graph.AddOp(data);
    if (name == "x1") {
        add1.set_input_x1(data);
    } else if (name == "x2") {
        add1.set_input_x2(data);
    } else if (name == "gamma") {
        add1.set_input_gamma(data);
    } else if (name == "beta") {
        add1.set_input_beta(data);
    } else if (name == "bias") {
        add1.set_input_bias(data);
    } else if (name == "scales") {
        add1.set_input_scales(data);
    } else if (name == "zero_points") {
        add1.set_input_zero_points(data);
    }
    inputs.push_back(data);
    return SUCCESS;
}

int CreateOppInGraph(int64_t rows, int64_t cols, float epsilon, const string& outPrefix,
                     std::vector<ge::Tensor>& inputTensors, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    auto add1 = op::QuantizeAddLayerNorm("add1");
    vector<int64_t> xShape = {rows, cols};
    vector<int64_t> colShape = {cols};

    if (AddProbeInput(graph, inputTensors, inputs, add1, "x1", xShape, GenX1, outPrefix + "_x1.bin") != SUCCESS ||
        AddProbeInput(graph, inputTensors, inputs, add1, "x2", xShape, GenX2, outPrefix + "_x2.bin") != SUCCESS ||
        AddProbeInput(graph, inputTensors, inputs, add1, "gamma", colShape, GenGamma, outPrefix + "_gamma.bin") !=
            SUCCESS ||
        AddProbeInput(graph, inputTensors, inputs, add1, "beta", colShape, GenBeta, outPrefix + "_beta.bin") !=
            SUCCESS ||
        AddProbeInput(graph, inputTensors, inputs, add1, "bias", colShape, GenBias, outPrefix + "_bias.bin") !=
            SUCCESS ||
        AddProbeInput(graph, inputTensors, inputs, add1, "scales", colShape, GenScale, outPrefix + "_scales.bin") !=
            SUCCESS) {
        return FAILED;
    }

    add1.set_attr_dtype(static_cast<int64_t>(DT_INT8));
    add1.set_attr_epsilon(epsilon);

    TensorDesc yDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_INT8);
    add1.update_output_desc_y(yDesc);
    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_FLOAT);
    add1.update_output_desc_x(xDesc);

    outputs.push_back(add1);
    return SUCCESS;
}

// Replay one representative case: wire the prepared bins with exact dtypes/shapes/attrs.
int RunCase(const string& caseDir, const string& outPrefix)
{
    CaseCfg cfg;
    if (!ParseCaseCfg(caseDir + "/manifest.txt", cfg)) {
        return FAILED;
    }
    string dimsStr;
    for (size_t i = 0; i < cfg.xDims.size(); i++) {
        dimsStr += std::to_string(cfg.xDims[i]);
        if (i + 1 < cfg.xDims.size()) {
            dimsStr += "x";
        }
    }
    printf("[CASE] case=%s dims=%s xdtype=%s scale=%s zp=%d outx=%d eps=%e\n", cfg.caseName.c_str(), dimsStr.c_str(),
           cfg.xDtype.c_str(), cfg.scaleDtype.c_str(), cfg.hasZp, cfg.outX, cfg.epsilon);

    Graph graph("qaln_case_replay");
    std::vector<ge::Tensor> inputTensors;
    std::vector<Operator> inputs;
    std::vector<Operator> outputs;

    auto add1 = op::QuantizeAddLayerNorm("add1");
    ge::DataType xDt = GeDt(cfg.xDtype);
    ge::DataType sDt = GeDt(cfg.scaleDtype);
    vector<int64_t> colShape = {cfg.cols};
    string p = caseDir + "/bin/";

    if (AddFileInput(graph, inputTensors, inputs, add1, "x1", cfg.xDims, p + "x1.bin", xDt) != SUCCESS ||
        AddFileInput(graph, inputTensors, inputs, add1, "x2", cfg.xDims, p + "x2.bin", xDt) != SUCCESS ||
        AddFileInput(graph, inputTensors, inputs, add1, "gamma", colShape, p + "gamma.bin", xDt) != SUCCESS ||
        AddFileInput(graph, inputTensors, inputs, add1, "beta", colShape, p + "beta.bin", xDt) != SUCCESS ||
        AddFileInput(graph, inputTensors, inputs, add1, "bias", colShape, p + "bias.bin", xDt) != SUCCESS ||
        AddFileInput(graph, inputTensors, inputs, add1, "scales", colShape, p + "scales.bin", sDt) != SUCCESS) {
        return FAILED;
    }
    if (cfg.hasZp != 0 && AddFileInput(graph, inputTensors, inputs, add1, "zero_points", colShape,
                                       p + "zero_points.bin", sDt) != SUCCESS) {
        return FAILED;
    }

    // attributes as recorded in the case metadata (dtype=DT_INT8, axis=-1)
    add1.set_attr_dtype(static_cast<int64_t>(DT_INT8));
    add1.set_attr_axis(static_cast<int64_t>(-1));
    add1.set_attr_epsilon(cfg.epsilon);
    add1.set_attr_additional_output(cfg.outX != 0);

    TensorDesc yDesc = TensorDesc(ge::Shape(cfg.xDims), FORMAT_ND, DT_INT8);
    add1.update_output_desc_y(yDesc);
    if (cfg.outX != 0) {
        TensorDesc xDesc = TensorDesc(ge::Shape(cfg.xDims), FORMAT_ND, xDt);
        add1.update_output_desc_x(xDesc);
    }

    outputs.push_back(add1);
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::Session* session = new Session(std::map<AscendString, AscendString>{});
    if (session == nullptr) {
        printf("[CASE] create session failed\n");
        return FAILED;
    }
    if (session->AddGraph(0, graph, std::map<AscendString, AscendString>{}) != SUCCESS) {
        printf("[CASE] AddGraph failed\n");
        delete session;
        return FAILED;
    }
    std::vector<ge::Tensor> output;
    if (session->RunGraph(0, inputTensors, output) != SUCCESS) {
        printf("[CASE] RunGraph failed\n");
        delete session;
        return FAILED;
    }
    printf("[CASE] RunGraph success, output tensor num=%zu\n", output.size());

    for (size_t i = 0; i < output.size(); i++) {
        int64_t sz = output[i].GetTensorDesc().GetShape().GetShapeSize();
        auto dt = output[i].GetTensorDesc().GetDataType();
        printf("[CASE] output[%zu] dtype=%d elems=%lld\n", i, static_cast<int>(dt), static_cast<long long>(sz));
    }
    // output[0] = y (int8)
    if (!output.empty() && output[0].GetData() != nullptr) {
        int64_t sz = output[0].GetTensorDesc().GetShape().GetShapeSize();
        WriteBin(outPrefix + "_y.bin", static_cast<uint64_t>(sz), output[0].GetData());
    }
    // output[1] = x (residual, input dtype), only when additional_output
    if (cfg.outX != 0 && output.size() > 1 && output[1].GetData() != nullptr) {
        int64_t sz = output[1].GetTensorDesc().GetShape().GetShapeSize();
        uint64_t bytes = static_cast<uint64_t>(sz) * DtItemBytes(output[1].GetTensorDesc().GetDataType());
        WriteBin(outPrefix + "_x.bin", bytes, output[1].GetData());
    }

    delete session;
    printf("[CASE] OK\n");
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    // argv first, env fallback (build.sh --run_example launches the binary without args,
    // so the runbook steers shape via QALN_ROWS / QALN_COLS / QALN_PREFIX, or replays a
    // frozen representative case via QALN_CASE_DIR).
    int64_t rows = gRowsDefault;
    int64_t cols = gColsDefault;
    int useZp = 0;
    float epsilon = 1e-5f;
    string outPrefix = "./qaln_ab";
    const char* envRows = getenv("QALN_ROWS");
    const char* envCols = getenv("QALN_COLS");
    const char* envPrefix = getenv("QALN_PREFIX");
    const char* envCaseDir = getenv("QALN_CASE_DIR");
    if (envRows != nullptr) {
        rows = strtoll(envRows, nullptr, 10);
    }
    if (envCols != nullptr) {
        cols = strtoll(envCols, nullptr, 10);
    }
    if (envPrefix != nullptr) {
        outPrefix = envPrefix;
    }
    if (argc > 1) {
        rows = strtoll(argv[1], nullptr, 10);
    }
    if (argc > 2) {
        cols = strtoll(argv[2], nullptr, 10);
    }
    if (argc > 3) {
        useZp = atoi(argv[3]);
    }
    if (argc > 4) {
        epsilon = strtof(argv[4], nullptr);
    }
    if (argc > 5) {
        outPrefix = argv[5];
    }
    if (envCaseDir != nullptr) {
        printf("[CASE] replay mode: caseDir=%s prefix=%s\n", envCaseDir, outPrefix.c_str());
    } else {
        printf("[AB] rows=%lld cols=%lld zp=%d eps=%e prefix=%s\n", static_cast<long long>(rows),
               static_cast<long long>(cols), useZp, epsilon, outPrefix.c_str());
    }

    const char* graphName = "qaln_mask_ab";
    Graph graph(graphName);
    std::vector<ge::Tensor> inputTensors;
    std::vector<Operator> inputs;
    std::vector<Operator> outputs;

    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("[AB] GEInitialize failed\n");
        return FAILED;
    }

    int ret = FAILED;
    if (envCaseDir != nullptr) {
        ret = RunCase(string(envCaseDir), outPrefix);
        if (ret != SUCCESS) {
            printf("[CASE] RunCase failed\n");
            GEFinalize();
            return FAILED;
        }
        GEFinalize();
        return SUCCESS;
    }

    if (useZp != 0) {
        // zero_points arm intentionally not wired: the A/B mirrors bb_l2_941 (no zero_points);
        // replay mode (QALN_CASE_DIR) covers zero_points cases from the package inputs.
        printf("[AB] zp=1 arm not supported by generate mode, use zp=0 or QALN_CASE_DIR\n");
        GEFinalize();
        return FAILED;
    }
    ret = CreateOppInGraph(rows, cols, epsilon, outPrefix, inputTensors, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("[AB] CreateOppInGraph failed\n");
        GEFinalize();
        return FAILED;
    }

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::Session* session = new Session(std::map<AscendString, AscendString>{});
    if (session == nullptr) {
        printf("[AB] create session failed\n");
        GEFinalize();
        return FAILED;
    }
    if (session->AddGraph(0, graph, std::map<AscendString, AscendString>{}) != SUCCESS) {
        printf("[AB] AddGraph failed\n");
        delete session;
        GEFinalize();
        return FAILED;
    }
    std::vector<ge::Tensor> output;
    if (session->RunGraph(0, inputTensors, output) != SUCCESS) {
        printf("[AB] RunGraph failed\n");
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("[AB] RunGraph success, output tensor num=%zu\n", output.size());

    for (size_t i = 0; i < output.size(); i++) {
        int64_t sz = output[i].GetTensorDesc().GetShape().GetShapeSize();
        auto dt = output[i].GetTensorDesc().GetDataType();
        printf("[AB] output[%zu] dtype=%d elems=%lld\n", i, static_cast<int>(dt), static_cast<long long>(sz));
    }
    // y is the int8 output: find it and dump
    for (size_t i = 0; i < output.size(); i++) {
        if (output[i].GetTensorDesc().GetDataType() == DT_INT8) {
            int64_t sz = output[i].GetTensorDesc().GetShape().GetShapeSize();
            const uint8_t* data = output[i].GetData();
            if (data != nullptr && WriteBin(outPrefix + "_y.bin", static_cast<uint64_t>(sz), data) == SUCCESS) {
                printf("[AB] OK\n");
            }
            break;
        }
    }

    delete session;
    GEFinalize();
    return SUCCESS;
}
