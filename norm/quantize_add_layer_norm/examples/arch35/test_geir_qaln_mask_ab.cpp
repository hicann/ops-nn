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
 * Usage: ./test_geir_qaln_mask_ab <rows> <cols> <zp 0> <epsilon> <outPrefix>
 */

#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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

#include "../../op_graph/quantize_add_layer_norm_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int64_t DEFAULT_ROWS = 672;
constexpr int64_t DEFAULT_COLS = 224;
constexpr size_t MAX_MANIFEST_LINE_LENGTH = 4096;

bool CheckedMultiply(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs <= 0 || rhs <= 0 || rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool GetElementCount(const vector<int64_t>& shape, int64_t& elementCount)
{
    elementCount = 1;
    for (int64_t dim : shape) {
        int64_t next = 0;
        if (!CheckedMultiply(elementCount, dim, next)) {
            return false;
        }
        elementCount = next;
    }
    return true;
}

bool ParseInt64(const string& text, int64_t& value)
{
    errno = 0;
    char* end = nullptr;
    long long parsed = strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || end == nullptr || *end != '\0') {
        return false;
    }
    value = static_cast<int64_t>(parsed);
    return true;
}

bool ParseFlag(const string& text, int& value)
{
    int64_t parsed = 0;
    if (!ParseInt64(text, parsed) || (parsed != 0 && parsed != 1)) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool ParsePositiveFloat(const string& text, float& value)
{
    errno = 0;
    char* end = nullptr;
    float parsed = strtof(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || end == nullptr || *end != '\0' || !std::isfinite(parsed) ||
        parsed <= 0.0F) {
        return false;
    }
    value = parsed;
    return true;
}

int32_t WriteBin(const string& path, uint64_t bytes, const uint8_t* data)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        printf("open %s for write failed\n", path.c_str());
        return FAILED;
    }
    size_t written = fwrite(data, sizeof(uint8_t), bytes, fp);
    int closeStatus = fclose(fp);
    if (written != bytes || closeStatus != 0) {
        printf("write %s failed\n", path.c_str());
        return FAILED;
    }
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
// Negative per-channel division scales, mirroring bb_l2_941: in [-2.125, -0.125].
float GenScale(int64_t c) { return -(0.125f * static_cast<float>(1 + (c % 16))); }

int32_t GenFp32Tensor(int64_t elemNum, float (*gen)(int64_t), const string& dumpPath, TensorDesc& desc, Tensor& tensor)
{
    if (elemNum <= 0 || static_cast<uint64_t>(elemNum) > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return FAILED;
    }
    desc.SetRealDimCnt(1);
    uint64_t dataLen = static_cast<uint64_t>(elemNum) * sizeof(float);
    std::vector<float> data;
    try {
        data.resize(static_cast<size_t>(elemNum));
    } catch (const std::bad_alloc&) {
        printf("[AB] allocation failed for %lld fp32 elements\n", static_cast<long long>(elemNum));
        return FAILED;
    }
    for (int64_t i = 0; i < elemNum; i++) {
        data[static_cast<size_t>(i)] = gen(i);
    }
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(data.data()), dataLen);
    if (!dumpPath.empty() && WriteBin(dumpPath, dataLen, reinterpret_cast<uint8_t*>(data.data())) != SUCCESS) {
        return FAILED;
    }
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
    std::ifstream manifest(path);
    if (!manifest.is_open()) {
        printf("[CASE] cannot open %s\n", path.c_str());
        return false;
    }
    bool hasCase = false;
    bool hasDims = false;
    bool hasXDtype = false;
    bool hasScaleDtype = false;
    bool hasZp = false;
    bool hasOutX = false;
    bool hasEpsilon = false;
    string manifestLine;
    while (std::getline(manifest, manifestLine)) {
        if (manifestLine.size() > MAX_MANIFEST_LINE_LENGTH) {
            printf("[CASE] manifest line is too long\n");
            return false;
        }
        while (!manifestLine.empty() && manifestLine.back() == '\r') {
            manifestLine.pop_back();
        }
        size_t eq = manifestLine.find('=');
        if (eq == string::npos) {
            continue;
        }
        string key = manifestLine.substr(0, eq);
        string value = manifestLine.substr(eq + 1);
        if (key == "case") {
            cfg.caseName = value;
            hasCase = true;
        } else if (key == "dims") {
            cfg.xDims.clear();
            size_t pos = 0;
            while (pos < value.size()) {
                size_t comma = value.find(',', pos);
                string tok = (comma == string::npos) ? value.substr(pos) : value.substr(pos, comma - pos);
                int64_t dim = 0;
                if (tok.empty() || !ParseInt64(tok, dim) || dim <= 0) {
                    printf("[CASE] invalid dimension: %s\n", tok.c_str());
                    return false;
                }
                cfg.xDims.push_back(dim);
                if (comma == string::npos) {
                    break;
                }
                pos = comma + 1;
            }
            hasDims = true;
        } else if (key == "xdtype") {
            cfg.xDtype = value;
            hasXDtype = true;
        } else if (key == "scaledtype") {
            cfg.scaleDtype = value;
            hasScaleDtype = true;
        } else if (key == "zp") {
            if (!ParseFlag(value, cfg.hasZp)) {
                printf("[CASE] invalid zp flag: %s\n", value.c_str());
                return false;
            }
            hasZp = true;
        } else if (key == "outx") {
            if (!ParseFlag(value, cfg.outX)) {
                printf("[CASE] invalid outx flag: %s\n", value.c_str());
                return false;
            }
            hasOutX = true;
        } else if (key == "epsilon") {
            if (!ParsePositiveFloat(value, cfg.epsilon)) {
                printf("[CASE] invalid epsilon: %s\n", value.c_str());
                return false;
            }
            hasEpsilon = true;
        }
    }
    if (manifest.bad()) {
        printf("[CASE] failed while reading manifest\n");
        return false;
    }
    if (!hasCase || !hasDims || !hasXDtype || !hasScaleDtype || !hasZp || !hasOutX || !hasEpsilon ||
        cfg.caseName.empty() || cfg.xDims.empty()) {
        printf("[CASE] manifest is missing required fields\n");
        return false;
    }
    int64_t elementCount = 0;
    bool dtypePairSupported = ((cfg.xDtype == "fp16" || cfg.xDtype == "fp32") && cfg.scaleDtype == "fp32") ||
                              (cfg.xDtype == "bf16" && (cfg.scaleDtype == "bf16" || cfg.scaleDtype == "fp32"));
    if (!GetElementCount(cfg.xDims, elementCount) ||
        (cfg.xDtype != "fp16" && cfg.xDtype != "bf16" && cfg.xDtype != "fp32") || !dtypePairSupported ||
        !std::isfinite(cfg.epsilon) || cfg.epsilon <= 0.0F) {
        printf("[CASE] manifest contains unsupported or unsafe values\n");
        return false;
    }
    cfg.cols = cfg.xDims.back();
    return true;
}

std::unique_ptr<uint8_t[]> ReadBin(const string& path, uint64_t expectedBytes)
{
    if (expectedBytes == 0 || expectedBytes > std::numeric_limits<size_t>::max()) {
        return {};
    }
    FILE* fp = fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        printf("[CASE] cannot open %s\n", path.c_str());
        return {};
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        printf("[CASE] cannot seek %s\n", path.c_str());
        fclose(fp);
        return {};
    }
    long sz = ftell(fp);
    if (sz <= 0 || static_cast<uint64_t>(sz) != expectedBytes || fseek(fp, 0, SEEK_SET) != 0) {
        if (sz <= 0) {
            printf("[CASE] cannot determine nonzero size for %s\n", path.c_str());
        } else if (static_cast<uint64_t>(sz) != expectedBytes) {
            printf("[CASE] %s size mismatch: file %lld bytes, expect %llu\n", path.c_str(), static_cast<long long>(sz),
                   static_cast<unsigned long long>(expectedBytes));
        } else {
            printf("[CASE] cannot rewind %s\n", path.c_str());
        }
        fclose(fp);
        return {};
    }
    size_t bytesToRead = static_cast<size_t>(expectedBytes);
    auto buffer = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[bytesToRead]);
    if (buffer == nullptr) {
        printf("[CASE] allocation failed while reading %s (%zu bytes)\n", path.c_str(), bytesToRead);
        fclose(fp);
        return {};
    }
    if (fread(buffer.get(), 1, bytesToRead, fp) != bytesToRead) {
        printf("[CASE] failed to read %s (%zu bytes)\n", path.c_str(), bytesToRead);
        fclose(fp);
        return {};
    }
    if (fclose(fp) != 0) {
        return {};
    }
    return buffer;
}

Status AddFileInput(const string& name, const vector<int64_t>& shape, const string& path, ge::DataType dt, Graph& graph,
                    std::vector<ge::Tensor>& inputTensors, std::vector<Operator>& inputs,
                    op::QuantizeAddLayerNorm& add1)
{
    int64_t elem = 1;
    uint32_t itemBytes = DtItemBytes(dt);
    if (!GetElementCount(shape, elem) ||
        static_cast<uint64_t>(elem) > std::numeric_limits<uint64_t>::max() / itemBytes) {
        return FAILED;
    }
    uint64_t expect = static_cast<uint64_t>(elem) * itemBytes;
    auto data = ReadBin(path, expect);
    if (data == nullptr) {
        return FAILED;
    }
    auto dataOp = op::Data("file_" + name).set_attr_index(0);
    TensorDesc desc = TensorDesc(ge::Shape(shape), FORMAT_ND, dt);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    dataOp.update_input_desc_x(desc);
    dataOp.update_output_desc_y(desc);
    inputTensors.push_back(Tensor(desc, data.get(), expect));
    if (graph.AddOp(dataOp) != ge::GRAPH_SUCCESS) {
        printf("[CASE] failed to add input %s to graph\n", name.c_str());
        return FAILED;
    }
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
    return SUCCESS;
}
} // namespace

static Status AddProbeInput(const string& name, const vector<int64_t>& shape, float (*gen)(int64_t),
                            const string& dumpPath, Graph& graph, std::vector<ge::Tensor>& inputTensors,
                            std::vector<Operator>& inputs, op::QuantizeAddLayerNorm& add1)
{
    auto data = op::Data("placeholder_" + name).set_attr_index(0);
    TensorDesc desc = TensorDesc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    Tensor tensor;
    int64_t elem = 1;
    if (!GetElementCount(shape, elem)) {
        return FAILED;
    }
    TensorDesc genDesc = desc;
    if (GenFp32Tensor(elem, gen, dumpPath, genDesc, tensor) != SUCCESS) {
        printf("gen tensor %s failed\n", name.c_str());
        return FAILED;
    }
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    inputTensors.push_back(tensor);
    if (graph.AddOp(data) != ge::GRAPH_SUCCESS) {
        printf("failed to add tensor %s to graph\n", name.c_str());
        return FAILED;
    }
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

    if (AddProbeInput("x1", xShape, GenX1, outPrefix + "_x1.bin", graph, inputTensors, inputs, add1) != SUCCESS ||
        AddProbeInput("x2", xShape, GenX2, outPrefix + "_x2.bin", graph, inputTensors, inputs, add1) != SUCCESS ||
        AddProbeInput("gamma", colShape, GenGamma, outPrefix + "_gamma.bin", graph, inputTensors, inputs, add1) !=
            SUCCESS ||
        AddProbeInput("beta", colShape, GenBeta, outPrefix + "_beta.bin", graph, inputTensors, inputs, add1) !=
            SUCCESS ||
        AddProbeInput("bias", colShape, GenBias, outPrefix + "_bias.bin", graph, inputTensors, inputs, add1) !=
            SUCCESS ||
        AddProbeInput("scales", colShape, GenScale, outPrefix + "_scales.bin", graph, inputTensors, inputs, add1) !=
            SUCCESS) {
        return FAILED;
    }

    add1.set_attr_dtype(static_cast<int64_t>(DT_INT8));
    add1.set_attr_epsilon(epsilon);

    TensorDesc yDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_INT8);
    add1.update_output_desc_y(yDesc);
    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_FLOAT);
    add1.update_output_desc_x(xDesc);

    if (graph.AddOp(add1) != ge::GRAPH_SUCCESS) {
        printf("failed to add QuantizeAddLayerNorm to graph\n");
        return FAILED;
    }
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
    string inputBinPrefix = caseDir + "/bin/";

    if (AddFileInput("x1", cfg.xDims, inputBinPrefix + "x1.bin", xDt, graph, inputTensors, inputs, add1) != SUCCESS ||
        AddFileInput("x2", cfg.xDims, inputBinPrefix + "x2.bin", xDt, graph, inputTensors, inputs, add1) != SUCCESS ||
        AddFileInput("gamma", colShape, inputBinPrefix + "gamma.bin", xDt, graph, inputTensors, inputs, add1) !=
            SUCCESS ||
        AddFileInput("beta", colShape, inputBinPrefix + "beta.bin", xDt, graph, inputTensors, inputs, add1) !=
            SUCCESS ||
        AddFileInput("bias", colShape, inputBinPrefix + "bias.bin", xDt, graph, inputTensors, inputs, add1) !=
            SUCCESS ||
        AddFileInput("scales", colShape, inputBinPrefix + "scales.bin", sDt, graph, inputTensors, inputs, add1) !=
            SUCCESS) {
        return FAILED;
    }
    if (cfg.hasZp != 0 && AddFileInput("zero_points", colShape, inputBinPrefix + "zero_points.bin", sDt, graph,
                                       inputTensors, inputs, add1) != SUCCESS) {
        return FAILED;
    }

    // Apply attributes recorded in the case metadata (dtype=DT_INT8, axis=-1).
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

    if (graph.AddOp(add1) != ge::GRAPH_SUCCESS) {
        printf("[CASE] failed to add QuantizeAddLayerNorm to graph\n");
        return FAILED;
    }
    outputs.push_back(add1);
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::Session session(std::map<AscendString, AscendString>{});
    if (session.AddGraph(0, graph, std::map<AscendString, AscendString>{}) != SUCCESS) {
        printf("[CASE] AddGraph failed\n");
        return FAILED;
    }
    std::vector<ge::Tensor> output;
    if (session.RunGraph(0, inputTensors, output) != SUCCESS) {
        printf("[CASE] RunGraph failed\n");
        return FAILED;
    }
    printf("[CASE] RunGraph success, output tensor num=%zu\n", output.size());

    for (size_t i = 0; i < output.size(); i++) {
        int64_t sz = output[i].GetTensorDesc().GetShape().GetShapeSize();
        auto dt = output[i].GetTensorDesc().GetDataType();
        printf("[CASE] output[%zu] dtype=%d elems=%lld\n", i, static_cast<int>(dt), static_cast<long long>(sz));
    }
    if (output.empty() || output[0].GetTensorDesc().GetDataType() != DT_INT8 || output[0].GetData() == nullptr) {
        printf("[CASE] required INT8 y output is missing\n");
        return FAILED;
    }
    int64_t ySize = output[0].GetTensorDesc().GetShape().GetShapeSize();
    if (ySize <= 0 || WriteBin(outPrefix + "_y.bin", static_cast<uint64_t>(ySize), output[0].GetData()) != SUCCESS) {
        return FAILED;
    }

    if (cfg.outX != 0) {
        if (output.size() < 2 || output[1].GetTensorDesc().GetDataType() != GeDt(cfg.xDtype) ||
            output[1].GetData() == nullptr) {
            printf("[CASE] required x output is missing or has the wrong dtype\n");
            return FAILED;
        }
        int64_t sz = output[1].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t itemBytes = DtItemBytes(output[1].GetTensorDesc().GetDataType());
        if (sz <= 0 || static_cast<uint64_t>(sz) > std::numeric_limits<uint64_t>::max() / itemBytes) {
            return FAILED;
        }
        uint64_t bytes = static_cast<uint64_t>(sz) * itemBytes;
        if (WriteBin(outPrefix + "_x.bin", bytes, output[1].GetData()) != SUCCESS) {
            return FAILED;
        }
    }

    printf("[CASE] OK\n");
    return SUCCESS;
}

int RunGeneratedCase(int64_t rows, int64_t cols, int64_t expectedElements, float epsilon, const string& outPrefix)
{
    Graph graph("qaln_mask_ab");
    std::vector<ge::Tensor> inputTensors;
    std::vector<Operator> inputs;
    std::vector<Operator> outputs;

    if (CreateOppInGraph(rows, cols, epsilon, outPrefix, inputTensors, inputs, outputs, graph) != SUCCESS) {
        printf("[AB] CreateOppInGraph failed\n");
        return FAILED;
    }

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::Session session(std::map<AscendString, AscendString>{});
    if (session.AddGraph(0, graph, std::map<AscendString, AscendString>{}) != SUCCESS) {
        printf("[AB] AddGraph failed\n");
        return FAILED;
    }
    std::vector<ge::Tensor> output;
    if (session.RunGraph(0, inputTensors, output) != SUCCESS) {
        printf("[AB] RunGraph failed\n");
        return FAILED;
    }
    printf("[AB] RunGraph success, output tensor num=%zu\n", output.size());

    for (size_t i = 0; i < output.size(); i++) {
        int64_t size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        auto dataType = output[i].GetTensorDesc().GetDataType();
        printf("[AB] output[%zu] dtype=%d elems=%lld\n", i, static_cast<int>(dataType), static_cast<long long>(size));
    }
    if (output.empty() || output[0].GetTensorDesc().GetDataType() != DT_INT8 ||
        output[0].GetTensorDesc().GetShape().GetShapeSize() != expectedElements || output[0].GetData() == nullptr ||
        WriteBin(outPrefix + "_y.bin", static_cast<uint64_t>(expectedElements), output[0].GetData()) != SUCCESS) {
        printf("[AB] required output[0] INT8 y is unavailable or could not be written\n");
        return FAILED;
    }

    printf("[AB] OK\n");
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    // argv first, env fallback (build.sh --run_example launches the binary without args,
    // so the runbook steers shape via QALN_ROWS / QALN_COLS / QALN_PREFIX, or replays a
    // frozen representative case via QALN_CASE_DIR).
    int64_t rows = DEFAULT_ROWS;
    int64_t cols = DEFAULT_COLS;
    int useZp = 0;
    float epsilon = 1e-5f;
    string outPrefix = "./qaln_ab";
    const char* envRows = getenv("QALN_ROWS");
    const char* envCols = getenv("QALN_COLS");
    const char* envPrefix = getenv("QALN_PREFIX");
    const char* envCaseDir = getenv("QALN_CASE_DIR");
    if (envCaseDir != nullptr && (envRows != nullptr || envCols != nullptr || argc > 1)) {
        printf("[CASE] replay mode reads shape, zero_points, and epsilon from manifest; do not pass generate-mode "
               "arguments or QALN_ROWS/QALN_COLS\n");
        return FAILED;
    }
    if (envRows != nullptr && !ParseInt64(envRows, rows)) {
        printf("[AB] invalid QALN_ROWS\n");
        return FAILED;
    }
    if (envCols != nullptr && !ParseInt64(envCols, cols)) {
        printf("[AB] invalid QALN_COLS\n");
        return FAILED;
    }
    if (envPrefix != nullptr) {
        outPrefix = envPrefix;
    }
    if (argc > 1 && !ParseInt64(argv[1], rows)) {
        printf("[AB] invalid rows argument\n");
        return FAILED;
    }
    if (argc > 2 && !ParseInt64(argv[2], cols)) {
        printf("[AB] invalid cols argument\n");
        return FAILED;
    }
    if (argc > 3 && !ParseFlag(argv[3], useZp)) {
        printf("[AB] invalid zero_points flag\n");
        return FAILED;
    }
    if (argc > 4 && !ParsePositiveFloat(argv[4], epsilon)) {
        printf("[AB] invalid epsilon argument\n");
        return FAILED;
    }
    if (argc > 5) {
        outPrefix = argv[5];
    }
    int64_t expectedElements = 0;
    if (!CheckedMultiply(rows, cols, expectedElements) || !std::isfinite(epsilon) || epsilon <= 0.0F ||
        (useZp != 0 && useZp != 1) || outPrefix.empty()) {
        printf("[AB] rows, cols, epsilon, zero_points flag, or output prefix is invalid\n");
        return FAILED;
    }
    if (envCaseDir != nullptr) {
        printf("[CASE] replay mode: caseDir=%s prefix=%s\n", envCaseDir, outPrefix.c_str());
    } else {
        printf("[AB] rows=%lld cols=%lld zp=%d eps=%e prefix=%s\n", static_cast<long long>(rows),
               static_cast<long long>(cols), useZp, epsilon, outPrefix.c_str());
    }

    if (envCaseDir == nullptr && useZp != 0) {
        // Generate mode mirrors bb_l2_941, which has no zero_points; replay mode covers offset cases.
        printf("[AB] zp=1 arm not supported by generate mode, use zp=0 or QALN_CASE_DIR\n");
        return FAILED;
    }

    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("[AB] GEInitialize failed\n");
        return FAILED;
    }

    int runStatus = envCaseDir != nullptr ? RunCase(string(envCaseDir), outPrefix) :
                                            RunGeneratedCase(rows, cols, expectedElements, epsilon, outPrefix);
    Status finalizeStatus = GEFinalize();
    if (finalizeStatus != SUCCESS) {
        printf("[AB] GEFinalize failed\n");
        return FAILED;
    }
    if (runStatus != SUCCESS) {
        printf("[AB] graph execution failed\n");
        return FAILED;
    }
    return SUCCESS;
}
