/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_QUANT_BATCH_MATMUL_V3_UTILS_H
#define TEST_QUANT_BATCH_MATMUL_V3_UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <limits.h>
#include <tuple>
#include <functional>

#ifdef __CCE_KT_TEST__
#include "quant_batch_matmul_v3_tiling_def.h"
#include "tikicpulib.h"
#include "gtest/gtest.h"
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "quant_batch_matmul_v3_apt.cpp"
#define PARAM_LIST_DEF GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,\
                       GM_ADDR bias, GM_ADDR perTokenScale, GM_ADDR y,\
                       GM_ADDR workspace, GM_ADDR tiling

#define PARAM_LIST x1, x2, scale, offset, bias, perTokenScale, y, workspace, tiling

using QuantBatchMatmulAptFunc = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);

static std::unordered_map<uint64_t, QuantBatchMatmulAptFunc> s_funcMapApt =
    {{260UL, quant_batch_matmul_v3<0, 1, 0, 1>}};
#else
#include "quant_batch_matmul_v3.cpp"
using QuantBatchMatmulFunc = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
struct TupleHash {
    uint64_t operator()(const std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>& tuple_key) const
    {
        const auto& a = std::get<0>(tuple_key);
        const auto& b = std::get<1>(tuple_key);
        const auto& c = std::get<2>(tuple_key);
        const auto& d = std::get<3>(tuple_key);
        uint64_t seed = std::hash<uint64_t>{}(a);
        seed ^= std::hash<uint64_t>{}(b) + 0x9e3779b9UL + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint64_t>{}(c) + 0x9e3779b9UL + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint64_t>{}(d) + 0x9e3779b9UL + (seed << 6) + (seed >> 2);
        return seed;
    }
};
static std::unordered_map<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>, QuantBatchMatmulFunc, TupleHash> funcMap =
    {{{1UL, 2UL, 0UL, 0UL}, &::quant_batch_matmul_v3<0, 2, 0, 0>},
     {{2UL, 2UL, 0UL, 0UL}, &::quant_batch_matmul_v3<2, 2, 0, 0>},
     {{1UL, 1UL, 0UL, 0UL}, &::quant_batch_matmul_v3<1, 1, 0, 0>},
     {{1UL, 1UL, 1UL, 0UL}, &::quant_batch_matmul_v3<1, 1, 1, 0>},
     {{0UL, 0UL, 1UL, 0UL}, &::quant_batch_matmul_v3<0, 0, 1, 0>},
     {{0UL, 0UL, 0UL, 0UL}, &::quant_batch_matmul_v3<0, 0, 0, 0>},
     {{0UL, 2UL, 1UL, 0UL}, &::quant_batch_matmul_v3<0, 2, 1, 0>}};
#endif

using namespace std;

struct QuantBatchMatmulV3TestParam {
    std::string socVersion;
    std::string caseName;
    std::string kernelUtTarget;
    std::string prefix;
    int64_t x1Dim;
    int64_t x2Dim;
    int64_t yDim;
    int64_t batchA;
    int64_t batchB;
    int64_t batchC;
    int64_t m;
    int64_t k;
    int64_t n;
    bool offsetFlag;
    bool pertokenFlag;
    bool biasFlag;
    bool transA;
    bool transB;
    size_t quantMode;
    bool fmapNz;
    bool weightNz;

    // output
    bool result; // false means tiling fail
    uint32_t blockDim;
    // Tiling defined in quant_batch_matmul_v3_tiling_key.h
    uint64_t trans;
    uint64_t kernel_template_type;
    uint64_t pertoken;
    uint64_t option_attr;
    // Tiling Key for A5 csv, default 0 in A2/A3
    uint64_t tilingKey;
    std::string tilingData;
};

class QuantBatchMatmulV3TestUtils {
public:
    static constexpr uint32_t MAX_BLOCK_DIM = 4;
    static void SplitStr2Vec(const string &input, const string &delimiter, vector<string> &output)
    {
        auto delimiterLen = delimiter.size();
        std::string::size_type currPos = 0;
        std::string::size_type nextPos = input.find(delimiter, currPos);
        while (nextPos != std::string::npos) {
            output.emplace_back(input.substr(currPos, nextPos - currPos));
            currPos = nextPos + delimiterLen;
            nextPos = input.find(delimiter, currPos);
        }

        if (currPos < input.size()) {
            output.emplace_back(input.substr(currPos));
        }
    }

    static std::string GetExeDirPath()
    {
        std::string exe_path("./");
        char path[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", path, sizeof(path));
        if (n > 0) {
            path[n] = '\0';
            exe_path.assign(path);
            auto pos = exe_path.find_last_of('/');
            if (pos != std::string::npos) {
                exe_path.erase(pos + 1);
            } else {
                exe_path.assign("./");
            }
        }

        return exe_path;
    }

    static vector<QuantBatchMatmulV3TestParam> GetParams(const string &socVersion, const string &testSuite)
    {
        std::vector<QuantBatchMatmulV3TestParam> params;
        std::string rootPath(GetExeDirPath() + "../../../../");
        std::string casePath(rootPath + "matmul/quant_batch_matmul_v3/tests/ut/op_kernel/test_quant_batch_matmul_v3.csv");
        std::ifstream csvData(casePath, std::ios::in);
        if (!csvData.is_open()) {
            std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
            return params;
        }

        std::string line;
        while (std::getline(csvData, line)) {
            std::vector<std::string> testParam;
            SplitStr2Vec(line, ",", testParam);

            QuantBatchMatmulV3TestParam param;
            size_t idx = 0UL;
            param.socVersion = testParam[idx++];
            param.caseName = testParam[idx++];
            param.kernelUtTarget = testParam[idx++];
            if (param.socVersion != socVersion || param.kernelUtTarget != testSuite) {
                continue;
            }

            param.prefix = testParam[idx++];
            idx++;  // skip coreNum
            param.x1Dim = stol(testParam[idx++]);
            param.x2Dim = stol(testParam[idx++]);
            param.yDim = stol(testParam[idx++]);
            param.batchA = stol(testParam[idx++]);
            param.batchB = stol(testParam[idx++]);
            param.batchC = stol(testParam[idx++]);
            param.m = stol(testParam[idx++]);
            param.k = stol(testParam[idx++]);
            param.n = stol(testParam[idx++]);
            param.offsetFlag = stol(testParam[idx++]);
            param.pertokenFlag = stol(testParam[idx++]);
            param.biasFlag = stol(testParam[idx++]);
            param.transA = stol(testParam[idx++]);
            param.transB = stol(testParam[idx++]);
            param.quantMode = stol(testParam[idx++]);
            idx++;  // skip x1Dtype
            idx++;  // skip x2Dtype
            idx++;  // skip scaleDtype
            idx++;  // skip perTokenScaleDtype
            idx++;  // skip biasDtype
            idx++;  // skip yDtype
            param.fmapNz = testParam[idx++] == "NZ";
            param.weightNz = testParam[idx++] == "NZ";
            param.result = (strcasecmp(testParam[idx++].c_str(), "true") == 0);
            param.blockDim = stol(testParam[idx++]);
            param.trans = stol(testParam[idx++]);
            param.kernel_template_type = stol(testParam[idx++]);
            param.pertoken = stol(testParam[idx++]);
            param.option_attr = stol(testParam[idx++]);
            param.tilingKey = stol(testParam[idx++]);
            param.tilingData = testParam[idx++];
            params.push_back(param);
        }

        return params;
    }
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    static void TestOneParamCase(const QuantBatchMatmulV3TestParam &param)
    {
        int64_t batchA = param.batchA > 0 ? param.batchA : 1L;
        int64_t batchB = param.batchB > 0 ? param.batchB : 1L;
        int64_t batchC = param.batchC > 0 ? param.batchC : 1L;
        size_t shape_x1 = batchA * param.m * param.k * sizeof(int8_t);
        size_t shape_x2 = batchB * param.k * param.n * sizeof(int8_t);
        size_t shape_bias = batchC * param.n * sizeof(int32_t);
        size_t shape_scale = param.n * param.k * sizeof(uint64_t);
        size_t shape_pertoken = param.m * param.k * sizeof(uint32_t);
        size_t shape_y = batchC * param.m * param.n * sizeof(uint32_t);
        size_t tiling_data_size = sizeof(QuantBatchMatmulV3TilingData);
        size_t workspace_size = 16 * 1024 * 1024 + 50 * 1024 * 1024;
        uint8_t *x1 = (uint8_t *)AscendC::GmAlloc(shape_x1);
        uint8_t *x2 = (uint8_t *)AscendC::GmAlloc(shape_x2);
        uint8_t *bias = (uint8_t *)AscendC::GmAlloc(shape_bias);
        uint8_t *offset = nullptr;
        uint8_t *scale = (uint8_t *)AscendC::GmAlloc(shape_scale);
        uint8_t *pertokenScale = (uint8_t *)AscendC::GmAlloc(shape_pertoken);
        uint8_t *y = (uint8_t *)AscendC::GmAlloc(shape_y);
        uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_size);
        uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

        memset(x1, 1, shape_x1);
        memset(x2, 1, shape_x2);
        memset(bias, 1, shape_bias);
        memset(scale, 1, shape_scale);
        memset(pertokenScale, 1, shape_pertoken);
        memset(workspace, 0, workspace_size);

        std::vector<std::string> tilingDataStr;
        SplitStr2Vec(param.tilingData, " ", tilingDataStr);
        std::vector<int32_t> tilingDataInt;
        tilingDataInt.reserve(tilingDataStr.size());
        for (auto &tilingValue : tilingDataStr) {
            tilingDataInt.push_back(atoi(tilingValue.c_str()));
        }

        ASSERT_EQ(tilingDataInt.size() * sizeof(int32_t), tiling_data_size);
        memcpy(tiling, tilingDataInt.data(), tiling_data_size);

        if (param.kernelUtTarget == "quant_batch_matmul_v3_int32") {
            AscendC::SetKernelMode(KernelMode::AIC_MODE);
        } else {
            AscendC::SetKernelMode(KernelMode::MIX_MODE);
        }

        auto wrapper = [param](
                           GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset, GM_ADDR bias, GM_ADDR pertokenScale,
                           GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
            auto key = std::make_tuple(param.trans, param.kernel_template_type, param.pertoken, param.option_attr);
            auto it = funcMap.find(key);
            if (it != funcMap.end()) {
                it->second(x1, x2, scale, offset, bias, pertokenScale, y, workSpace, tiling);
            } else {
                throw std::runtime_error("Unsupported parameter combination.");
            }
        };

        ICPU_RUN_KF(
            wrapper, std::min(MAX_BLOCK_DIM, param.blockDim), x1, x2, scale, offset, bias, pertokenScale, y, workspace,
            tiling);

        AscendC::GmFree(x1);
        AscendC::GmFree(x2);
        AscendC::GmFree(scale);
        AscendC::GmFree(bias);
        AscendC::GmFree(pertokenScale);
        AscendC::GmFree(y);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
    #endif
    #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // 实践中发现此处如果不传入s_funcMapApt会导致符号合并等编译器行为问题
    // 必须传入s_funcMapApt以实现多数据流
    static void TestOneParamCase910_95(const QuantBatchMatmulV3TestParam &param,
                                        decltype(s_funcMapApt)& funcMapApt) {
        std::function<void(PARAM_LIST_DEF)> func = [&param, &funcMapApt](PARAM_LIST_DEF){
            auto key = param.tilingKey;
            auto it = funcMapApt.find(key);
            if (it != funcMapApt.end()) {
                it->second(PARAM_LIST);
            } else {
                throw std::runtime_error("Unsupported tilingKey.");
            }
        };
        auto M = param.m;
        auto K = param.k;
        auto N = param.n;
        auto batchA = param.batchA;
        auto batchB = param.batchB;
        auto batchC = param.batchC;
        auto biasFlag = param.biasFlag;
        auto offsetFlag = param.offsetFlag;
        auto pertokenFlag = param.pertokenFlag;
        auto blockDim = param.blockDim;
        size_t shape_x1 = batchA * M * K * sizeof(int8_t);
        size_t shape_x2 = batchB * K * N * sizeof(int8_t);
        size_t shape_scale = N * sizeof(uint64_t);
        size_t shape_y = batchC * M * N * sizeof(uint32_t);
        size_t tiling_data_size = sizeof(DequantBmm::QuantBatchMatmulV3TilingDataParams);
        size_t workspace_size = 16 * 1024 * 1024 + 50 * 1024 * 1024;
        uint8_t *x1 = (uint8_t *)AscendC::GmAlloc(shape_x1);
        uint8_t *x2 = (uint8_t *)AscendC::GmAlloc(shape_x2);
        uint8_t *scale = (uint8_t *)AscendC::GmAlloc(shape_scale);
        uint8_t *y = (uint8_t *)AscendC::GmAlloc(shape_y);
        uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_size);
        uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
        uint8_t *bias = nullptr;
        uint8_t *offset = nullptr;
        uint8_t *pertokenScale = nullptr;
        
        memset(x1, 1, shape_x1);
        memset(x2, 1, shape_x2);
        memset(scale, 1, shape_scale);
        memset(y, 0, shape_y);
        memset(workspace, 0, workspace_size);

        std::vector<std::string> tilingDataStr;
        QuantBatchMatmulV3TestUtils::SplitStr2Vec(param.tilingData, " ", tilingDataStr);
        std::vector<int32_t> tilingDataInt;
        tilingDataInt.reserve(tilingDataStr.size());
        for (auto &tilingValue : tilingDataStr) {
            tilingDataInt.push_back(atoi(tilingValue.c_str()));
        }
        ASSERT_EQ(tilingDataInt.size() * sizeof(int32_t), tiling_data_size);
        memcpy(tiling, tilingDataInt.data(), tiling_data_size);

        ICPU_RUN_KF(func, std::min(QuantBatchMatmulV3TestUtils::MAX_BLOCK_DIM, param.blockDim), x1, x2, scale, offset, bias, pertokenScale, y, workspace, tiling);
        AscendC::GmFree((void *)x1);
        AscendC::GmFree((void *)x2);
        AscendC::GmFree((void *)scale);
        if (biasFlag) AscendC::GmFree((void *)bias);
        if (pertokenFlag) AscendC::GmFree((void *)pertokenScale);
        if (offsetFlag) AscendC::GmFree((void *)offset);
        AscendC::GmFree((void *)y);
        AscendC::GmFree((void *)workspace);
        AscendC::GmFree((void *)tiling);
    }
    #endif
};

#endif