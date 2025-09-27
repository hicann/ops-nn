/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_QUANT_BATCH_MATMUL_V3_UTILS_H
#define TEST_QUANT_BATCH_MATMUL_V3_UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <limits.h>

#include "quant_batch_matmul_v3_tiling_def.h"
#include "tikicpulib.h"
#include "gtest/gtest.h"

using namespace std;

extern "C" __global__ __aicore__ void quant_batch_matmul_v3(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,
                                                            GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
                                                            GM_ADDR workSpace, GM_ADDR tiling);

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
        std::string rootPath(GetExeDirPath() + "../../../../../../../");
        std::string casePath(rootPath + "ops/matmul/quant_batch_matmul_v3/tests/ut/op_host/test_quant_batch_matmul_v3.csv");
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
            param.tilingKey = stol(testParam[idx++]);
            param.tilingData = testParam[idx++];
            params.push_back(param);
        }

        return params;
    }

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
            AscendC::SetKernelMode(KernelMode::MIX_AIC_1_1);
        }
        ICPU_SET_TILING_KEY(param.tilingKey);
        ICPU_RUN_KF(quant_batch_matmul_v3, std::min(MAX_BLOCK_DIM, param.blockDim), x1, x2, scale, offset, bias,
                    pertokenScale, y, workspace, tiling);

        AscendC::GmFree(x1);
        AscendC::GmFree(x2);
        AscendC::GmFree(scale);
        AscendC::GmFree(bias);
        AscendC::GmFree(pertokenScale);
        AscendC::GmFree(y);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

#endif