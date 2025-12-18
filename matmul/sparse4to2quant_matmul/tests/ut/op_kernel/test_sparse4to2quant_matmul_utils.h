/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_SPARSE4TO2_QUANT_MATMUL_UTILS_H
#define TEST_SPARSE4TO2_QUANT_MATMUL_UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <limits.h>

#include "./sparse4to2quant_matmul_tiling_def.h"
#include "tikicpulib.h"
#include "gtest/gtest.h"

using namespace std;

extern "C" __global__ __aicore__ void sparse4to2quant_matmul(GM_ADDR x, GM_ADDR sparseWeight, GM_ADDR index,
                                                             GM_ADDR xScale, GM_ADDR sparseWeightScale, GM_ADDR bias,GM_ADDR y,
                                                             GM_ADDR workSpace, GM_ADDR tiling);

struct Sparse4to2QuantMatmulTestParam {
    std::string socVersion;
    std::string caseName;
    std::string kernelUtTarget;
    std::string prefix;
    int64_t x1Dim;
    int64_t x2Dim;
    int64_t yDim;
    int64_t m;
    int64_t k;
    int64_t n;
    bool pertokenFlag;
    bool biasFlag;
    size_t quantMode;
    // output
    bool result; // false means tiling fail
    uint32_t blockDim;
    uint64_t tilingKey;
    std::string tilingData;
};

class Sparse4to2QuantMatmulTestUtils {
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

    static vector<Sparse4to2QuantMatmulTestParam> GetParams(const string &socVersion, const string &testSuite)
    {
        std::vector<Sparse4to2QuantMatmulTestParam> params;
        std::string rootPath(GetExeDirPath() + "../../../../");
        std::string casePath(rootPath + "matmul/sparse4to2quant_matmul/tests/ut/op_kernel/test_sparse4to2quant_matmul.csv");
        std::ifstream csvData(casePath, std::ios::in);
        if (!csvData.is_open()) {
            std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
            return params;
        }

        std::string line;
        while (std::getline(csvData, line)) {
            std::vector<std::string> testParam;
            SplitStr2Vec(line, ",", testParam);

            Sparse4to2QuantMatmulTestParam param;
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
            param.m = stol(testParam[idx++]);
            param.k = stol(testParam[idx++]);
            param.n = stol(testParam[idx++]);
            param.pertokenFlag = stol(testParam[idx++]);
            param.biasFlag = stol(testParam[idx++]);
            param.quantMode = stol(testParam[idx++]);
            idx++;  // skip x1Dtype
            idx++;  // skip x2Dtype
            idx++;  // skip scaleDtype
            idx++;  // skip perTokenScaleDtype
            idx++;  // skip biasDtype
            idx++;  // skip yDtype
            idx++;  // skip format
            idx++;  // skip wformat
            param.result = (strcasecmp(testParam[idx++].c_str(), "true") == 0);
            param.blockDim = stol(testParam[idx++]);
            param.tilingKey = stol(testParam[idx++]);
            param.tilingData = testParam[idx++];
            params.push_back(param);
        }

        return params;
    }

    static void TestOneParamCase(const Sparse4to2QuantMatmulTestParam &param)
    {
        size_t shape_x1 = param.m * param.k * sizeof(int8_t);
        size_t shape_x2 = param.n * param.k * sizeof(int8_t);
        size_t shape_index = shape_x2 / 4;
        size_t shape_bias = param.n * sizeof(int32_t);
        size_t shape_scale = param.n * param.k * sizeof(uint64_t);
        size_t shape_pertoken = param.m * param.k * sizeof(uint32_t);
        size_t shape_y = param.m * param.n * sizeof(uint32_t);
        size_t tiling_data_size = sizeof(SparseQmm::Sparse4to2QuantMatmulTilingData);
        size_t workspace_size = 16 * 1024 * 1024 + 50 * 1024 * 1024;
        uint8_t *x1 = (uint8_t *)AscendC::GmAlloc(shape_x1);
        uint8_t *x2 = (uint8_t *)AscendC::GmAlloc(shape_x2);
        uint8_t *index = (uint8_t *)AscendC::GmAlloc(shape_index);
        uint8_t *bias = (uint8_t *)AscendC::GmAlloc(shape_bias);

        uint8_t *scale = (uint8_t *)AscendC::GmAlloc(shape_scale);
        uint8_t *pertokenScale = (uint8_t *)AscendC::GmAlloc(shape_pertoken);
        uint8_t *y = (uint8_t *)AscendC::GmAlloc(shape_y);
        uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspace_size);
        uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

        memset(x1, 1, shape_x1);
        memset(x2, 1, shape_x2);
        memset(index, 1, shape_index);
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

        AscendC::SetKernelMode(KernelMode::MIX_AIC_1_1);
        ICPU_SET_TILING_KEY(param.tilingKey);
        ICPU_RUN_KF(sparse4to2quant_matmul, std::min(MAX_BLOCK_DIM, param.blockDim), x1, x2, index, pertokenScale, scale, bias,
                    y, workspace, tiling);

        AscendC::GmFree(x1);
        AscendC::GmFree(x2);
        AscendC::GmFree(index);
        AscendC::GmFree(scale);
        AscendC::GmFree(bias);
        AscendC::GmFree(pertokenScale);
        AscendC::GmFree(y);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

#endif