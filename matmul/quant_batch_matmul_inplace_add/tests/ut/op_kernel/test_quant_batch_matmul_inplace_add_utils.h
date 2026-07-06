/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_QUANT_BATCH_MATMUL_INPLACE_ADD_UTILS_H
#define TEST_QUANT_BATCH_MATMUL_INPLACE_ADD_UTILS_H

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef __CCE_KT_TEST__
#include "quant_batch_matmul_inplace_add_tiling_def.h"
#include "tikicpulib.h"
#include "gtest/gtest.h"
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/quant_batch_matmul_inplace_add_tiling_key.h"
#include "arch35/quant_batch_matmul_inplace_add.cpp"

using namespace QuantBatchMatmulInplaceAddArch35TilingKey;

#if SUPPORT_MX_WITHOUT_BATCH_TILING_KEY
#define QBMIIA_UT_SUPPORT_MX_WITHOUT_BATCH 1
#else
#define QBMIIA_UT_SUPPORT_MX_WITHOUT_BATCH 0
#endif

#define QBMIIA_PARAM_LIST_DEF \
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x2Scale, GM_ADDR yIn, GM_ADDR x1Scale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling

#define QBMIIA_PARAM_LIST x1, x2, x2Scale, yIn, x1Scale, y, workspace, tiling

using QuantBatchMatmulInplaceAddAptFunc = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                                                   GM_ADDR);

static std::unordered_map<uint64_t, QuantBatchMatmulInplaceAddAptFunc> s_funcMapApt = {
    {1UL, quant_batch_matmul_inplace_add<1, 0, TPL_NO_VEC_EPILOGUE_WITH_MMAPI>},
    {17UL, quant_batch_matmul_inplace_add<1, 0, TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI>},
#if QBMIIA_UT_SUPPORT_MX_WITHOUT_BATCH
    {33UL, quant_batch_matmul_inplace_add<1, 0, TPL_NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH>},
    {49UL, quant_batch_matmul_inplace_add<1, 0, TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH>},
#endif
};
#endif

using namespace std;

struct QuantBatchMatmulInplaceAddTestParam {
    string socVersion;
    string caseName;
    string kernelUtTarget;
    string prefix;
    int64_t batchA;
    int64_t batchB;
    int64_t batchC;
    int64_t m;
    int64_t k;
    int64_t n;
    bool transA;
    bool transB;
    int64_t groupSize;
    bool hasX1Scale;
    bool hasX2Scale;
    bool result;
    uint32_t numBlocks;
    uint64_t tilingKey;
    string tilingData;
};

class QuantBatchMatmulInplaceAddTestUtils {
public:
    static constexpr uint32_t MAX_NUM_BLOCKS = 4;

    static void SplitStr2Vec(const string& input, const string& delimiter, vector<string>& output)
    {
        auto delimiterLen = delimiter.size();
        string::size_type currPos = 0;
        string::size_type nextPos = input.find(delimiter, currPos);
        while (nextPos != string::npos) {
            output.emplace_back(input.substr(currPos, nextPos - currPos));
            currPos = nextPos + delimiterLen;
            nextPos = input.find(delimiter, currPos);
        }
        if (currPos < input.size()) {
            output.emplace_back(input.substr(currPos));
        }
    }

    static string GetExeDirPath()
    {
        string exePath("./");
        char path[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", path, sizeof(path));
        if (n > 0) {
            path[n] = '\0';
            exePath.assign(path);
            auto pos = exePath.find_last_of('/');
            if (pos != string::npos) {
                exePath.erase(pos + 1);
            } else {
                exePath.assign("./");
            }
        }
        return exePath;
    }

    static vector<QuantBatchMatmulInplaceAddTestParam> GetParams(const string& socVersion, const string& testSuite)
    {
        vector<QuantBatchMatmulInplaceAddTestParam> params;
        string rootPath(GetExeDirPath() + "../../../../");
        string casePath(
            rootPath +
            "matmul/quant_batch_matmul_inplace_add/tests/ut/op_kernel/test_quant_batch_matmul_inplace_add.csv");
        ifstream csvData(casePath, ios::in);
        if (!csvData.is_open()) {
            cout << "cannot open case file " << casePath << ", maybe not exist" << endl;
            return params;
        }

        string line;
        while (getline(csvData, line)) {
            vector<string> testParam;
            SplitStr2Vec(line, ",", testParam);
            if (testParam.empty() || testParam[0] == "socVersion") {
                continue;
            }

            QuantBatchMatmulInplaceAddTestParam param;
            size_t idx = 0UL;
            param.socVersion = testParam[idx++];
            param.caseName = testParam[idx++];
            param.kernelUtTarget = testParam[idx++];
            if (param.socVersion != socVersion || param.kernelUtTarget != testSuite) {
                continue;
            }

            param.prefix = testParam[idx++];
            idx++; // skip coreNum
            param.batchA = stol(testParam[idx++]);
            param.batchB = stol(testParam[idx++]);
            param.batchC = stol(testParam[idx++]);
            param.m = stol(testParam[idx++]);
            param.k = stol(testParam[idx++]);
            param.n = stol(testParam[idx++]);
            param.transA = stol(testParam[idx++]) != 0;
            param.transB = stol(testParam[idx++]) != 0;
            param.groupSize = stol(testParam[idx++]);
            param.hasX1Scale = stol(testParam[idx++]) != 0;
            param.hasX2Scale = stol(testParam[idx++]) != 0;
            param.result = (strcasecmp(testParam[idx++].c_str(), "true") == 0);
            param.numBlocks = static_cast<uint32_t>(stol(testParam[idx++]));
            param.tilingKey = stoull(testParam[idx++]);
            param.tilingData = testParam[idx++];
            params.push_back(param);
        }
        return params;
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    static constexpr bool IsMxWithoutBatchUtSupported() { return QBMIIA_UT_SUPPORT_MX_WITHOUT_BATCH != 0; }

    static bool IsMxWithoutBatchTilingData(const QuantBatchMatmulInplaceAddTestParam& param)
    {
        constexpr uint64_t kernelTypeShift = 4UL;
        constexpr uint64_t kernelTypeMask = 0xFUL;
        uint64_t kernelType = (param.tilingKey >> kernelTypeShift) & kernelTypeMask;
        return kernelType == TPL_NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH ||
               kernelType == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH;
    }

    static bool ShouldSkipMxWithoutBatchUt(const QuantBatchMatmulInplaceAddTestParam& param)
    {
        return IsMxWithoutBatchTilingData(param) && !IsMxWithoutBatchUtSupported();
    }

    static size_t GetTilingDataSize(const QuantBatchMatmulInplaceAddTestParam& param)
    {
        if (IsMxWithoutBatchTilingData(param)) {
            return sizeof(QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData);
        }
        return sizeof(QMMIA::QuantBatchMatmulInplaceAddTilingData);
    }

    static void InitDefaultTilingData(const QuantBatchMatmulInplaceAddTestParam& param,
                                      QMMIA::QuantBatchMatmulInplaceAddTilingData& tilingData)
    {
        memset(&tilingData, 0, sizeof(tilingData));
        tilingData.params.batchA = static_cast<uint32_t>(param.batchA > 0 ? param.batchA : 1);
        tilingData.params.batchB = static_cast<uint32_t>(param.batchB > 0 ? param.batchB : 1);
        tilingData.params.batchC = static_cast<uint32_t>(param.batchC > 0 ? param.batchC : 1);
        tilingData.matmulTiling.m = static_cast<uint32_t>(param.m);
        tilingData.matmulTiling.k = static_cast<uint32_t>(param.k);
        tilingData.matmulTiling.n = static_cast<uint32_t>(param.n);
        tilingData.matmulTiling.baseM = static_cast<uint32_t>(std::min<int64_t>(param.m, 128));
        tilingData.matmulTiling.baseN = static_cast<uint32_t>(std::min<int64_t>(param.n, 128));
        tilingData.matmulTiling.baseK = static_cast<uint32_t>(std::min<int64_t>(param.k, 64));
        tilingData.matmulTiling.stepKa = 1;
        tilingData.matmulTiling.stepKb = 1;
        tilingData.matmulTiling.scaleFactorA = 1;
        tilingData.matmulTiling.scaleFactorB = 1;
        tilingData.matmulTiling.nBufferNum = 2;
        tilingData.matmulTiling.dbL0C = 2;
        tilingData.adaptiveSlidingWin.mBaseTailSplitCnt = 1;
        tilingData.adaptiveSlidingWin.nBaseTailSplitCnt = 1;
    }

    static void InitDefaultWithoutBatchTilingData(
        const QuantBatchMatmulInplaceAddTestParam& param,
        QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData& tilingData)
    {
        constexpr uint8_t MX_PERGROUP_MODE = 0x1U << 3;
        memset(&tilingData, 0, sizeof(tilingData));
        tilingData.m = static_cast<uint32_t>(param.m);
        tilingData.k = static_cast<uint32_t>(param.k);
        tilingData.n = static_cast<uint32_t>(param.n);
        tilingData.baseM = static_cast<uint16_t>(std::min<int64_t>(param.m, 128));
        tilingData.baseN = static_cast<uint16_t>(std::min<int64_t>(param.n, 128));
        tilingData.baseK = static_cast<uint16_t>(std::min<int64_t>(param.k, 64));
        tilingData.scaleKL1 = static_cast<uint32_t>(tilingData.baseK);
        tilingData.groupSizeK = static_cast<uint16_t>(param.groupSize);
        tilingData.mBaseTailSplitCnt = 1;
        tilingData.nBaseTailSplitCnt = 1;
        tilingData.stepKa = 1;
        tilingData.stepKb = 1;
        tilingData.x1QuantMode = MX_PERGROUP_MODE;
        tilingData.x2QuantMode = MX_PERGROUP_MODE;
        tilingData.nBufferNum = 2;
        tilingData.dbL0C = 2;
    }

    static void FillTilingBuffer(const QuantBatchMatmulInplaceAddTestParam& param, uint8_t* tiling)
    {
        size_t tilingDataSize = GetTilingDataSize(param);
        if (param.tilingData.empty() || param.tilingData == "AUTO") {
            if (IsMxWithoutBatchTilingData(param)) {
                QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData tilingStruct;
                InitDefaultWithoutBatchTilingData(param, tilingStruct);
                memcpy(tiling, &tilingStruct, tilingDataSize);
            } else {
                QMMIA::QuantBatchMatmulInplaceAddTilingData tilingStruct;
                InitDefaultTilingData(param, tilingStruct);
                memcpy(tiling, &tilingStruct, tilingDataSize);
            }
            return;
        }

        vector<string> tilingDataStr;
        SplitStr2Vec(param.tilingData, " ", tilingDataStr);
        vector<int32_t> tilingDataInt;
        tilingDataInt.reserve(tilingDataStr.size());
        for (auto& tilingValue : tilingDataStr) {
            tilingDataInt.push_back(atoi(tilingValue.c_str()));
        }
        ASSERT_EQ(tilingDataInt.size() * sizeof(int32_t), tilingDataSize);
        memcpy(tiling, tilingDataInt.data(), tilingDataSize);
    }

    static size_t GetX1ScaleBytes(const QuantBatchMatmulInplaceAddTestParam& param)
    {
        if (!param.hasX1Scale) {
            return 0;
        }
        if (param.groupSize > 0) {
            int64_t groupNum = (param.k + param.groupSize - 1) / param.groupSize;
            return static_cast<size_t>(param.batchA) * param.m * groupNum * 2 * sizeof(uint8_t);
        }
        return sizeof(float);
    }

    static size_t GetX2ScaleBytes(const QuantBatchMatmulInplaceAddTestParam& param)
    {
        if (!param.hasX2Scale) {
            return 0;
        }
        if (param.groupSize > 0) {
            int64_t groupNum = (param.k + param.groupSize - 1) / param.groupSize;
            if (param.transB) {
                return static_cast<size_t>(param.batchB) * param.n * groupNum * 2 * sizeof(uint8_t);
            }
            return static_cast<size_t>(groupNum) * param.n * 2 * sizeof(uint8_t);
        }
        return sizeof(float);
    }

    static void TestOneParamCase950(const QuantBatchMatmulInplaceAddTestParam& param,
                                    decltype(s_funcMapApt)& funcMapApt)
    {
        if (ShouldSkipMxWithoutBatchUt(param)) {
#ifdef __CCE_KT_TEST__
            GTEST_SKIP() << "Skip MX without-batch TensorAPI UT before asc-devkit 9.1.0: " << param.caseName;
#else
            cout << "Skip MX without-batch TensorAPI UT before asc-devkit 9.1.0: " << param.caseName << endl;
            return;
#endif
        }

        int64_t batchA = param.batchA > 0 ? param.batchA : 1L;
        int64_t batchB = param.batchB > 0 ? param.batchB : 1L;
        int64_t batchC = param.batchC > 0 ? param.batchC : 1L;
        size_t shapeX1 = static_cast<size_t>(batchA) * param.m * param.k * sizeof(DTYPE_X1);
        size_t shapeX2 = static_cast<size_t>(batchB) * param.k * param.n * sizeof(DTYPE_X2);
        size_t shapeY = static_cast<size_t>(batchC) * param.m * param.n * sizeof(DTYPE_Y);
        size_t shapeX1Scale = GetX1ScaleBytes(param);
        size_t shapeX2Scale = GetX2ScaleBytes(param);
        size_t tilingDataSize = GetTilingDataSize(param);
        size_t workspaceSize = 16 * 1024 * 1024 + 50 * 1024 * 1024;

        uint8_t* x1 = static_cast<uint8_t*>(AscendC::GmAlloc(shapeX1));
        uint8_t* x2 = static_cast<uint8_t*>(AscendC::GmAlloc(shapeX2));
        uint8_t* yIn = static_cast<uint8_t*>(AscendC::GmAlloc(shapeY));
        uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(shapeY));
        uint8_t* x1Scale = shapeX1Scale > 0 ? static_cast<uint8_t*>(AscendC::GmAlloc(shapeX1Scale)) : nullptr;
        uint8_t* x2Scale = shapeX2Scale > 0 ? static_cast<uint8_t*>(AscendC::GmAlloc(shapeX2Scale)) : nullptr;
        uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(workspaceSize));
        uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

        memset(x1, 1, shapeX1);
        memset(x2, 1, shapeX2);
        memset(yIn, 1, shapeY);
        memset(y, 0, shapeY);
        if (x1Scale != nullptr) {
            memset(x1Scale, 1, shapeX1Scale);
        }
        if (x2Scale != nullptr) {
            memset(x2Scale, 1, shapeX2Scale);
        }
        memset(workspace, 0, workspaceSize);
        FillTilingBuffer(param, tiling);

        auto func = [&param, &funcMapApt](QBMIIA_PARAM_LIST_DEF) {
            auto it = funcMapApt.find(param.tilingKey);
            if (it != funcMapApt.end()) {
                it->second(QBMIIA_PARAM_LIST);
            } else {
                throw runtime_error("Unsupported tilingKey.");
            }
        };

        ICPU_RUN_KF(func, std::min(MAX_NUM_BLOCKS, param.numBlocks), x1, x2, x2Scale, yIn, x1Scale, y, workspace,
                    tiling);

        AscendC::GmFree(x1);
        AscendC::GmFree(x2);
        AscendC::GmFree(yIn);
        AscendC::GmFree(y);
        if (x1Scale != nullptr) {
            AscendC::GmFree(x1Scale);
        }
        if (x2Scale != nullptr) {
            AscendC::GmFree(x2Scale);
        }
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
#endif
};

#endif
