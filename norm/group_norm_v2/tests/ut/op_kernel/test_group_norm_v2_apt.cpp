/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_group_norm_v2.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <sstream>
#include <string>
#include "gtest/gtest.h"
#include "group_norm_v2_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "register/op_def_registry.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void group_norm_v2(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling);

class group_norm_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "group_norm_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "group_norm_v2_test TearDown\n" << endl;
    }
};

static inline int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void ExcuteTestCase(
    const std::vector<int64_t>& xShape, const std::vector<int64_t>& wShape, const std::string& dtype, int64_t tilingKey,
    uint32_t blockNum, uint8_t* tiling)
{
    uint32_t typeSize = 4;
    if (dtype != "float") {
        typeSize = 2;
    }
    uint32_t realBlockNum = (blockNum + 1) / 2;
    size_t xFileSize = GetShapeSize(xShape) * typeSize;
    size_t weightFileSize = GetShapeSize(wShape) * typeSize;
    size_t meanFileSize = GetShapeSize(wShape) * 4;

    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc((xFileSize + 31) / 32 * 32);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc((weightFileSize + 31) / 32 * 32);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc((weightFileSize + 31) / 32 * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc((xFileSize + 31) / 32 * 32);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(group_norm_v2, realBlockNum, x, gamma, beta, y, mean, rstd, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(group_norm_v2_test, test_welford_perf_float32)
{
    std::vector<int64_t> xShape = {2, 4, 32, 32};
    std::vector<int64_t> wShape = {4};
    std::string dtype = "float";
    uint64_t tilingKey = 1100;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(GroupNormV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    GroupNormV2TilingData* tilingDatafromBin = reinterpret_cast<GroupNormV2TilingData*>(tiling);

    tilingDatafromBin->numGroups = 2;
    tilingDatafromBin->hwNum = 1024;
    tilingDatafromBin->elemNum = 2048;
    tilingDatafromBin->shapeC = 4;
    tilingDatafromBin->shapeD = 32;
    tilingDatafromBin->realCoreNum = 2;
    tilingDatafromBin->numPerCore = 1;
    tilingDatafromBin->numLastCore = 0;
    tilingDatafromBin->processSize = 2048;
    tilingDatafromBin->loopNum = 1;
    tilingDatafromBin->loopTail = 0;
    tilingDatafromBin->innerLoopNum = 1;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->tilingKey = 1100;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->parallelN = 1;
    tilingDatafromBin->ubSize = 61440;
    tilingDatafromBin->dichotomyAddPower = 11;
    tilingDatafromBin->dichotomyAddK = 1;
    tilingDatafromBin->dichotomyAddLastNum = 0;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(group_norm_v2_test, test_twopass_perf_float32)
{
    std::vector<int64_t> xShape = {2, 4, 8, 8};
    std::vector<int64_t> wShape = {4};
    std::string dtype = "float";
    uint64_t tilingKey = 1110;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(GroupNormV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    GroupNormV2TilingData* tilingDatafromBin = reinterpret_cast<GroupNormV2TilingData*>(tiling);

    tilingDatafromBin->numGroups = 2;
    tilingDatafromBin->hwNum = 64;
    tilingDatafromBin->elemNum = 128;
    tilingDatafromBin->shapeC = 4;
    tilingDatafromBin->shapeD = 8;
    tilingDatafromBin->realCoreNum = 2;
    tilingDatafromBin->numPerCore = 1;
    tilingDatafromBin->numLastCore = 0;
    tilingDatafromBin->processSize = 128;
    tilingDatafromBin->loopNum = 1;
    tilingDatafromBin->loopTail = 0;
    tilingDatafromBin->innerLoopNum = 1;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->tilingKey = 1110;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->parallelN = 1;
    tilingDatafromBin->ubSize = 61440;
    tilingDatafromBin->dichotomyAddPower = 7;
    tilingDatafromBin->dichotomyAddK = 1;
    tilingDatafromBin->dichotomyAddLastNum = 0;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}