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
 * \file test_lp_norm_v2_apt.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <sstream>
#include <string>
#include "gtest/gtest.h"
#include "lp_norm_v2_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "register/op_def_registry.h"
#include "string.h"
#include <iostream>
#include <string>
#include "lp_norm_v2_apt.cpp"
#endif

#include <cstdint>

using namespace std;
using namespace LpNormV2;

class lp_norm_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "lp_norm_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "lp_norm_v2_test TearDown\n" << endl;
    }
};

std::string Shape2Str(const std::vector<int64_t>& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i != shape.size() - 1) {
            oss << ",";
        }
    }
    oss << "]";
    return oss.str();
}

static inline int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

TEST_F(lp_norm_v2_test, test_p2_float32)
{
    std::vector<int64_t> xShape = {2, 3, 4};
    std::vector<int64_t> yShape = {2, 1, 4};
    std::string dtype = "float";
    uint32_t blockNum = 1;
    size_t tilingSize = sizeof(optiling::LpNormV2TilingData);

    uint32_t typeSize = 4;
    size_t xFileSize = GetShapeSize(xShape) * typeSize;
    size_t yFileSize = GetShapeSize(yShape) * typeSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc((xFileSize + 31) / 32 * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc((yFileSize + 31) / 32 * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    optiling::LpNormV2TilingData* tilingData = reinterpret_cast<optiling::LpNormV2TilingData*>(tiling);

    memset(tilingData, 0, tilingSize);
    tilingData->reduceTiling.factorACntPerCore = 1;
    tilingData->reduceTiling.factorATotalCnt = 1;
    tilingData->reduceTiling.ubFactorA = 2;
    tilingData->reduceTiling.factorRCntPerCore = 1;
    tilingData->reduceTiling.factorRTotalCnt = 1;
    tilingData->reduceTiling.ubFactorR = 1;
    tilingData->reduceTiling.groupR = 1;
    tilingData->reduceTiling.outSize = 8;
    tilingData->reduceTiling.basicBlock = 47616;
    tilingData->reduceTiling.resultBlock = 15616;
    tilingData->reduceTiling.coreNum = 1;
    tilingData->reduceTiling.useNddma = 1;
    tilingData->reduceTiling.meanVar = 0.333333;
    tilingData->reduceTiling.shape[0] = 2;
    tilingData->reduceTiling.shape[1] = 3;
    tilingData->reduceTiling.shape[2] = 4;
    tilingData->reduceTiling.stride[0] = 12;
    tilingData->reduceTiling.stride[1] = 4;
    tilingData->reduceTiling.stride[2] = 1;
    tilingData->reduceTiling.dstStride[0] = 4;
    tilingData->reduceTiling.dstStride[1] = 4;
    tilingData->reduceTiling.dstStride[2] = 1;
    tilingData->reduceTiling.sliceNum[0] = 1;
    tilingData->reduceTiling.sliceNum[1] = 1;
    tilingData->reduceTiling.sliceNum[2] = 1;
    tilingData->reduceTiling.sliceShape[0] = 2;
    tilingData->reduceTiling.sliceShape[1] = 1;
    tilingData->reduceTiling.sliceShape[2] = 4;
    tilingData->reduceTiling.sliceStride[0] = 12;
    tilingData->reduceTiling.sliceStride[1] = 4;
    tilingData->reduceTiling.sliceStride[2] = 1;
    tilingData->epsilon = 1e-12;
    tilingData->p = 2.0;
    tilingData->recp = 0.0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(67114025);
    ICPU_RUN_KF((::lp_norm_v2<true, 20, 10, 0, TEMPLATE_P2>), blockNum, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(lp_norm_v2_test, test_p1_float32)
{
    std::vector<int64_t> xShape = {2, 4, 5};
    std::vector<int64_t> yShape = {2, 1, 5};
    std::string dtype = "float";
    uint32_t blockNum = 1;
    size_t tilingSize = sizeof(optiling::LpNormV2TilingData);

    uint32_t typeSize = 4;
    size_t xFileSize = GetShapeSize(xShape) * typeSize;
    size_t yFileSize = GetShapeSize(yShape) * typeSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc((xFileSize + 31) / 32 * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc((yFileSize + 31) / 32 * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    optiling::LpNormV2TilingData* tilingData = reinterpret_cast<optiling::LpNormV2TilingData*>(tiling);

    memset(tilingData, 0, tilingSize);
    tilingData->reduceTiling.factorACntPerCore = 1;
    tilingData->reduceTiling.factorATotalCnt = 1;
    tilingData->reduceTiling.ubFactorA = 2;
    tilingData->reduceTiling.factorRCntPerCore = 1;
    tilingData->reduceTiling.factorRTotalCnt = 1;
    tilingData->reduceTiling.ubFactorR = 1;
    tilingData->reduceTiling.groupR = 1;
    tilingData->reduceTiling.outSize = 10;
    tilingData->reduceTiling.basicBlock = 52736;
    tilingData->reduceTiling.resultBlock = 13056;
    tilingData->reduceTiling.coreNum = 1;
    tilingData->reduceTiling.useNddma = 1;
    tilingData->reduceTiling.meanVar = 0.25;
    tilingData->reduceTiling.shape[0] = 2;
    tilingData->reduceTiling.shape[1] = 4;
    tilingData->reduceTiling.shape[2] = 5;
    tilingData->reduceTiling.stride[0] = 20;
    tilingData->reduceTiling.stride[1] = 5;
    tilingData->reduceTiling.stride[2] = 1;
    tilingData->reduceTiling.dstStride[0] = 5;
    tilingData->reduceTiling.dstStride[1] = 5;
    tilingData->reduceTiling.dstStride[2] = 1;
    tilingData->reduceTiling.sliceNum[0] = 1;
    tilingData->reduceTiling.sliceNum[1] = 1;
    tilingData->reduceTiling.sliceNum[2] = 1;
    tilingData->reduceTiling.sliceShape[0] = 2;
    tilingData->reduceTiling.sliceShape[1] = 1;
    tilingData->reduceTiling.sliceShape[2] = 5;
    tilingData->reduceTiling.sliceStride[0] = 20;
    tilingData->reduceTiling.sliceStride[1] = 5;
    tilingData->reduceTiling.sliceStride[2] = 1;
    tilingData->epsilon = 1e-12;
    tilingData->p = 1.0;
    tilingData->recp = 0.0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(33559593);
    ICPU_RUN_KF((::lp_norm_v2<true, 20, 10, 0, TEMPLATE_P1>), blockNum, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(lp_norm_v2_test, test_p_inf_float32)
{
    std::vector<int64_t> xShape = {20, 30, 40};
    std::vector<int64_t> yShape = {20, 1, 40};
    std::string dtype = "float";
    uint32_t blockNum = 1;
    size_t tilingSize = sizeof(optiling::LpNormV2TilingData);

    uint32_t typeSize = 4;
    size_t xFileSize = GetShapeSize(xShape) * typeSize;
    size_t yFileSize = GetShapeSize(yShape) * typeSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc((xFileSize + 31) / 32 * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc((yFileSize + 31) / 32 * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    optiling::LpNormV2TilingData* tilingData = reinterpret_cast<optiling::LpNormV2TilingData*>(tiling);

    memset(tilingData, 0, tilingSize);
    tilingData->reduceTiling.factorACntPerCore = 1;
    tilingData->reduceTiling.factorATotalCnt = 20;
    tilingData->reduceTiling.ubFactorA = 1;
    tilingData->reduceTiling.factorRCntPerCore = 1;
    tilingData->reduceTiling.factorRTotalCnt = 1;
    tilingData->reduceTiling.ubFactorR = 1;
    tilingData->reduceTiling.groupR = 1;
    tilingData->reduceTiling.outSize = 800;
    tilingData->reduceTiling.basicBlock = 58368;
    tilingData->reduceTiling.resultBlock = 1792;
    tilingData->reduceTiling.coreNum = 20;
    tilingData->reduceTiling.useNddma = 0;
    tilingData->reduceTiling.meanVar = 0.033333;
    tilingData->reduceTiling.shape[0] = 20;
    tilingData->reduceTiling.shape[1] = 30;
    tilingData->reduceTiling.shape[2] = 40;
    tilingData->reduceTiling.stride[0] = 1200;
    tilingData->reduceTiling.stride[1] = 40;
    tilingData->reduceTiling.stride[2] = 1;
    tilingData->reduceTiling.dstStride[0] = 40;
    tilingData->reduceTiling.dstStride[1] = 40;
    tilingData->reduceTiling.dstStride[2] = 1;
    tilingData->reduceTiling.sliceNum[0] = 1;
    tilingData->reduceTiling.sliceNum[1] = 1;
    tilingData->reduceTiling.sliceNum[2] = 1;
    tilingData->reduceTiling.sliceShape[0] = 20;
    tilingData->reduceTiling.sliceShape[1] = 1;
    tilingData->reduceTiling.sliceShape[2] = 40;
    tilingData->reduceTiling.sliceStride[0] = 1200;
    tilingData->reduceTiling.sliceStride[1] = 40;
    tilingData->reduceTiling.sliceStride[2] = 1;
    tilingData->epsilon = 1e-12;
    tilingData->p = INFINITY;
    tilingData->recp = 0.0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(167777321);
    ICPU_RUN_KF((::lp_norm_v2<true, 20, 10, 0, TEMPLATE_P_INF>), blockNum, x, y, workspace, (uint8_t*)tilingData);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

