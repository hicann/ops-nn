/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_layer_norm_apt.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <sstream>
#include <string>
#include "gtest/gtest.h"
#include "layer_norm_tiling_def.h"

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

extern "C" __global__ __aicore__ void layer_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR variance, GM_ADDR workspace,
    GM_ADDR tiling);

class layer_norm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "layer_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "layer_norm_test TearDown\n" << endl;
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

void ExcuteTestCase(
    const std::vector<int64_t>& xShape, const std::vector<int64_t>& gammaShape, const std::string& dtype,
    int64_t tilingKey, uint32_t blockNum, uint8_t* tiling)
{
    uint32_t typeSize = 4;
    uint32_t fp32TypeSize = 4;
    if (dtype != "float") {
        typeSize = 2;
    }
    uint32_t realBlockNum = (blockNum + 1) / 2;
    size_t xFileSize = GetShapeSize(xShape) * typeSize;
    size_t gammaFileSize = GetShapeSize(gammaShape) * typeSize;
    size_t meanFileSize = GetShapeSize(gammaShape) * fp32TypeSize;

    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc((xFileSize + 31) / 32 * 32);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc((gammaFileSize + 31) / 32 * 32);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc((gammaFileSize + 31) / 32 * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc((xFileSize + 31) / 32 * 32);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(layer_norm, realBlockNum, x, gamma, beta, y, mean, variance, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(layer_norm_test, test_two_pass_float32)
{
    std::vector<int64_t> xShape = {32, 64};
    std::vector<int64_t> gammaShape = {64};
    std::string dtype = "float";
    uint64_t tilingKey = 300000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(LayerNormV3TilingDataRegBaseTwoPass);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    LayerNormV3TilingDataRegBaseTwoPass* tilingDatafromBin =
        reinterpret_cast<LayerNormV3TilingDataRegBaseTwoPass*>(tiling);

    tilingDatafromBin->r = 64;
    tilingDatafromBin->rAlign = 64;
    tilingDatafromBin->a = 32;
    tilingDatafromBin->aFactor = 16;
    tilingDatafromBin->aBlockFactor = 16;
    tilingDatafromBin->blockNum = 2;
    tilingDatafromBin->binaryAddQuotient = 2;
    tilingDatafromBin->binaryAddK = 6;
    tilingDatafromBin->binaryAddLastNum = 0;
    tilingDatafromBin->powerOfTwoForR = 6;
    tilingDatafromBin->tmpBufferSize = 8192;
    tilingDatafromBin->nullptrGamma = 0;
    tilingDatafromBin->nullptrBeta = 0;
    tilingDatafromBin->epsilon = 1e-5;
    ExcuteTestCase(xShape, gammaShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(layer_norm_test, test_welford_float32)
{
    std::vector<int64_t> xShape = {32, 128};
    std::vector<int64_t> gammaShape = {128};
    std::string dtype = "float";
    uint64_t tilingKey = 400000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(LayerNormV3TilingDataWelford);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    LayerNormV3TilingDataWelford* tilingDatafromBin = reinterpret_cast<LayerNormV3TilingDataWelford*>(tiling);

    tilingDatafromBin->M = 32;
    tilingDatafromBin->N = 128;
    tilingDatafromBin->rAlign = 128;
    tilingDatafromBin->numBlocks = 2;
    tilingDatafromBin->mainBlockCount = 2;
    tilingDatafromBin->mainBlockFactor = 16;
    tilingDatafromBin->tailBlockFactor = 0;
    tilingDatafromBin->tileLength = 128;
    tilingDatafromBin->welfordTempSize = 8192;
    tilingDatafromBin->welfordUpdateTimes = 2;
    tilingDatafromBin->welfordUpdateTail = 0;
    tilingDatafromBin->nullptrGamma = 0;
    tilingDatafromBin->nullptrBeta = 0;
    tilingDatafromBin->apiTempBufferSize = 8192;
    tilingDatafromBin->epsilon = 1e-5;
    ExcuteTestCase(xShape, gammaShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(layer_norm_test, test_two_pass_perf_float32)
{
    std::vector<int64_t> xShape = {32, 64};
    std::vector<int64_t> gammaShape = {64};
    std::string dtype = "float";
    uint64_t tilingKey = 500000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(LayerNormV3TilingDataRegBaseTwoPassPerf);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    LayerNormV3TilingDataRegBaseTwoPassPerf* tilingDatafromBin =
        reinterpret_cast<LayerNormV3TilingDataRegBaseTwoPassPerf*>(tiling);

    tilingDatafromBin->a = 32;
    tilingDatafromBin->aBlockFactor = 16;
    tilingDatafromBin->aUbFactor = 64;
    tilingDatafromBin->aUbFactorAlignB32 = 64;
    tilingDatafromBin->r = 64;
    tilingDatafromBin->rAlign = 64;
    tilingDatafromBin->formerBlockUbLoops = 1;
    tilingDatafromBin->tailBlockUbLoops = 0;
    tilingDatafromBin->powerOfTwoForR = 6;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->nullptrGamma = 0;
    tilingDatafromBin->nullptrBeta = 0;
    ExcuteTestCase(xShape, gammaShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(layer_norm_test, test_no_reduce_float32)
{
    std::vector<int64_t> xShape = {32, 64};
    std::vector<int64_t> gammaShape = {64};
    std::string dtype = "float";
    uint64_t tilingKey = 600000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(LayerNormV3TilingDataRegBaseNoReduce);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    LayerNormV3TilingDataRegBaseNoReduce* tilingDatafromBin =
        reinterpret_cast<LayerNormV3TilingDataRegBaseNoReduce*>(tiling);

    tilingDatafromBin->a = 32;
    tilingDatafromBin->aBlockFactor = 16;
    tilingDatafromBin->aUbFactor = 64;
    tilingDatafromBin->aUbFactorAlignB32 = 64;
    tilingDatafromBin->formerBlockUbLoops = 1;
    tilingDatafromBin->tailBlockUbLoops = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->nullptrGamma = 0;
    tilingDatafromBin->nullptrBeta = 0;
    ExcuteTestCase(xShape, gammaShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(layer_norm_test, test_two_pass_half)
{
    std::vector<int64_t> xShape = {32, 64};
    std::vector<int64_t> gammaShape = {64};
    std::string dtype = "half";
    uint64_t tilingKey = 310000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(LayerNormV3TilingDataRegBaseTwoPass);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    LayerNormV3TilingDataRegBaseTwoPass* tilingDatafromBin =
        reinterpret_cast<LayerNormV3TilingDataRegBaseTwoPass*>(tiling);

    tilingDatafromBin->r = 64;
    tilingDatafromBin->rAlign = 64;
    tilingDatafromBin->a = 32;
    tilingDatafromBin->aFactor = 16;
    tilingDatafromBin->aBlockFactor = 16;
    tilingDatafromBin->blockNum = 2;
    tilingDatafromBin->binaryAddQuotient = 2;
    tilingDatafromBin->binaryAddK = 6;
    tilingDatafromBin->binaryAddLastNum = 0;
    tilingDatafromBin->powerOfTwoForR = 6;
    tilingDatafromBin->tmpBufferSize = 8192;
    tilingDatafromBin->nullptrGamma = 0;
    tilingDatafromBin->nullptrBeta = 0;
    tilingDatafromBin->epsilon = 1e-5;
    ExcuteTestCase(xShape, gammaShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}