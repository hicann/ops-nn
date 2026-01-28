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
 * \file test_batch_norm_grad.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <sstream>
#include <string>
#include "gtest/gtest.h"
#include "batch_norm_grad_tiling_def.h"

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

extern "C" __global__ __aicore__ void batch_norm_grad(GM_ADDR y_backprop, GM_ADDR x, GM_ADDR scale, GM_ADDR reserve_space1,
    GM_ADDR reserve_space2, GM_ADDR reserve_space3, GM_ADDR x_backprop, GM_ADDR scale_backprop, GM_ADDR offset_backprop, 
    GM_ADDR reserve_space4, GM_ADDR reserve_space5, GM_ADDR workspace, GM_ADDR tiling);

class batch_norm_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "batch_norm_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "batch_norm_grad_test TearDown\n" << endl;
    }
};

std::string Shape2Str(const std::vector<int64_t>& shape) {
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

static inline int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for(auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void ExcuteTestCase(const std::vector<int64_t> &xShape, const std::vector<int64_t> &wShape, const std::string &dtype,
                    int64_t tilingKey, uint32_t blockNum, uint8_t *tiling, bool is_training = true)
{
    uint32_t typeSize = 4;
    uint32_t fp32TypeSize = 4;
    if (dtype != "float") {
        typeSize = 2;
    }
    // 每个block对应 1 aic + 2 aiv
    uint32_t realBlockNum = (blockNum + 1) / 2;
    size_t xFileSize = GetShapeSize(xShape) * typeSize;
    size_t weightFileSize = GetShapeSize(wShape) * typeSize;
    size_t meanFileSize = GetShapeSize(wShape) * fp32TypeSize;

    size_t workspaceFileSize = 16*1024*1024;

    uint8_t *dy = (uint8_t *)AscendC::GmAlloc((xFileSize + 31)/32*32);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc((xFileSize + 31)/32*32);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc((weightFileSize + 31)/32*32);
    uint8_t *mean = (uint8_t *)AscendC::GmAlloc((meanFileSize + 31)/32*32);
    uint8_t *var = (uint8_t *)AscendC::GmAlloc((meanFileSize + 31)/32*32);
    uint8_t *reserve_space3 = (uint8_t *)AscendC::GmAlloc((meanFileSize + 31)/32*32);
    uint8_t *dx = (uint8_t *)AscendC::GmAlloc((xFileSize + 31)/32*32);
    uint8_t *dscale = (uint8_t *)AscendC::GmAlloc((weightFileSize + 31)/32*32);
    uint8_t *dbias = (uint8_t *)AscendC::GmAlloc((weightFileSize + 31)/32*32);
    uint8_t *reserver_space_4 = (uint8_t *)AscendC::GmAlloc((weightFileSize + 31)/32*32);
    uint8_t *reserver_space_5 = (uint8_t *)AscendC::GmAlloc((weightFileSize + 31)/32*32);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(batch_norm_grad, realBlockNum, dy, x, scale, mean, var, reserve_space3, dx,
                dscale, dbias, reserver_space_4, reserver_space_5, workspace, tiling);

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(mean);
    AscendC::GmFree(var);
    AscendC::GmFree(reserve_space3);
    AscendC::GmFree(dx);
    AscendC::GmFree(dscale);
    AscendC::GmFree(dbias);
    AscendC::GmFree(reserver_space_4);
    AscendC::GmFree(reserver_space_5);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(batch_norm_grad_test, test_split_r1_float32)
{
    std::vector<int64_t> xShape = {5, 2, 32, 128};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 31000000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradRARRecomputeTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradRARRecomputeTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradRARRecomputeTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 5;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 4096;
    tilingDatafromBin->baseTilingData.rAlign = 8192;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 0;
    tilingDatafromBin->baseTilingData.formerBlockDim = 1;
    tilingDatafromBin->baseTilingData.tailBlockDim = 2;
    tilingDatafromBin->generalBinAddTilingData.binaryAddQuotient = 8192;
    tilingDatafromBin->generalBinAddTilingData.binaryAddk = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->tailBinAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->tailBinAddTilingData.binaryAddk = 0;
    tilingDatafromBin->tailBinAddTilingData.binaryAddLastNum = 32;
    tilingDatafromBin->ubRDimFactor = 8192;
    tilingDatafromBin->ubRDimFactorAlign = 8192;
    tilingDatafromBin->ubRDimLoopNum = 2;
    tilingDatafromBin->ubRDimTail = 4096;
    tilingDatafromBin->ubRDimTailFactor = 4096;
    tilingDatafromBin->ubRDimTailFactorAlign = 4096;
    tilingDatafromBin->ubRDimTailLoopNum = 1;
    tilingDatafromBin->ubRDimTailTail = 0;
    tilingDatafromBin->ubRDimTailTailFactor = 0;
    tilingDatafromBin->ubRDimTailTailFactorAlign = 0;
    tilingDatafromBin->ubRDimTailTailLoopNum = 0;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_test, test_full_load_float32)
{
    std::vector<int64_t> xShape = {32, 2, 13 ,16};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 10000000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradRARFullLoadTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradRARFullLoadTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradRARFullLoadTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 32;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 208;
    tilingDatafromBin->baseTilingData.rAlign = 6656;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 0;
    tilingDatafromBin->baseTilingData.formerBlockDim = 1;
    tilingDatafromBin->baseTilingData.tailBlockDim = 2;
    tilingDatafromBin->binaryAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->binaryAddTilingData.binaryAddk = 0;
    tilingDatafromBin->binaryAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->formerUbDim = 2;
    tilingDatafromBin->ubLoopOfFormerBlock = 1;
    tilingDatafromBin->ubTailOfFormerBlock = 1;
    tilingDatafromBin->ubLoopOfTailBlock = 1;
    tilingDatafromBin->ubTailOfTailBlock = 2;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_test, test_split_r0_float32)
{
    std::vector<int64_t> xShape = {1, 2, 448, 448};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 32000000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormGradRARRecomputeTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradRARRecomputeTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradRARRecomputeTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.r1Dim = 1;
    tilingDatafromBin->baseTilingData.aDim = 2;
    tilingDatafromBin->baseTilingData.r0Dim = 200704;
    tilingDatafromBin->baseTilingData.rAlign = 0;
    tilingDatafromBin->baseTilingData.blockNum = 2;
    tilingDatafromBin->baseTilingData.tailBlockNum = 2;
    tilingDatafromBin->baseTilingData.formerBlockDim = 0;
    tilingDatafromBin->baseTilingData.tailBlockDim = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddQuotient = 8192;
    tilingDatafromBin->generalBinAddTilingData.binaryAddk = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->tailBinAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->tailBinAddTilingData.binaryAddk = 0;
    tilingDatafromBin->tailBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->ubRDimFactor = 14880;
    tilingDatafromBin->ubRDimFactorAlign = 14880;
    tilingDatafromBin->ubRDimLoopNum = 8;
    tilingDatafromBin->ubRDimTail = 81664;
    tilingDatafromBin->ubRDimTailFactor = 7440;
    tilingDatafromBin->ubRDimTailFactorAlign = 7440;
    tilingDatafromBin->ubRDimTailLoopNum = 10;
    tilingDatafromBin->ubRDimTailTail = 7264;
    tilingDatafromBin->ubRDimTailTailFactor = 7440;
    tilingDatafromBin->ubRDimTailTailFactorAlign = 7440;
    tilingDatafromBin->ubRDimTailTailLoopNum = 2;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_test, test_infer_channel_last_float32)
{
    std::vector<int64_t> xShape = {10, 2, 3, 200};
    std::vector<int64_t> wShape = {200};
    std::string dtype = "float";
    uint64_t tilingKey = 900000;
    uint32_t blockNum = 25;
    size_t tilingSize = sizeof(BatchNormGradInferChannelLastTilingData) + sizeof(float);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradInferChannelLastTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradInferChannelLastTilingData*>(tiling);
    
    tilingDatafromBin->dxTilingData.totalTiles = 2;
    tilingDatafromBin->dxTilingData.tilesPerCore = 1;
    tilingDatafromBin->dxTilingData.usedCoreNums = 2;
    tilingDatafromBin->dxTilingData.aDim = 200;
    tilingDatafromBin->dxTilingData.aOuter = 2;
    tilingDatafromBin->dxTilingData.bOuter = 1;
    tilingDatafromBin->dxTilingData.tileBlockALen = 192;
    tilingDatafromBin->dxTilingData.tileBlockATail = 8;
    tilingDatafromBin->dxTilingData.tileBlockAPaddingNum = 184;
    tilingDatafromBin->dxTilingData.tileBlockBLen = 60;
    tilingDatafromBin->dxTilingData.tileBlockBTail = 60;
    tilingDatafromBin->dxTilingData.epsilon = 0;
    tilingDatafromBin->binAddRFactorStg1 = 128;
    tilingDatafromBin->binAddRLoopStg1 = 0;
    tilingDatafromBin->binAddRTotalLoopStg1 = 1;
    tilingDatafromBin->binAddRTailStg1 = 60;
    tilingDatafromBin->binAddBasicBlockLoopStg1 = 0;
    tilingDatafromBin->binAddMainFoldCountStg1 = 0;
    tilingDatafromBin->binAddCacheBufferCountStg1 = 1;
    tilingDatafromBin->binAddResultCacheIDStg1 = 0;
    tilingDatafromBin->aDimStg1 = 200;
    tilingDatafromBin->aOuterStg1 = 25;
    tilingDatafromBin->aInnerStg1 = 8;
    tilingDatafromBin->aTailStg1 = 8;
    tilingDatafromBin->aOuterPerCoreStg1 = 1;
    tilingDatafromBin->usedCoreNumsStg1 = 25;
    tilingDatafromBin->enableDx = 1;
    tilingDatafromBin->enableDgamma = 1;
    tilingDatafromBin->enableDbeta = 1;

    ExcuteTestCase(xShape, wShape, dtype, tilingKey, blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_test, test_infer_splitR1_float32)
{
    std::vector<int64_t> xShape = {1, 5, 1, 1};
    std::vector<int64_t> wShape = {1};
    std::string dtype = "float";
    uint64_t tilingKey = 910001;
    uint32_t blockNum = 1;
    size_t tilingSize = sizeof(BatchNormGradInferTilingData) + sizeof(float);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradInferTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradInferTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.totalTiles = 1;
    tilingDatafromBin->baseTilingData.tilesPerCore = 1;
    tilingDatafromBin->baseTilingData.usedCoreNums = 1;
    tilingDatafromBin->baseTilingData.r1Dim = 1;
    tilingDatafromBin->baseTilingData.aDim = 1;
    tilingDatafromBin->baseTilingData.r0Dim = 5;
    tilingDatafromBin->baseTilingData.r1Outer = 1;
    tilingDatafromBin->baseTilingData.aOuter = 1;
    tilingDatafromBin->baseTilingData.r0Outer = 1;
    tilingDatafromBin->baseTilingData.tileBlockR1Len = 1;
    tilingDatafromBin->baseTilingData.tileBlockR1Tail = 1;
    tilingDatafromBin->baseTilingData.tileBlockALen = 1;
    tilingDatafromBin->baseTilingData.tileBlockATail = 1;
    tilingDatafromBin->baseTilingData.tileBlockR0Len = 64;
    tilingDatafromBin->baseTilingData.tileBlockR0Tail = 5;
    tilingDatafromBin->baseTilingData.tileBlockAPaddingNum = 0;
    tilingDatafromBin->baseTilingData.epsilon = 0;
    tilingDatafromBin->generalBinAddTilingData.binaryAddQuotient = 4;
    tilingDatafromBin->generalBinAddTilingData.binaryAddk = 0;
    tilingDatafromBin->generalBinAddTilingData.binaryAddLastNum = 0;
    tilingDatafromBin->tailBinAddTilingData.binaryAddQuotient = 1;
    tilingDatafromBin->tailBinAddTilingData.binaryAddk = 0;
    tilingDatafromBin->tailBinAddTilingData.binaryAddLastNum = 0;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->formerBlockDim = 1;
    tilingDatafromBin->tailBlockDim = 2;
    tilingDatafromBin->ubRDimFactor = 5;
    tilingDatafromBin->ubRDimFactorAlign = 8;
    tilingDatafromBin->ubRDimLoopNum = 1;
    tilingDatafromBin->ubRDimTail = 0;
    tilingDatafromBin->ubRDimTailFactor = 0;
    tilingDatafromBin->ubRDimTailFactorAlign = 0;
    tilingDatafromBin->ubRDimTailLoopNum = 0;
    tilingDatafromBin->ubRDimTailTail = 0;
    tilingDatafromBin->ubRDimTailTailFactor = 0;
    tilingDatafromBin->ubRDimTailTailFactorAlign = 0;
    tilingDatafromBin->ubRDimTailTailLoopNum = 0;
    tilingDatafromBin->enableDx = 1;
    tilingDatafromBin->enableDgamma = 1;
    tilingDatafromBin->enableDbeta = 1;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey, blockNum, (uint8_t *)tilingDatafromBin);
}

TEST_F(batch_norm_grad_test, test_infer_splitR0_float32)
{
    std::vector<int64_t> xShape = {1, 192, 192, 1, 1};
    std::vector<int64_t> wShape = {1};
    std::string dtype = "float";
    uint64_t tilingKey = 910002;
    uint32_t blockNum = 3;
    size_t tilingSize = sizeof(BatchNormGradInferTilingData) + sizeof(float);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    BatchNormGradInferTilingData* tilingDatafromBin = reinterpret_cast<BatchNormGradInferTilingData*>(tiling);

    tilingDatafromBin->baseTilingData.totalTiles = 3;
    tilingDatafromBin->baseTilingData.tilesPerCore = 1;
    tilingDatafromBin->baseTilingData.usedCoreNums = 3;
    tilingDatafromBin->baseTilingData.r1Dim = 1;
    tilingDatafromBin->baseTilingData.aDim = 1;
    tilingDatafromBin->baseTilingData.r0Dim = 36864;
    tilingDatafromBin->baseTilingData.r1Outer = 1;
    tilingDatafromBin->baseTilingData.aOuter = 1;
    tilingDatafromBin->baseTilingData.r0Outer = 3;
    tilingDatafromBin->baseTilingData.tileBlockR1Len = 1;
    tilingDatafromBin->baseTilingData.tileBlockR1Tail = 1;
    tilingDatafromBin->baseTilingData.tileBlockALen = 1;
    tilingDatafromBin->baseTilingData.tileBlockATail = 1;
    tilingDatafromBin->baseTilingData.tileBlockR0Len = 15808;
    tilingDatafromBin->baseTilingData.tileBlockR0Tail = 5248;
    tilingDatafromBin->baseTilingData.tileBlockAPaddingNum = 0;
    tilingDatafromBin->baseTilingData.epsilon = 0;
    tilingDatafromBin->generalBinAddTilingData.binaryAddQuotient = 8192;
    tilingDatafromBin->generalBinAddTilingData.binaryAddk = 1;
    tilingDatafromBin->generalBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->tailBinAddTilingData.binaryAddQuotient = 4096;
    tilingDatafromBin->tailBinAddTilingData.binaryAddk = 0;
    tilingDatafromBin->tailBinAddTilingData.binaryAddLastNum = 64;
    tilingDatafromBin->blockNum = 1;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->formerBlockDim = 1;
    tilingDatafromBin->tailBlockDim = 2;
    tilingDatafromBin->ubRDimFactor = 15392;
    tilingDatafromBin->ubRDimFactorAlign = 15392;
    tilingDatafromBin->ubRDimLoopNum = 2;
    tilingDatafromBin->ubRDimTail = 6080;
    tilingDatafromBin->ubRDimTailFactor = 7696;
    tilingDatafromBin->ubRDimTailFactorAlign = 7696;
    tilingDatafromBin->ubRDimTailLoopNum = 0;
    tilingDatafromBin->ubRDimTailTail = 6080;
    tilingDatafromBin->ubRDimTailTailFactor = 7696;
    tilingDatafromBin->ubRDimTailTailFactorAlign = 7696;
    tilingDatafromBin->ubRDimTailTailLoopNum = 2;
    tilingDatafromBin->enableDx = 1;
    tilingDatafromBin->enableDgamma = 1;
    tilingDatafromBin->enableDbeta = 1;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey, blockNum, (uint8_t *)tilingDatafromBin);
}