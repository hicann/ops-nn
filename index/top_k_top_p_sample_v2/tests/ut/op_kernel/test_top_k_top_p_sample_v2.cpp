/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "top_k_top_p_sample_v2_tiling_def.h"
#include "data_utils.h"

#include <cstdint>
using namespace std;

extern "C" __global__ __aicore__ void top_k_top_p_sample_v2(
    GM_ADDR logits, GM_ADDR topKs, GM_ADDR topPs, GM_ADDR q, GM_ADDR minPs, GM_ADDR logitsSelectIdx, GM_ADDR logitsTopKpSelect, GM_ADDR logitsIdx, GM_ADDR logitsSortMasked,
    GM_ADDR workspace, GM_ADDR tiling);

class top_k_top_p_sample_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "top_k_top_p_sample_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "top_k_top_p_sample_v2_test TearDown\n" << endl;
    }
};

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_half)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 1 1024 -1 1 1 1 0 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_bf16)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(float);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(float);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(float);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 1 1024 0 1 0 1 0 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;
    

    uint64_t tillingKey = 1027;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_float)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(float);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(float);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(float);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 1 1024 0 1 2 1 0 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;
    

    uint64_t tillingKey = 1000;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_isNeedLogits)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(float);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(float);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(float);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 1 1024 0 1 2 1 0 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 1;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;
    

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));
    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_isNeedSampleResult)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(float);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(float);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(float);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 1 1024 0 1 2 1 0 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 1;
    

    uint64_t tillingKey = 1000;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_inputIsLogits)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(float);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(float);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(float);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 1 1024 0 1 2 0 0 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 0;
    topKTopPSampleV2TilingData->isNeedSampleResult = 1;
    

    uint64_t tillingKey = 1000;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_qsample)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 1 1024 0 1 1 1 1 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/q.bin", qSize, q, qSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_isInputMinPs)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 1 1024 0 1 1 1 0 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/minPs.bin", minPsSize, minPs, minPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_skipTopK)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 -10 0 0 1 1 1 0 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_skipTopP)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 1 1024 1 2 1 1 0 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_LogitsGuessK)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 -10 0 0 1 1 1 0 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_guessK)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 -10 0 0 1 1 0 0 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 0;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_sortAll)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 -10 0 0 1 1 1 1 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 1;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 0;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_sortAllmultinomial)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 -10 0 0 1 1 1 0 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 1;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 0;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 1;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_skipTopKTopPSoftmax)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 -10 0 1 2 1 0 1 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 0;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_skipTopKTopP)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 -10 0 1 2 1 1 1 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 1;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 32;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_v2_test, top_k_top_p_sample_v2_test_isInputMinPsGlobal)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t minPsSize = 8 * sizeof(half);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t logitsIdxSize = 8 * 64 * sizeof(int64_t);
    size_t logitsSortMaskedSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleV2TilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* minPs = (uint8_t*)AscendC::GmAlloc(minPsSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* logitsIdx = (uint8_t*)AscendC::GmAlloc(logitsIdxSize);
    uint8_t* logitsSortMasked = (uint8_t*)AscendC::GmAlloc(logitsSortMaskedSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample_v2/tests/ut/op_kernel/top_k_top_p_sample_v2_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_v2_data/");
    system("cd ./top_k_top_p_sample_v2_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_v2_data/ && python3 gen_data.py 8 64 0 10000 -2 -1 0 1 1 1 0 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_v2_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/topPs.bin", topPsSize, topPs, topPsSize);
    ReadFile(path + "/top_k_top_p_sample_v2_data/minPs.bin", minPsSize, minPs, minPsSize);
    TopKTopPSampleV2TilingData* topKTopPSampleV2TilingData = reinterpret_cast<TopKTopPSampleV2TilingData*>(tiling);



    topKTopPSampleV2TilingData->numCore = 40;
    topKTopPSampleV2TilingData->rowNum = 8;
    topKTopPSampleV2TilingData->rowLen = 64;
    topKTopPSampleV2TilingData->headCoreNum = topKTopPSampleV2TilingData->rowNum % topKTopPSampleV2TilingData->numCore;//rowNum % numCore
    topKTopPSampleV2TilingData->perHeadCoreRowNum = (topKTopPSampleV2TilingData->rowNum + topKTopPSampleV2TilingData->numCore - 1) / topKTopPSampleV2TilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleV2TilingData->tailCoreRowNum =  topKTopPSampleV2TilingData->rowNum / topKTopPSampleV2TilingData->numCore;
    topKTopPSampleV2TilingData->perHeadCorePartNum = 0;
    topKTopPSampleV2TilingData->tailCorePartNum = 0;
    topKTopPSampleV2TilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleV2TilingData->innerLoopTime = (topKTopPSampleV2TilingData->rowLen + topKTopPSampleV2TilingData->innerLoopEle -1) / topKTopPSampleV2TilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTail = topKTopPSampleV2TilingData->rowLen % topKTopPSampleV2TilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleV2TilingData->innerLoopEleTailPad = (topKTopPSampleV2TilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleV2TilingData->softmaxLoopTime = (topKTopPSampleV2TilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTail = topKTopPSampleV2TilingData->rowLen % (32768/4);
    topKTopPSampleV2TilingData->softmaxLoopEleTailPad = (topKTopPSampleV2TilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleV2TilingData->eightKPartNum = (topKTopPSampleV2TilingData->rowLen + 1023)  / 1024;
    topKTopPSampleV2TilingData->eightKPartTail = topKTopPSampleV2TilingData->rowLen % 1024;
    topKTopPSampleV2TilingData->eightKPartTailPad =  (topKTopPSampleV2TilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleV2TilingData->mrgMode = 1;
    topKTopPSampleV2TilingData->headOffset = topKTopPSampleV2TilingData->headCoreNum * topKTopPSampleV2TilingData->perHeadCoreRowNum * topKTopPSampleV2TilingData->rowLen;
    topKTopPSampleV2TilingData->isNeedLogits = 0;
    topKTopPSampleV2TilingData->eps = 1e-8;
    topKTopPSampleV2TilingData->topKGuess = 0;
    topKTopPSampleV2TilingData->ksMAX = 1024;
    topKTopPSampleV2TilingData->inputIsLogits = 1;
    topKTopPSampleV2TilingData->isNeedSampleResult = 0;

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample_v2, 40, logits, topKs, topPs, q, minPs, logitsSelectIdx, logitsTopKpSelect, logitsIdx, logitsSortMasked, workspace, (uint8_t*)(topKTopPSampleV2TilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)minPs);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)logitsIdx);
    AscendC::GmFree((void *)logitsSortMasked);
    AscendC::GmFree((void *)tiling);

    free(path_);
}