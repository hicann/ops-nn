/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "top_k_top_p_sample_tiling_def.h"
#include "data_utils.h"

#include <cstdint>
using namespace std;

extern "C" __global__ __aicore__ void top_k_top_p_sample(
    GM_ADDR logits, GM_ADDR topKs, GM_ADDR topPs, GM_ADDR q, GM_ADDR logitsSelectIdx, GM_ADDR logitsTopKpSelect,
    GM_ADDR workspace, GM_ADDR tiling);

class top_k_top_p_sample_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "top_k_top_p_sample_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "top_k_top_p_sample_test TearDown\n" << endl;
    }
};

TEST_F(top_k_top_p_sample_test, top_k_top_p_sample_test_half)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(int64_t);
    size_t tilingSize = sizeof(TopKTopPSampleTilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample/tests/ut/op_kernel/top_k_top_p_sample_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_data/");
    system("cd ./top_k_top_p_sample_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_data/ && python3 gen_data.py 8 64 0 10000 1 1024 0 1 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleTilingData* topKTopPSampleTilingData = reinterpret_cast<TopKTopPSampleTilingData*>(tiling);



    topKTopPSampleTilingData->numCore = 40;
    topKTopPSampleTilingData->rowNum = 8;
    topKTopPSampleTilingData->rowLen = 64;
    topKTopPSampleTilingData->headCoreNum = topKTopPSampleTilingData->rowNum % topKTopPSampleTilingData->numCore;//rowNum % numCore
    topKTopPSampleTilingData->perHeadCoreRowNum = (topKTopPSampleTilingData->rowNum + topKTopPSampleTilingData->numCore - 1) / topKTopPSampleTilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleTilingData->tailCoreRowNum =  topKTopPSampleTilingData->rowNum / topKTopPSampleTilingData->numCore;
    topKTopPSampleTilingData->perHeadCorePartNum = 0;
    topKTopPSampleTilingData->tailCorePartNum = 0;
    topKTopPSampleTilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleTilingData->innerLoopTime = (topKTopPSampleTilingData->rowLen + topKTopPSampleTilingData->innerLoopEle -1) / topKTopPSampleTilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleTilingData->innerLoopEleTail = topKTopPSampleTilingData->rowLen % topKTopPSampleTilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleTilingData->innerLoopEleTailPad = (topKTopPSampleTilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleTilingData->softmaxLoopTime = (topKTopPSampleTilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleTilingData->softmaxLoopEleTail = topKTopPSampleTilingData->rowLen % (32768/4);
    topKTopPSampleTilingData->softmaxLoopEleTailPad = (topKTopPSampleTilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleTilingData->eightKPartNum = (topKTopPSampleTilingData->rowLen + 1023)  / 1024;
    topKTopPSampleTilingData->eightKPartTail = topKTopPSampleTilingData->rowLen % 1024;
    topKTopPSampleTilingData->eightKPartTailPad =  (topKTopPSampleTilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleTilingData->mrgMode = 1;
    topKTopPSampleTilingData->isNeedLogits = 0;
    topKTopPSampleTilingData->eps = 1e-8;
    topKTopPSampleTilingData->topKGuess = 32;
    

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample, 40, logits, topKs, topPs, q, logitsSelectIdx, logitsTopKpSelect, workspace, (uint8_t*)(topKTopPSampleTilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_test, top_k_top_p_sample_test_bf16)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(float);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(float);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(int64_t);
    size_t tilingSize = sizeof(TopKTopPSampleTilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample/tests/ut/op_kernel/top_k_top_p_sample_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_data/");
    system("cd ./top_k_top_p_sample_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_data/ && python3 gen_data.py 8 64 0 10000 1 1024 0 1 0");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleTilingData* topKTopPSampleTilingData = reinterpret_cast<TopKTopPSampleTilingData*>(tiling);



    topKTopPSampleTilingData->numCore = 40;
    topKTopPSampleTilingData->rowNum = 8;
    topKTopPSampleTilingData->rowLen = 64;
    topKTopPSampleTilingData->headCoreNum = topKTopPSampleTilingData->rowNum % topKTopPSampleTilingData->numCore;//rowNum % numCore
    topKTopPSampleTilingData->perHeadCoreRowNum = (topKTopPSampleTilingData->rowNum + topKTopPSampleTilingData->numCore - 1) / topKTopPSampleTilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleTilingData->tailCoreRowNum =  topKTopPSampleTilingData->rowNum / topKTopPSampleTilingData->numCore;
    topKTopPSampleTilingData->perHeadCorePartNum = 0;
    topKTopPSampleTilingData->tailCorePartNum = 0;
    topKTopPSampleTilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleTilingData->innerLoopTime = (topKTopPSampleTilingData->rowLen + topKTopPSampleTilingData->innerLoopEle -1) / topKTopPSampleTilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleTilingData->innerLoopEleTail = topKTopPSampleTilingData->rowLen % topKTopPSampleTilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleTilingData->innerLoopEleTailPad = (topKTopPSampleTilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleTilingData->softmaxLoopTime = (topKTopPSampleTilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleTilingData->softmaxLoopEleTail = topKTopPSampleTilingData->rowLen % (32768/4);
    topKTopPSampleTilingData->softmaxLoopEleTailPad = (topKTopPSampleTilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleTilingData->eightKPartNum = (topKTopPSampleTilingData->rowLen + 1023)  / 1024;
    topKTopPSampleTilingData->eightKPartTail = topKTopPSampleTilingData->rowLen % 1024;
    topKTopPSampleTilingData->eightKPartTailPad =  (topKTopPSampleTilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleTilingData->mrgMode = 1;
    topKTopPSampleTilingData->isNeedLogits = 0;
    topKTopPSampleTilingData->eps = 1e-8;
    topKTopPSampleTilingData->topKGuess = 32;
    

    uint64_t tillingKey = 1027;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample, 40, logits, topKs, topPs, q, logitsSelectIdx, logitsTopKpSelect, workspace, (uint8_t*)(topKTopPSampleTilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(top_k_top_p_sample_test, top_k_top_p_sample_test_isNeedLogits)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t logitsSize = 8 * 64 * sizeof(half);
    size_t topKsSize = 8 * sizeof(int32_t);
    size_t topPsSize = 8 * sizeof(half);
    size_t qSize = 8 * 64 * sizeof(float);
    size_t logitsSelectIdxSize = 8 * sizeof(int64_t);
    size_t logitsTopKpSelectSize = 8 * 64 * sizeof(float);
    size_t tilingSize = sizeof(TopKTopPSampleTilingData);

    size_t workspaceSize = 40 * 1024 * 1024 + 128 * 256 * sizeof(float) * 6;

    uint8_t* logits = (uint8_t*)AscendC::GmAlloc(logitsSize);
    uint8_t* topKs = (uint8_t*)AscendC::GmAlloc(topKsSize);
    uint8_t* topPs = (uint8_t*)AscendC::GmAlloc(topPsSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* logitsSelectIdx = (uint8_t*)AscendC::GmAlloc(logitsSelectIdxSize);
    uint8_t* logitsTopKpSelect = (uint8_t*)AscendC::GmAlloc(logitsTopKpSelectSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system("cp -r ../../../../index/top_k_top_p_sample/tests/ut/op_kernel/top_k_top_p_sample_data ./");
    system("chmod -R 755 ./top_k_top_p_sample_data/");
    system("cd ./top_k_top_p_sample_data/ && rm -rf ./*bin");
    system("cd ./top_k_top_p_sample_data/ && python3 gen_data.py 8 64 0 10000 1 1024 0 1 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/top_k_top_p_sample_data/logits.bin", logitsSize, logits, logitsSize);
    ReadFile(path + "/top_k_top_p_sample_data/topKs.bin", topKsSize, topKs, topKsSize);
    ReadFile(path + "/top_k_top_p_sample_data/topPs.bin", topPsSize, topPs, topPsSize);
    TopKTopPSampleTilingData* topKTopPSampleTilingData = reinterpret_cast<TopKTopPSampleTilingData*>(tiling);



    topKTopPSampleTilingData->numCore = 40;
    topKTopPSampleTilingData->rowNum = 8;
    topKTopPSampleTilingData->rowLen = 64;
    topKTopPSampleTilingData->headCoreNum = topKTopPSampleTilingData->rowNum % topKTopPSampleTilingData->numCore;//rowNum % numCore
    topKTopPSampleTilingData->perHeadCoreRowNum = (topKTopPSampleTilingData->rowNum + topKTopPSampleTilingData->numCore - 1) / topKTopPSampleTilingData->numCore;//(rowNum + numCore -1) / numCore
    topKTopPSampleTilingData->tailCoreRowNum =  topKTopPSampleTilingData->rowNum / topKTopPSampleTilingData->numCore;
    topKTopPSampleTilingData->perHeadCorePartNum = 0;
    topKTopPSampleTilingData->tailCorePartNum = 0;
    topKTopPSampleTilingData->innerLoopEle = 4096 * 2;
    topKTopPSampleTilingData->innerLoopTime = (topKTopPSampleTilingData->rowLen + topKTopPSampleTilingData->innerLoopEle -1) / topKTopPSampleTilingData->innerLoopEle;//(rowLen + innerLoopEle -1) / innerLoopEle
    topKTopPSampleTilingData->innerLoopEleTail = topKTopPSampleTilingData->rowLen % topKTopPSampleTilingData->innerLoopEle;//rowLen % innerLoopEle
    topKTopPSampleTilingData->innerLoopEleTailPad = (topKTopPSampleTilingData->innerLoopEleTail + 31) / 32 * 32;//safeCeli(innerLoopEleTail, BlockByte(32)) * BlockByte
    topKTopPSampleTilingData->softmaxLoopTime = (topKTopPSampleTilingData->rowLen + (32768/4)-1) / (32768/4);
    topKTopPSampleTilingData->softmaxLoopEleTail = topKTopPSampleTilingData->rowLen % (32768/4);
    topKTopPSampleTilingData->softmaxLoopEleTailPad = (topKTopPSampleTilingData->softmaxLoopEleTail  + 31) / 32 * 32;
    topKTopPSampleTilingData->eightKPartNum = (topKTopPSampleTilingData->rowLen + 1023)  / 1024;
    topKTopPSampleTilingData->eightKPartTail = topKTopPSampleTilingData->rowLen % 1024;
    topKTopPSampleTilingData->eightKPartTailPad =  (topKTopPSampleTilingData->eightKPartTail + 31) / 32 * 32;
    topKTopPSampleTilingData->mrgMode = 1;
    topKTopPSampleTilingData->isNeedLogits = 1;
    topKTopPSampleTilingData->eps = 1e-8;
    topKTopPSampleTilingData->topKGuess = 32;
    

    uint64_t tillingKey = 1001;
    ICPU_SET_TILING_KEY(tillingKey);
    ICPU_RUN_KF(top_k_top_p_sample, 40, logits, topKs, topPs, q, logitsSelectIdx, logitsTopKpSelect, workspace, (uint8_t*)(topKTopPSampleTilingData));

    AscendC::GmFree((void *)logits);
    AscendC::GmFree((void *)topKs);
    AscendC::GmFree((void *)topPs);
    AscendC::GmFree((void *)q);
    AscendC::GmFree((void *)logitsSelectIdx);
    AscendC::GmFree((void *)logitsTopKpSelect);
    AscendC::GmFree((void *)tiling);

    free(path_);
}