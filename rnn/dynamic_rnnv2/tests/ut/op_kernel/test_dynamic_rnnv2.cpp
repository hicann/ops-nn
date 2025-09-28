/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <unistd.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include <iostream>
#include <string>
#include "data_utils.h"
#include "dynamic_rnnv2_tiling_def.h"
#include "string.h"
#endif
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
using namespace std;

extern "C" __global__ __aicore__ void dynamic_rnn_v2(
    GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias, GM_ADDR seqLength,
    GM_ADDR initH, GM_ADDR initC,
    GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo,
    GM_ADDR mask,
    GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
    GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
    GM_ADDR workspace, GM_ADDR rnnTiling);


class dynamic_r_n_n_v2_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_r_n_n_v2_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "dynamic_r_n_n_v2_test TearDown\n" << std::endl;
    }
};

TEST_F(dynamic_r_n_n_v2_test, test_case_1) {
    size_t time = 1;
    size_t batch = 16;
    size_t input_size = 512;
    size_t hidden_size = 512;

    int32_t block_dim = 24;

    string path = get_current_dir_name();

    system("cp -r ../../../../rnn/dynamic_rnnv2/tests/ut/op_kernel/rnnv2_data ./");
    system("chmod -R 755 ./rnnv2_data/");
    system("cd ./rnnv2_data/ && rm -rf ./*bin");
    system("cd ./rnnv2_data/ && python3 gen_data.py 1 16 512 512");

    size_t aGMByteSize = time * batch * input_size * sizeof(float);
    size_t b1GMByteSize = input_size * 4 * hidden_size * sizeof(float);
    size_t b2GMByteSize = hidden_size * 4 * hidden_size * sizeof(float);
    size_t initHCGMByteSize = batch * hidden_size * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicRNNTilingData);
    size_t cGMByteSize = time * batch * 4 * hidden_size * sizeof(float);
    size_t workspaceGMByteSize = 96*1024*1024;
    size_t biasGMByteSize = 4 * hidden_size * sizeof(float);

    uint8_t* cGM = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM1 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM2 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM3 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM4 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM5 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM6 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM7 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);

    uint8_t* workspaceGM = (uint8_t*)AscendC::GmAlloc(workspaceGMByteSize);
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(biasGMByteSize);
    uint8_t* goldenGM = (uint8_t*)AscendC::GmAlloc(cGMByteSize);

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(aGMByteSize);
    uint8_t* b1GM = (uint8_t*)AscendC::GmAlloc(b1GMByteSize);
    uint8_t* b2GM = (uint8_t*)AscendC::GmAlloc(b2GMByteSize);
    uint8_t* initHGM = (uint8_t*)AscendC::GmAlloc(initHCGMByteSize);
    uint8_t* initCGM = (uint8_t*)AscendC::GmAlloc(initHCGMByteSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    DynamicRNNTilingData* tilingDatafromBin = reinterpret_cast<DynamicRNNTilingData*>(tiling);
    TCubeTiling inputMMTilingData;
    TCubeTiling hiddenMMTilingData;
    tilingDatafromBin->tilingKey = 10000001;
    tilingDatafromBin->usedCoreNum = 48;
    tilingDatafromBin->timeStep = 1;
    tilingDatafromBin->batch = 16;
    tilingDatafromBin->inputSize = 512;
    tilingDatafromBin->hiddenSize = 512;
    tilingDatafromBin->isBias = 1;
    tilingDatafromBin->isInithc = 1;
    tilingDatafromBin->isSeqLength = 0;
    tilingDatafromBin->isHF32 = 0;
    tilingDatafromBin->isCached = 14141414;
    tilingDatafromBin->cacheLength = 0;
    tilingDatafromBin->gateOrder = 1;
    tilingDatafromBin->direction = 0;
    tilingDatafromBin->isTraining = 1;
    tilingDatafromBin->cellClip = -1.0;
    tilingDatafromBin->forgetBias = 0.0;

    inputMMTilingData.usedCoreNum = 8;
    inputMMTilingData.M = 16;
    inputMMTilingData.N = 2048;
    inputMMTilingData.Ka = 512;
    inputMMTilingData.Kb = 512;
    inputMMTilingData.singleCoreM = 16;
    inputMMTilingData.singleCoreN = 256;
    inputMMTilingData.singleCoreK = 512;
    inputMMTilingData.baseM = 16;
    inputMMTilingData.baseN = 256;
    inputMMTilingData.baseK = 32;
    inputMMTilingData.depthA1 = 16;
    inputMMTilingData.depthB1 = 8;
    inputMMTilingData.stepM = 1;
    inputMMTilingData.stepN = 1;
    inputMMTilingData.isBias = 1;
    inputMMTilingData.transLength = 32768;
    inputMMTilingData.iterateOrder = 1;
    inputMMTilingData.shareMode = 0;
    inputMMTilingData.shareL1Size = 295936;
    inputMMTilingData.shareL0CSize = 16384;
    inputMMTilingData.shareUbSize = 0;
    inputMMTilingData.batchM = 1;
    inputMMTilingData.batchN = 1;
    inputMMTilingData.singleBatchM = 1;
    inputMMTilingData.singleBatchN = 1;
    inputMMTilingData.stepKa = 16;
    inputMMTilingData.stepKb = 4;
    inputMMTilingData.dbL0A = 2;
    inputMMTilingData.dbL0B = 2;
    inputMMTilingData.dbL0C = 1;
    inputMMTilingData.ALayoutInfoB = 0;
    inputMMTilingData.ALayoutInfoS = 0;
    inputMMTilingData.ALayoutInfoN = 0;
    inputMMTilingData.ALayoutInfoG = 0;
    inputMMTilingData.ALayoutInfoD = 0;
    inputMMTilingData.BLayoutInfoB = 0;
    inputMMTilingData.BLayoutInfoS = 0;
    inputMMTilingData.BLayoutInfoN = 0;
    inputMMTilingData.BLayoutInfoG = 0;
    inputMMTilingData.BLayoutInfoD = 0;
    inputMMTilingData.CLayoutInfoB = 0;
    inputMMTilingData.CLayoutInfoS1 = 0;
    inputMMTilingData.CLayoutInfoN = 0;
    inputMMTilingData.CLayoutInfoG = 0;
    inputMMTilingData.CLayoutInfoS2 = 0;
    inputMMTilingData.BatchNum = 0;
    tilingDatafromBin->inputMMParam = inputMMTilingData;

    hiddenMMTilingData.usedCoreNum = 8;
    hiddenMMTilingData.M = 16;
    hiddenMMTilingData.N = 2048;
    hiddenMMTilingData.Ka = 512;
    hiddenMMTilingData.Kb = 512;
    hiddenMMTilingData.singleCoreM = 16;
    hiddenMMTilingData.singleCoreN = 256;
    hiddenMMTilingData.singleCoreK = 512;
    hiddenMMTilingData.baseM = 16;
    hiddenMMTilingData.baseN = 256;
    hiddenMMTilingData.baseK = 32;
    hiddenMMTilingData.depthA1 = 16;
    hiddenMMTilingData.depthB1 = 8;
    hiddenMMTilingData.stepM = 1;
    hiddenMMTilingData.stepN = 1;
    hiddenMMTilingData.isBias = 0;
    hiddenMMTilingData.transLength = 32768;
    hiddenMMTilingData.iterateOrder = 1;
    hiddenMMTilingData.shareMode = 0;
    hiddenMMTilingData.shareL1Size = 294912;
    hiddenMMTilingData.shareL0CSize = 16384;
    hiddenMMTilingData.shareUbSize = 0;
    hiddenMMTilingData.batchM = 1;
    hiddenMMTilingData.batchN = 1;
    hiddenMMTilingData.singleBatchM = 1;
    hiddenMMTilingData.singleBatchN = 1;
    hiddenMMTilingData.stepKa = 16;
    hiddenMMTilingData.stepKb = 4;
    hiddenMMTilingData.dbL0A = 2;
    hiddenMMTilingData.dbL0B = 2;
    hiddenMMTilingData.dbL0C = 1;
    hiddenMMTilingData.ALayoutInfoB = 0;
    hiddenMMTilingData.ALayoutInfoS = 0;
    hiddenMMTilingData.ALayoutInfoN = 0;
    hiddenMMTilingData.ALayoutInfoG = 0;
    hiddenMMTilingData.ALayoutInfoD = 0;
    hiddenMMTilingData.BLayoutInfoB = 0;
    hiddenMMTilingData.BLayoutInfoS = 0;
    hiddenMMTilingData.BLayoutInfoN = 0;
    hiddenMMTilingData.BLayoutInfoG = 0;
    hiddenMMTilingData.BLayoutInfoD = 0;
    hiddenMMTilingData.CLayoutInfoB = 0;
    hiddenMMTilingData.CLayoutInfoS1 = 0;
    hiddenMMTilingData.CLayoutInfoN = 0;
    hiddenMMTilingData.CLayoutInfoG = 0;
    hiddenMMTilingData.CLayoutInfoS2 = 0;
    hiddenMMTilingData.BatchNum = 0;
    tilingDatafromBin->hiddenMMParam = hiddenMMTilingData;

    ReadFile(path + "/rnnv2_data/x.bin", aGMByteSize, aGM, aGMByteSize);
    ReadFile(path + "/rnnv2_data/w1.bin", b1GMByteSize, b1GM, b1GMByteSize);
    ReadFile(path + "/rnnv2_data/w2.bin", b2GMByteSize, b2GM, b2GMByteSize);
    ReadFile(path + "/rnnv2_data/b.bin", biasGMByteSize, biasGM, biasGMByteSize);
    ReadFile(path + "/rnnv2_data/out.bin", cGMByteSize, goldenGM, cGMByteSize);
    ReadFile(path + "/rnnv2_data/inth.bin", initHCGMByteSize, initHGM, initHCGMByteSize);
    ReadFile(path + "/rnnv2_data/intc.bin", initHCGMByteSize, initCGM, initHCGMByteSize);

    for (int i = 0; i < cGMByteSize / sizeof(float); i++) {
        *((float*)cGM + i) = i;
        *((half*)outGM1 + i) = i;
        *((half*)outGM2 + i) = i;
    }
    ICPU_SET_TILING_KEY(10000001);
    ICPU_RUN_KF(dynamic_rnn_v2, block_dim, aGM, b1GM, b2GM, biasGM, nullptr, initHGM, initCGM,
                nullptr, nullptr, nullptr, nullptr,
                cGM, outGM1, outGM2, outGM3, outGM4, outGM5, outGM6, outGM7,
                workspaceGM, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(aGM);
    AscendC::GmFree(b1GM);
    AscendC::GmFree(b2GM);
    AscendC::GmFree(cGM);
    AscendC::GmFree(outGM1);
    AscendC::GmFree(outGM2);
    AscendC::GmFree(outGM3);
    AscendC::GmFree(outGM4);
    AscendC::GmFree(outGM5);
    AscendC::GmFree(outGM6);
    AscendC::GmFree(outGM7);
    AscendC::GmFree(workspaceGM);
    AscendC::GmFree(biasGM);
}


TEST_F(dynamic_r_n_n_v2_test, test_case_2) {
    size_t time = 1;
    size_t batch = 16;
    size_t input_size = 512;
    size_t hidden_size = 512;

    int32_t block_dim = 24;

    string path = get_current_dir_name();

    system("cp -r ../../../../rnn/dynamic_rnnv2/tests/ut/op_kernel/rnnv2_data ./");
    system("chmod -R 755 ./rnnv2_data/");
    system("cd ./rnnv2_data/ && rm -rf ./*bin");
    system("cd ./rnnv2_data/ && python3 gen_data.py 1 16 512 512");

    size_t aGMByteSize = time * batch * input_size * sizeof(float);
    size_t b1GMByteSize = input_size * 4 * hidden_size * sizeof(float);
    size_t b2GMByteSize = hidden_size * 4 * hidden_size * sizeof(float);
    size_t initHCGMByteSize = batch * hidden_size * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicRNNTilingData);
    size_t cGMByteSize = time * batch * 4 * hidden_size * sizeof(float);
    size_t workspaceGMByteSize = 96*1024*1024;
    size_t biasGMByteSize = 4 * hidden_size * sizeof(float);

    uint8_t* cGM = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM1 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM2 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM3 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM4 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM5 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM6 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);
    uint8_t* outGM7 = (uint8_t*)AscendC::GmAlloc(cGMByteSize);

    uint8_t* workspaceGM = (uint8_t*)AscendC::GmAlloc(workspaceGMByteSize);
    uint8_t* biasGM = (uint8_t*)AscendC::GmAlloc(biasGMByteSize);
    uint8_t* goldenGM = (uint8_t*)AscendC::GmAlloc(cGMByteSize);

    uint8_t* aGM = (uint8_t*)AscendC::GmAlloc(aGMByteSize);
    uint8_t* b1GM = (uint8_t*)AscendC::GmAlloc(b1GMByteSize);
    uint8_t* b2GM = (uint8_t*)AscendC::GmAlloc(b2GMByteSize);
    uint8_t* initHGM = (uint8_t*)AscendC::GmAlloc(initHCGMByteSize);
    uint8_t* initCGM = (uint8_t*)AscendC::GmAlloc(initHCGMByteSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    DynamicRNNTilingData* tilingDatafromBin = reinterpret_cast<DynamicRNNTilingData*>(tiling);
    TCubeTiling inputMMTilingData;
    TCubeTiling hiddenMMTilingData;
    tilingDatafromBin->tilingKey = 10000002;
    tilingDatafromBin->usedCoreNum = 48;
    tilingDatafromBin->timeStep = 1;
    tilingDatafromBin->batch = 16;
    tilingDatafromBin->inputSize = 512;
    tilingDatafromBin->hiddenSize = 512;
    tilingDatafromBin->isBias = 1;
    tilingDatafromBin->isInithc = 1;
    tilingDatafromBin->isSeqLength = 0;
    tilingDatafromBin->isHF32 = 0;
    tilingDatafromBin->isCached = 14141414;
    tilingDatafromBin->cacheLength = 0;
    tilingDatafromBin->gateOrder = 1;
    tilingDatafromBin->direction = 0;
    tilingDatafromBin->isTraining = 1;
    tilingDatafromBin->cellClip = -1.0;
    tilingDatafromBin->forgetBias = 0.0;

    inputMMTilingData.usedCoreNum = 8;
    inputMMTilingData.M = 16;
    inputMMTilingData.N = 2048;
    inputMMTilingData.Ka = 512;
    inputMMTilingData.Kb = 512;
    inputMMTilingData.singleCoreM = 16;
    inputMMTilingData.singleCoreN = 256;
    inputMMTilingData.singleCoreK = 512;
    inputMMTilingData.baseM = 16;
    inputMMTilingData.baseN = 256;
    inputMMTilingData.baseK = 32;
    inputMMTilingData.depthA1 = 16;
    inputMMTilingData.depthB1 = 8;
    inputMMTilingData.stepM = 1;
    inputMMTilingData.stepN = 1;
    inputMMTilingData.isBias = 1;
    inputMMTilingData.transLength = 32768;
    inputMMTilingData.iterateOrder = 1;
    inputMMTilingData.shareMode = 0;
    inputMMTilingData.shareL1Size = 295936;
    inputMMTilingData.shareL0CSize = 16384;
    inputMMTilingData.shareUbSize = 0;
    inputMMTilingData.batchM = 1;
    inputMMTilingData.batchN = 1;
    inputMMTilingData.singleBatchM = 1;
    inputMMTilingData.singleBatchN = 1;
    inputMMTilingData.stepKa = 16;
    inputMMTilingData.stepKb = 4;
    inputMMTilingData.dbL0A = 2;
    inputMMTilingData.dbL0B = 2;
    inputMMTilingData.dbL0C = 1;
    inputMMTilingData.ALayoutInfoB = 0;
    inputMMTilingData.ALayoutInfoS = 0;
    inputMMTilingData.ALayoutInfoN = 0;
    inputMMTilingData.ALayoutInfoG = 0;
    inputMMTilingData.ALayoutInfoD = 0;
    inputMMTilingData.BLayoutInfoB = 0;
    inputMMTilingData.BLayoutInfoS = 0;
    inputMMTilingData.BLayoutInfoN = 0;
    inputMMTilingData.BLayoutInfoG = 0;
    inputMMTilingData.BLayoutInfoD = 0;
    inputMMTilingData.CLayoutInfoB = 0;
    inputMMTilingData.CLayoutInfoS1 = 0;
    inputMMTilingData.CLayoutInfoN = 0;
    inputMMTilingData.CLayoutInfoG = 0;
    inputMMTilingData.CLayoutInfoS2 = 0;
    inputMMTilingData.BatchNum = 0;
    tilingDatafromBin->inputMMParam = inputMMTilingData;

    hiddenMMTilingData.usedCoreNum = 8;
    hiddenMMTilingData.M = 16;
    hiddenMMTilingData.N = 2048;
    hiddenMMTilingData.Ka = 512;
    hiddenMMTilingData.Kb = 512;
    hiddenMMTilingData.singleCoreM = 16;
    hiddenMMTilingData.singleCoreN = 256;
    hiddenMMTilingData.singleCoreK = 512;
    hiddenMMTilingData.baseM = 16;
    hiddenMMTilingData.baseN = 256;
    hiddenMMTilingData.baseK = 32;
    hiddenMMTilingData.depthA1 = 16;
    hiddenMMTilingData.depthB1 = 8;
    hiddenMMTilingData.stepM = 1;
    hiddenMMTilingData.stepN = 1;
    hiddenMMTilingData.isBias = 0;
    hiddenMMTilingData.transLength = 32768;
    hiddenMMTilingData.iterateOrder = 1;
    hiddenMMTilingData.shareMode = 0;
    hiddenMMTilingData.shareL1Size = 294912;
    hiddenMMTilingData.shareL0CSize = 16384;
    hiddenMMTilingData.shareUbSize = 0;
    hiddenMMTilingData.batchM = 1;
    hiddenMMTilingData.batchN = 1;
    hiddenMMTilingData.singleBatchM = 1;
    hiddenMMTilingData.singleBatchN = 1;
    hiddenMMTilingData.stepKa = 16;
    hiddenMMTilingData.stepKb = 4;
    hiddenMMTilingData.dbL0A = 2;
    hiddenMMTilingData.dbL0B = 2;
    hiddenMMTilingData.dbL0C = 1;
    hiddenMMTilingData.ALayoutInfoB = 0;
    hiddenMMTilingData.ALayoutInfoS = 0;
    hiddenMMTilingData.ALayoutInfoN = 0;
    hiddenMMTilingData.ALayoutInfoG = 0;
    hiddenMMTilingData.ALayoutInfoD = 0;
    hiddenMMTilingData.BLayoutInfoB = 0;
    hiddenMMTilingData.BLayoutInfoS = 0;
    hiddenMMTilingData.BLayoutInfoN = 0;
    hiddenMMTilingData.BLayoutInfoG = 0;
    hiddenMMTilingData.BLayoutInfoD = 0;
    hiddenMMTilingData.CLayoutInfoB = 0;
    hiddenMMTilingData.CLayoutInfoS1 = 0;
    hiddenMMTilingData.CLayoutInfoN = 0;
    hiddenMMTilingData.CLayoutInfoG = 0;
    hiddenMMTilingData.CLayoutInfoS2 = 0;
    hiddenMMTilingData.BatchNum = 0;
    tilingDatafromBin->hiddenMMParam = hiddenMMTilingData;

    ReadFile(path + "/rnnv2_data/x.bin", aGMByteSize, aGM, aGMByteSize);
    ReadFile(path + "/rnnv2_data/w1.bin", b1GMByteSize, b1GM, b1GMByteSize);
    ReadFile(path + "/rnnv2_data/w2.bin", b2GMByteSize, b2GM, b2GMByteSize);
    ReadFile(path + "/rnnv2_data/b.bin", biasGMByteSize, biasGM, biasGMByteSize);
    ReadFile(path + "/rnnv2_data/out.bin", cGMByteSize, goldenGM, cGMByteSize);
    ReadFile(path + "/rnnv2_data/inth.bin", initHCGMByteSize, initHGM, initHCGMByteSize);
    ReadFile(path + "/rnnv2_data/intc.bin", initHCGMByteSize, initCGM, initHCGMByteSize);

    for (int i = 0; i < cGMByteSize / sizeof(float); i++) {
        *((float*)cGM + i) = i;
        *((half*)outGM1 + i) = i;
        *((half*)outGM2 + i) = i;
    }
    ICPU_SET_TILING_KEY(10000002);
    ICPU_RUN_KF(dynamic_rnn_v2, block_dim, aGM, b1GM, b2GM, biasGM, nullptr, initHGM, initCGM,
                nullptr, nullptr, nullptr, nullptr,
                cGM, outGM1, outGM2, outGM3, outGM4, outGM5, outGM6, outGM7,
                workspaceGM, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(aGM);
    AscendC::GmFree(b1GM);
    AscendC::GmFree(b2GM);
    AscendC::GmFree(cGM);
    AscendC::GmFree(outGM1);
    AscendC::GmFree(outGM2);
    AscendC::GmFree(outGM3);
    AscendC::GmFree(outGM4);
    AscendC::GmFree(outGM5);
    AscendC::GmFree(outGM6);
    AscendC::GmFree(outGM7);
    AscendC::GmFree(workspaceGM);
    AscendC::GmFree(biasGM);
}
