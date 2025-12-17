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

#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include <iostream>
#include <string>
#include "data_utils.h"
#include "string.h"
#include "tikicpulib.h"
#include "quant_batch_matmul_v4_tiling_def.h"
#include "quant_batch_matmul_v4.cpp"
#endif

#include <cstdint>

using namespace std;
// using namespace AscendC;

class quant_batch_matmul_v4_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "quant_batch_matmul_v4_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "quant_batch_matmul_v4_test TearDown\n" << endl;
    }
};

TEST_F(quant_batch_matmul_v4_test, quant_batch_matmul_v4_perblock_0)
{
    // m = 10, k = 1024, n = 1536, A8W8 Perblock
    // baseM = 128, baseK = 128, baseN = 256
    // transA = true, transB = false, groupSizeM = 1, groupSizeK = 128, groupSizeN = 128, outDtype = bf16
    size_t m = 128;
    size_t k = 1024;
    size_t n = 1536;
    size_t nkGroup = 8; // ceildiv(k, 128)
    size_t nNGroup = 12; // ceildiv(n, 128)
    size_t x1Size = m * k * sizeof(int8_t);
    size_t x2Size = n * k * sizeof(int8_t);
    size_t biasSize = n * sizeof(float);
    size_t x1ScaleSize = m * nkGroup * sizeof(float);
    size_t x2ScaleSize = nNGroup * nkGroup * sizeof(float);
    size_t ySize = m * n * sizeof(uint16_t);
    size_t tilingSize = sizeof(QuantBatchMatmulV4PerblockTilingData);

    size_t sysWorkspaceSize = 20 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;

    uint8_t *x1GM = (uint8_t *)AscendC::GmAlloc(x1Size);
    uint8_t *x2GM = (uint8_t *)AscendC::GmAlloc(x2Size);
    uint8_t *biasGM = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *x1ScaleGM = (uint8_t *)AscendC::GmAlloc(x1ScaleSize);
    uint8_t *x2ScaleGM = (uint8_t *)AscendC::GmAlloc(x2ScaleSize);
    uint8_t *yScaleGM = nullptr;
    uint8_t *x1OffsetGM = nullptr;
    uint8_t *x2OffsetGM = nullptr;
    uint8_t *yOffsetGM = nullptr;
    uint8_t *x2TableGM = nullptr;
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    memset(x1GM, 0, x1Size);
    memset(x2GM, 0, x2Size);
    memset(biasGM, 0, biasSize);
    memset(x1ScaleGM, 0, x1ScaleSize);
    memset(x2ScaleGM, 0, x2ScaleSize);
    memset(yGM, 0, ySize);

    system("cp -r ../../../../matmul/quant_batch_matmul_v4/tests/ut/op_kernel/quant_batch_matmul_v4_data ./");
    system("chmod -R 755 ./quant_batch_matmul_v4_data/");
    system("cd ./quant_batch_matmul_v4_data/ && rm -rf ./*bin");
    system("cd ./quant_batch_matmul_v4_data/ && python3 gen_data.py 128 1024 1536 4303356032 5");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/quant_batch_matmul_v4_data/x1.bin", x1Size, x1GM, x1Size);
    ReadFile(path + "/quant_batch_matmul_v4_data/x2.bin", x2Size, x2GM, x2Size);
    ReadFile(path + "/quant_batch_matmul_v4_data/bias.bin", biasSize, biasGM, biasSize);
    ReadFile(path + "/quant_batch_matmul_v4_data/x1_scale.bin", x1ScaleSize, x1ScaleGM, x1ScaleSize);
    ReadFile(path + "/quant_batch_matmul_v4_data/x2_scale.bin", x2ScaleSize, x2ScaleGM, x2ScaleSize);

    QuantBatchMatmulV4PerblockTilingData* tiling_data = reinterpret_cast<QuantBatchMatmulV4PerblockTilingData*>(tiling);
    tiling_data->matmulTiling.usedCoreNum = 6;
    tiling_data->matmulTiling.M = m;
    tiling_data->matmulTiling.N = n;
    tiling_data->matmulTiling.Ka = k;
    tiling_data->matmulTiling.Kb = k;
    tiling_data->matmulTiling.singleCoreM = 128;
    tiling_data->matmulTiling.singleCoreN = 256;
    tiling_data->matmulTiling.singleCoreK = k;
    tiling_data->matmulTiling.baseM = 128;
    tiling_data->matmulTiling.baseN = 256;
    tiling_data->matmulTiling.baseK = 128;
    tiling_data->matmulTiling.depthA1 = 8;
    tiling_data->matmulTiling.depthB1 = 8;
    tiling_data->matmulTiling.stepM = 1;
    tiling_data->matmulTiling.stepN = 1;
    tiling_data->matmulTiling.stepKa = 4;
    tiling_data->matmulTiling.stepKb = 4;
    tiling_data->matmulTiling.isBias = 0;
    tiling_data->matmulTiling.transLength = 0;
    tiling_data->matmulTiling.iterateOrder = 0;
    tiling_data->matmulTiling.shareL1Size = 0;
    tiling_data->matmulTiling.shareL0CSize = 0;
    tiling_data->matmulTiling.shareUbSize = 0;
    tiling_data->matmulTiling.batchM = 0;
    tiling_data->matmulTiling.batchN = 0;
    tiling_data->matmulTiling.singleBatchM = 0;
    tiling_data->matmulTiling.singleBatchN = 0;
    tiling_data->matmulTiling.depthAL1CacheUB = 0;
    tiling_data->matmulTiling.depthBL1CacheUB = 0;
    tiling_data->matmulTiling.dbL0A = 2;
    tiling_data->matmulTiling.dbL0B = 2;
    tiling_data->matmulTiling.dbL0C = 1;
    tiling_data->tileL2cacheTiling.mTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.nTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.mTileBlock = 1;
    tiling_data->tileL2cacheTiling.nTileBlock = 6;
    tiling_data->tileL2cacheTiling.calOrder = 1;
    tiling_data->groupSizeM = 1;
    tiling_data->groupSizeK = 128;
    tiling_data->groupSizeN = 128;
    tiling_data->ubCalcM = 64;
    tiling_data->ubCalcN = 256;
    tiling_data->transA = false;
    tiling_data->transB = true;

    auto quant_batch_matmul_v4_wrapper = [](GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale,
                                            GM_ADDR y_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y_offset,
                                            GM_ADDR x2_table, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::quant_batch_matmul_v4<1, 4, 0, 0, 1>(
            x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, x2_table, y, workspace, tiling);
    };

    ICPU_RUN_KF(
        quant_batch_matmul_v4_wrapper, 6, x1GM, x2GM, biasGM, x1ScaleGM, x2ScaleGM, yScaleGM, x1OffsetGM, x2OffsetGM,
        yOffsetGM, x2TableGM, yGM, workspace, tiling);
    
    AscendC::GmFree((void*)x1GM);
    AscendC::GmFree((void*)x2GM);
    AscendC::GmFree((void*)biasGM);
    AscendC::GmFree((void*)x1ScaleGM);
    AscendC::GmFree((void*)x2ScaleGM);
    AscendC::GmFree((void*)yGM);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}