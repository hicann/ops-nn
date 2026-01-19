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

#include <unistd.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"

#include "fused_quant_mat_mul_tiling_def.h"
#include "fused_quant_mat_mul.cpp"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
using namespace std;

class fused_quant_mat_mul_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "fused_quant_mat_mul_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "fused_quant_mat_mul_test TearDown\n" << endl;
    }
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

TEST_F(fused_quant_mat_mul_test, fused_quant_mat_mul_test_1) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t m = 60;
    size_t k = 787;
    size_t n = 11;

    size_t x1Size = m * k * sizeof(int8_t);
    size_t x2Size = n * k * sizeof(int8_t);
    size_t biasSize = n * sizeof(float);
    size_t x1ScaleSize = m * sizeof(float);
    size_t x2ScaleSize = 1 * sizeof(float); // pertensor
    size_t ySize = m * n * sizeof(uint16_t);
    size_t tilingSize = sizeof(QuantBatchMatmulV3TilingData);

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
    uint8_t *x3GM = nullptr;
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(allWorkspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    memset(x1GM, 0, x1Size);
    memset(x2GM, 0, x2Size);
    memset(biasGM, 0, biasSize);
    memset(x1ScaleGM, 0, x1ScaleSize);
    memset(x2ScaleGM, 0, x2ScaleSize);
    memset(yGM, 0, ySize);

    system("cp -r ../../../../matmul/fused_quant_mat_mul/tests/ut/op_kernel/fused_quant_mat_mul_data ./");
    system("chmod -R 755 ./fused_quant_mat_mul_data/");
    system("cd ./fused_quant_mat_mul_data/ && rm -rf ./*bin");
    system("cd ./fused_quant_mat_mul_data/ && python3 gen_data_fp32.py 60 787 11 0"); // 0:x2pertensor 1:x2perchannel

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/fused_quant_mat_mul_data/x1.bin", x1Size, x1GM, x1Size);
    ReadFile(path + "/fused_quant_mat_mul_data/x2.bin", x2Size, x2GM, x2Size);
    ReadFile(path + "/fused_quant_mat_mul_data/bias.bin", biasSize, biasGM, biasSize);
    ReadFile(path + "/fused_quant_mat_mul_data/x1_scale.bin", x1ScaleSize, x1ScaleGM, x1ScaleSize);
    ReadFile(path + "/fused_quant_mat_mul_data/x2_scale.bin", x2ScaleSize, x2ScaleGM, x2ScaleSize);

    QuantBatchMatmulV3TilingData *tiling_data = reinterpret_cast<QuantBatchMatmulV3TilingData*>(tiling);
    tiling_data->params.batchA = 1;
    tiling_data->params.batchB= 1;
    tiling_data->params.batchC = 1;
    tiling_data->params.batchA1 = 0;
    tiling_data->params.batchA2 = 0;
    tiling_data->params.batchA3 = 0;
    tiling_data->params.batchA4 = 0;
    tiling_data->params.batchB1 = 0;
    tiling_data->params.batchB2 = 0;
    tiling_data->params.batchB3 = 0;
    tiling_data->params.batchB4 = 0;
    tiling_data->params.batchC1 = 0;
    tiling_data->params.batchC2 = 0;
    tiling_data->params.batchC3 = 0;
    tiling_data->params.batchC4 = 0;
    tiling_data->params.singleCoreBatch = 0;
    tiling_data->params.isPerTensor = 1;
    tiling_data->params.isPertoken = 1;
    tiling_data->params.isDoubleScale = 0;
    tiling_data->params.biasThreeDim = 0;
    tiling_data->params.ubCalcM = 60;
    tiling_data->params.ubCalcN = 16;
    tiling_data->params.needUbBuffer = 7680;
    tiling_data->params.realSingleCoreM = 0;
    tiling_data->params.realSingleCoreN = 0;
    tiling_data->params.biasDtype = 0; 
    tiling_data->params.ubSize = 0;
    tiling_data->params.isMClash = 0;
    tiling_data->params.isNClash = 0;
    tiling_data->params.groupSizeM = 0;
    tiling_data->params.groupSizeN = 0;
    tiling_data->params.groupSizeK = 0;

    tiling_data->matmulTiling.usedCoreNum = 1;
    tiling_data->matmulTiling.M = 60;
    tiling_data->matmulTiling.N = 11;
    tiling_data->matmulTiling.Ka = 787;
    tiling_data->matmulTiling.Kb = 787;
    tiling_data->matmulTiling.singleCoreM = 64;
    tiling_data->matmulTiling.singleCoreN = 16;
    tiling_data->matmulTiling.singleCoreK = 787;
    tiling_data->matmulTiling.baseM = 64;
    tiling_data->matmulTiling.baseN = 16;
    tiling_data->matmulTiling.baseK = 512;
    tiling_data->matmulTiling.depthA1 = 8;
    tiling_data->matmulTiling.depthB1 = 8;
    tiling_data->matmulTiling.stepM = 1;
    tiling_data->matmulTiling.stepN = 1;
    tiling_data->matmulTiling.isBias = 1;
    tiling_data->matmulTiling.transLength = 0;
    tiling_data->matmulTiling.iterateOrder = 0;
    tiling_data->matmulTiling.shareMode = 0; 
    tiling_data->matmulTiling.shareL1Size = 98432;
    tiling_data->matmulTiling.shareL0CSize = 8192;
    tiling_data->matmulTiling.shareUbSize = 0;
    tiling_data->matmulTiling.batchM = 1;
    tiling_data->matmulTiling.batchN = 1;
    tiling_data->matmulTiling.singleBatchM = 1;
    tiling_data->matmulTiling.singleBatchN = 1;
    tiling_data->matmulTiling.stepKa = 4;
    tiling_data->matmulTiling.stepKb = 4;
    tiling_data->matmulTiling.depthAL1CacheUB = 0;
    tiling_data->matmulTiling.depthBL1CacheUB = 0;
    tiling_data->matmulTiling.dbL0A = 2;
    tiling_data->matmulTiling.dbL0B = 2;
    tiling_data->matmulTiling.dbL0C = 2;

    tiling_data->tileL2cacheTiling.mTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.nTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.mTileBlock = 1;
    tiling_data->tileL2cacheTiling.nTileBlock = 1;
    tiling_data->tileL2cacheTiling.calOrder = 1;
    tiling_data->tileL2cacheTiling.isBasicTiling = 1;

    auto fused_quant_mat_mul_wrapper = [](GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale,
                                            GM_ADDR y_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y_offset,
                                            GM_ADDR x2_table, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::fused_quant_mat_mul<1, 1, 1, 0, 3>(
            x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, x2_table, x3, y, workspace, tiling);
    };

    ICPU_RUN_KF(
        fused_quant_mat_mul_wrapper, 24, x1GM, x2GM, biasGM, x1ScaleGM, x2ScaleGM, yScaleGM, x1OffsetGM, x2OffsetGM,
        yOffsetGM, x2TableGM, x3GM, yGM, workspace, tiling);

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
