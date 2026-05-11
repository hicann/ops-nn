/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <array>
#include <vector>

#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#include "../transpose_quant_batch_mat_mul_tiling_def.h"
#include "arch35/transpose_quant_batch_mat_mul.cpp"
#endif

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

using namespace std;

class transpose_quant_batch_mat_mul_test : public testing::Test {
   protected:
    static void SetUpTestCase() { cout << "transpose_quant_batch_mat_mul_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "transpose_quant_batch_mat_mul_test TearDown\n" << endl; }
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

// Test 1: MXFP8, medium size (M=64, Batch=1, K=64, N=64)
TEST_F(transpose_quant_batch_mat_mul_test, transpose_quant_batch_mat_mul_fp8_medium) {
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    int32_t M = 64;
    int32_t Batch = 1;
    int32_t K = 64;
    int32_t N = 64;

    size_t shape_x1 = M * Batch * K * sizeof(int8_t);
    size_t shape_x2 = Batch * K * N * sizeof(int8_t);
    size_t shape_x1_scale = M  * sizeof(float);
    size_t shape_x2_scale = N  * sizeof(float);
    size_t shape_output = M * Batch * N * sizeof(uint16_t);

    size_t sysWorkspaceSize = 20UL * 1024UL * 1024UL;
    size_t usrWorkspaceSize = shape_x1 + shape_x2;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(BatchMatMulV3TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *x1GM = (uint8_t *)AscendC::GmAlloc(shape_x1);
    uint8_t *x2GM = (uint8_t *)AscendC::GmAlloc(shape_x2);
    uint8_t *biasGM = nullptr;
    uint8_t *x1_scaleGM = (uint8_t *)AscendC::GmAlloc(shape_x1_scale);
    uint8_t *x2_scaleGM = (uint8_t *)AscendC::GmAlloc(shape_x2_scale);
    uint8_t *outputGM = (uint8_t *)AscendC::GmAlloc(shape_output);
    uint8_t *contextGM = (uint8_t *)AscendC::GmAlloc(sizeof(HcclCombinOpParam));
    
    memset(x1GM, 0, shape_x1);
    memset(x2GM, 0, shape_x2);
    memset(x1_scaleGM, 0, shape_x1_scale);
    memset(x2_scaleGM, 0, shape_x2_scale);
    memset(outputGM, 0, shape_output);

    system("cp -r ../../../../matmul/transpose_quant_batch_mat_mul/tests/ut/op_kernel/transpose_quant_batch_mat_mul_data ./");
    system("chmod -R 755 ./transpose_quant_batch_mat_mul_data/");
    system("cd ./transpose_quant_batch_mat_mul_data/ && rm -rf ./*bin");
    system("cd ./transpose_quant_batch_mat_mul_data/ && python3 gen_data.py 64 1 64 64 1,0,2 0,1,2 1,0,2 0");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/transpose_quant_batch_mat_mul_data/shape_x1.bin", shape_x1, x1GM, shape_x1);
    ReadFile(path + "/transpose_quant_batch_mat_mul_data/shape_x2.bin", shape_x2, x2GM, shape_x2);
    ReadFile(path + "/transpose_quant_batch_mat_mul_data/shape_x1_scale.bin", shape_x1_scale, x1_scaleGM, shape_x1_scale);
    ReadFile(path + "/transpose_quant_batch_mat_mul_data/shape_x2_scale.bin", shape_x2_scale, x2_scaleGM, shape_x2_scale);
    ReadFile(path + "/transpose_quant_batch_mat_mul_data/shape_output.bin", shape_output, outputGM, shape_output);

    BatchMatMulV3TilingData *tiling_data = reinterpret_cast<BatchMatMulV3TilingData*>(tiling);
    
    tiling_data->matMulTilingData.tCubeTiling.usedCoreNum = 4;
    tiling_data->matMulTilingData.tCubeTiling.M = M;
    tiling_data->matMulTilingData.tCubeTiling.N = N;
    tiling_data->matMulTilingData.tCubeTiling.Ka = K;
    tiling_data->matMulTilingData.tCubeTiling.Kb = K;
    tiling_data->matMulTilingData.tCubeTiling.singleCoreM = 64;
    tiling_data->matMulTilingData.tCubeTiling.singleCoreN = 64;
    tiling_data->matMulTilingData.tCubeTiling.singleCoreK = 64;
    tiling_data->matMulTilingData.tCubeTiling.baseM = 64;
    tiling_data->matMulTilingData.tCubeTiling.baseN = 64;
    tiling_data->matMulTilingData.tCubeTiling.baseK = 64;
    tiling_data->matMulTilingData.tCubeTiling.depthA1 = 1;
    tiling_data->matMulTilingData.tCubeTiling.depthB1 = 1;
    tiling_data->matMulTilingData.tCubeTiling.stepM = 1;
    tiling_data->matMulTilingData.tCubeTiling.stepN = 1;
    tiling_data->matMulTilingData.tCubeTiling.isBias = 0;
    tiling_data->matMulTilingData.tCubeTiling.transLength = 0;
    tiling_data->matMulTilingData.tCubeTiling.iterateOrder = 0;
    tiling_data->matMulTilingData.tCubeTiling.shareMode = 0;
    tiling_data->matMulTilingData.tCubeTiling.shareL1Size = 40960;
    tiling_data->matMulTilingData.tCubeTiling.shareL0CSize = 16384;
    tiling_data->matMulTilingData.tCubeTiling.shareUbSize = 0;
    tiling_data->matMulTilingData.tCubeTiling.batchM = 1;
    tiling_data->matMulTilingData.tCubeTiling.batchN = 1;
    tiling_data->matMulTilingData.tCubeTiling.singleBatchM = 1;
    tiling_data->matMulTilingData.tCubeTiling.singleBatchN = 1;
    tiling_data->matMulTilingData.tCubeTiling.stepKa = 1;
    tiling_data->matMulTilingData.tCubeTiling.stepKb = 1;
    tiling_data->matMulTilingData.tCubeTiling.depthAL1CacheUB = 0;
    tiling_data->matMulTilingData.tCubeTiling.depthBL1CacheUB = 0;
    tiling_data->matMulTilingData.tCubeTiling.dbL0A = 2;
    tiling_data->matMulTilingData.tCubeTiling.dbL0B = 2;
    tiling_data->matMulTilingData.tCubeTiling.dbL0C = 1;
    tiling_data->matMulTilingData.mTailCnt = 0;
    tiling_data->matMulTilingData.nTailCnt = 0;
    tiling_data->matMulTilingData.kTailCnt = 0;
    tiling_data->matMulTilingData.mBaseTailSplitCnt = 1;
    tiling_data->matMulTilingData.nBaseTailSplitCnt = 1;
    tiling_data->matMulTilingData.mTailMain = 0;
    tiling_data->matMulTilingData.nTailMain = 0;
    tiling_data->matMulTilingData.isHf32 = 0;
    tiling_data->matMulTilingData.aswWindowLen = 0;
    tiling_data->matMulTilingData.l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;

    tiling_data->aBatchDimAll = Batch;
    tiling_data->bBatchDimAll = Batch;
    tiling_data->cBatchDimAll = Batch;
    tiling_data->biasBatchDimAll = 1;
    tiling_data->aBatchDim0 = 1;
    tiling_data->bBatchDim0 = 1;
    tiling_data->cBatchDim0 = 1;
    tiling_data->aBatchDim1 = Batch;
    tiling_data->bBatchDim1 = Batch;
    tiling_data->cBatchDim1 = Batch;
    tiling_data->aBatchDim2 = 1;
    tiling_data->bBatchDim2 = 1;
    tiling_data->cBatchDim2 = 1;
    tiling_data->aBatchDim3 = 1;
    tiling_data->bBatchDim3 = 1;
    tiling_data->cBatchDim3 = 1;
    tiling_data->iterBatch = 1;
    tiling_data->batchOutNum = 1;
    tiling_data->batchSplitFactor = 1;

    auto transpose_quant_batch_mat_mul_wrapper = [](GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, 
                                                     GM_ADDR x2_scale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::transpose_quant_batch_mat_mul<TRANSPOSE_QUANT_BATCH_MAT_MUL_PERM_X1_1_0_2, 
                                        TRANSPOSE_QUANT_BATCH_MAT_MUL_PERM_X2_0_1_2, 
                                        TRANSPOSE_QUANT_BATCH_MAT_MUL_BATCH_SPLIT_FALSE, 
                                        TRANSPOSE_QUANT_BATCH_MAT_MUL_FP8>(x1, x2, bias, x1_scale, x2_scale, y, workspace, tiling);
    };
    ICPU_RUN_KF(transpose_quant_batch_mat_mul_wrapper, 4, x1GM, x2GM, nullptr, x1_scaleGM, x2_scaleGM, outputGM, workspace, tiling);
    
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)x1GM);
    AscendC::GmFree((void*)x2GM);
    AscendC::GmFree((void*)x1_scaleGM);
    AscendC::GmFree((void*)x2_scaleGM);
    AscendC::GmFree((void*)outputGM);
    AscendC::GmFree((void*)contextGM);
    free(path_);
}
