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

// #include "increDebug.h"
#include <iostream>
#include <string>

#include "data_utils.h"
#include "batch_mat_mul_v3_tiling_def.h"
#include "batch_mat_mul_v3.cpp"
#include "string.h"
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
// #include "kernel_operator.h"
using namespace std;

class batch_mat_mul_v3_test : public testing::Test {
   protected:
    static void SetUpTestCase() { cout << "batch_mat_mul_v3_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "batch_mat_mul_v3_test TearDown\n" << endl; }
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};


TEST_F(batch_mat_mul_v3_test, batch_mat_mul_v3_test_0) {
    // {{1,1,1,1, 32, 128}, {1,1,1,1,128, 128}}
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t shape_a = 32 * 128 * sizeof(uint16_t);
    size_t shape_b = 128 * 128 * sizeof(uint16_t);
    size_t shape_output = 32 * 128 * sizeof(uint16_t);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(BatchMatmulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(shape_a);
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(shape_b);
    uint8_t *biasGM = nullptr;
    uint8_t *offsetWGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(shape_output);
    uint8_t *contextGM = (uint8_t *)AscendC::GmAlloc(sizeof(HcclCombinOpParam));
    memset(aGM, 0, shape_a);
    memset(bGM, 0, shape_b);
    memset(output, 0, shape_output);
    system("cp -r ../../../../matmul/batch_mat_mul_v3/tests/ut/op_kernel/batch_mat_mul_v3_data ./");
    system("chmod -R 755 ./batch_mat_mul_v3_data/");
    system("cd ./batch_mat_mul_v3_data/ && rm -rf ./*bin");
    system("cd ./batch_mat_mul_v3_data/ && python3 gen_data.py 1 32 128 128");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/batch_mat_mul_v3_data/shape_a.bin", shape_a, aGM, shape_a);
    ReadFile(path + "/batch_mat_mul_v3_data/shape_b.bin", shape_b, bGM, shape_b);
    ReadFile(path + "/batch_mat_mul_v3_data/shape_output.bin", shape_output, output, shape_output);

    BatchMatmulTilingData *tiling_data = reinterpret_cast<BatchMatmulTilingData*>(tiling);

    tiling_data->matmulTiling.matmulTiling.usedCoreNum = 24;
    tiling_data->matmulTiling.matmulTiling.M = 32;
    tiling_data->matmulTiling.matmulTiling.N = 128;
    tiling_data->matmulTiling.matmulTiling.Ka = 128;
    tiling_data->matmulTiling.matmulTiling.Kb = 128;
    tiling_data->matmulTiling.matmulTiling.singleCoreM = 32;
    tiling_data->matmulTiling.matmulTiling.singleCoreN = 128;
    tiling_data->matmulTiling.matmulTiling.singleCoreK = 128;
    tiling_data->matmulTiling.matmulTiling.baseM = 32;
    tiling_data->matmulTiling.matmulTiling.baseN = 128;
    tiling_data->matmulTiling.matmulTiling.baseK = 64;
    tiling_data->matmulTiling.matmulTiling.depthA1 = 2;
    tiling_data->matmulTiling.matmulTiling.depthB1 = 2;
    tiling_data->matmulTiling.matmulTiling.stepM = 1;
    tiling_data->matmulTiling.matmulTiling.stepN = 1;
    tiling_data->matmulTiling.matmulTiling.stepKa = 2;
    tiling_data->matmulTiling.matmulTiling.stepKb = 2;
    tiling_data->matmulTiling.matmulTiling.isBias = 0;
    tiling_data->matmulTiling.matmulTiling.transLength = 131072;
    tiling_data->matmulTiling.matmulTiling.iterateOrder = 0;
    tiling_data->matmulTiling.matmulTiling.shareL1Size = 458752;
    tiling_data->matmulTiling.matmulTiling.shareL0CSize = 131072;
    tiling_data->matmulTiling.matmulTiling.shareUbSize = 0;
    tiling_data->matmulTiling.matmulTiling.batchM = 1;
    tiling_data->matmulTiling.matmulTiling.batchN = 1;
    tiling_data->matmulTiling.matmulTiling.singleBatchM = 1;
    tiling_data->matmulTiling.matmulTiling.singleBatchN = 1;
    tiling_data->matmulTiling.matmulTiling.BatchNum = 1;
    tiling_data->matmulTiling.matmulRunInfo.transA = 0;
    tiling_data->matmulTiling.matmulRunInfo.transB = 0;
    tiling_data->matmulTiling.matmulRunInfo.nd2nzA = 0;
    tiling_data->matmulTiling.matmulRunInfo.nd2nzB = 0;
    tiling_data->multiBatchInfo.batchUsedCoreNum = 24;
    tiling_data->multiBatchInfo.aBatchDimAll = 1;
    tiling_data->multiBatchInfo.bBatchDimAll = 1;
    tiling_data->multiBatchInfo.cBatchDimAll = 1;
    tiling_data->multiBatchInfo.aBatchDim0 = 1;
    tiling_data->multiBatchInfo.bBatchDim0 = 1;
    tiling_data->multiBatchInfo.cBatchDim0 = 1;
    tiling_data->multiBatchInfo.aBatchDim1 = 1;
    tiling_data->multiBatchInfo.bBatchDim1 = 1;
    tiling_data->multiBatchInfo.cBatchDim1 = 1;
    tiling_data->multiBatchInfo.aBatchDim2 = 1;
    tiling_data->multiBatchInfo.bBatchDim2 = 1;
    tiling_data->multiBatchInfo.cBatchDim2 = 1;
    tiling_data->multiBatchInfo.aBatchDim3 = 1;
    tiling_data->multiBatchInfo.bBatchDim3 = 1;
    tiling_data->multiBatchInfo.cBatchDim3 = 1;
    tiling_data->multiBatchInfo.iterBatch = 1;
    ICPU_SET_TILING_KEY(10000000000000001001UL);

    auto batch_mat_mul_v3_wrapper = [](GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR contextGM, GM_ADDR y, GM_ADDR workspace,
                                 GM_ADDR tiling) {
        ::batch_mat_mul_v3<0, 1, 0, 0, 1>(x, w, bias, contextGM, y, workspace, tiling);
    };
    ICPU_RUN_KF(batch_mat_mul_v3_wrapper, 24, aGM, bGM, nullptr, contextGM, output, workspace, tiling);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    free(path_);
}
