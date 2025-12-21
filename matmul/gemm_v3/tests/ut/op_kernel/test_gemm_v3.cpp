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

//#include "increDebug.h"
#include "gemm_v3_tiling_def.h"
#include "gemm_v3.cpp"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
using namespace std;

extern "C" __global__ __aicore__ void gemm_v3(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGm,
    GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);

class gemm_v3_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "gemm_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "gemm_v3_test TearDown\n" << endl;
    }
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

TEST_F(gemm_v3_test, gemm_v3_test_1) {
    // {{512, 512}, {512, 512}}
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t shape_a = 512 * 512 * sizeof(uint16_t);
    size_t shape_b = 512 * 512 * sizeof(uint16_t);
    size_t shape_c = 512 * 512 * sizeof(uint32_t);
    size_t shape_y = 512 * 512 * sizeof(uint32_t);

    size_t sysWorkspaceSize = 20 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(shape_a);
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(shape_b);
    uint8_t *cGM = (uint8_t *)AscendC::GmAlloc(shape_c);
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(shape_y);

    uint8_t *contextGM = (uint8_t *)AscendC::GmAlloc(sizeof(HcclCombinOpParam));

    memset(aGM, 0, shape_a);
    memset(bGM, 0, shape_b);
    memset(cGM, 0, shape_c);
    memset(yGM, 0, shape_y);

    system("cp -r ../../../../matmul/gemm_v3/tests/ut/op_kernel/gemm_v3_data ./");
    system("chmod -R 755 ./gemm_v3_data/");
    system("cd ./gemm_v3_data/ && rm -rf ./*bin");
    system("cd ./gemm_v3_data/ && python3 gen_data.py 512 512 512");
    // system("cd ./fa_data/ && python3 gen_tiling.py case2");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/gemm_v3_data/shape_a.bin", shape_a, aGM, shape_a);
    ReadFile(path + "/gemm_v3_data/shape_b.bin", shape_b, bGM, shape_b);
    ReadFile(path + "/gemm_v3_data/shape_c.bin", shape_c, cGM, shape_c);
    ReadFile(path + "/gemm_v3_data/shape_y.bin", shape_y, yGM, shape_y);

    // mock tilingData
    MatmulTilingData *tiling_data = reinterpret_cast<MatmulTilingData*>(tiling);
    tiling_data->matmulTiling.usedCoreNum = 24;
    tiling_data->matmulTiling.M = 512;
    tiling_data->matmulTiling.N = 512;
    tiling_data->matmulTiling.Ka = 512;
    tiling_data->matmulTiling.Kb = 512;
    tiling_data->matmulTiling.singleCoreM = 96;
    tiling_data->matmulTiling.singleCoreN = 128;
    tiling_data->matmulTiling.singleCoreK = 512;
    tiling_data->matmulTiling.baseM = 96;
    tiling_data->matmulTiling.baseN = 128;
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
    tiling_data->matmulTiling.shareL1Size = 311296;
    tiling_data->matmulTiling.shareL0CSize = 49152;
    tiling_data->matmulTiling.shareUbSize = 0;
    tiling_data->matmulTiling.batchM = 1;
    tiling_data->matmulTiling.batchN = 1;
    tiling_data->matmulTiling.singleBatchM = 1;
    tiling_data->matmulTiling.singleBatchN = 1;
    tiling_data->matmulRunInfo.transA = 0;
    tiling_data->matmulRunInfo.transB = 0;
    tiling_data->matmulRunInfo.nd2nzA = 0;
    tiling_data->matmulRunInfo.nd2nzB = 0;
    tiling_data->tileL2cacheTiling.mTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.nTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.mTileBlock = 6;
    tiling_data->tileL2cacheTiling.nTileBlock = 4;
    tiling_data->tileL2cacheTiling.calOrder = 0;
    tiling_data->l2cacheUseInfo.l2CacheFlag = 0;
    
    auto wrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        ::gemm_v3<0, 0, 0, 1, 0>(aGM, bGM, cGM, yGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(wrapper, 24, aGM, bGM, cGM, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)cGM);
    AscendC::GmFree((void*)yGM);
    free(path_);
}

TEST_F(gemm_v3_test, gemm_v3_test_2) {
    // {{512, 512}, {512, 3168}}
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t shape_a = 512 * 512 * sizeof(uint16_t);
    size_t shape_b = 512 * 3168 * sizeof(uint16_t);
    size_t shape_c = 512 * 3168 * sizeof(uint32_t);
    size_t shape_y = 512 * 3168 * sizeof(uint32_t);

    size_t sysWorkspaceSize = 20 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(shape_a);
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(shape_b);
    uint8_t *cGM = (uint8_t *)AscendC::GmAlloc(shape_c);
    uint8_t *yGM = (uint8_t *)AscendC::GmAlloc(shape_y);

    uint8_t *contextGM = (uint8_t *)AscendC::GmAlloc(sizeof(HcclCombinOpParam));

    memset(aGM, 0, shape_a);
    memset(bGM, 0, shape_b);
    memset(cGM, 0, shape_c);
    memset(yGM, 0, shape_y);

    system("cp -r ../../../../matmul/gemm_v3/tests/ut/op_kernel/gemm_v3_data ./");
    system("chmod -R 755 ./gemm_v3_data/");
    system("cd ./gemm_v3_data/ && rm -rf ./*bin");
    system("cd ./gemm_v3_data/ && python3 gen_data.py 512 512 3168");
    // system("cd ./fa_data/ && python3 gen_tiling.py case2");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/gemm_v3_data/shape_a.bin", shape_a, aGM, shape_a);
    ReadFile(path + "/gemm_v3_data/shape_b.bin", shape_b, bGM, shape_b);
    ReadFile(path + "/gemm_v3_data/shape_c.bin", shape_c, cGM, shape_c);
    ReadFile(path + "/gemm_v3_data/shape_y.bin", shape_y, yGM, shape_y);

    // mock tilingData
    MatmulTilingData *tiling_data = reinterpret_cast<MatmulTilingData*>(tiling);
    tiling_data->matmulTiling.usedCoreNum = 24;
    tiling_data->matmulTiling.M = 512;
    tiling_data->matmulTiling.N = 3168;
    tiling_data->matmulTiling.Ka = 512;
    tiling_data->matmulTiling.Kb = 512;
    tiling_data->matmulTiling.singleCoreM = 112;
    tiling_data->matmulTiling.singleCoreN = 256;
    tiling_data->matmulTiling.singleCoreK = 512;
    tiling_data->matmulTiling.baseM = 112;
    tiling_data->matmulTiling.baseN = 256;
    tiling_data->matmulTiling.baseK = 64;
    tiling_data->matmulTiling.depthA1 = 16;
    tiling_data->matmulTiling.depthB1 = 8;
    tiling_data->matmulTiling.stepM = 1;
    tiling_data->matmulTiling.stepN = 1;
    tiling_data->matmulTiling.stepKa = 8;
    tiling_data->matmulTiling.stepKb = 4;
    tiling_data->matmulTiling.isBias = 0;
    tiling_data->matmulTiling.transLength = 0;
    tiling_data->matmulTiling.iterateOrder = 0;
    tiling_data->matmulTiling.shareL1Size = 262144;
    tiling_data->matmulTiling.shareL0CSize = 131072;
    tiling_data->matmulTiling.shareUbSize = 0;
    tiling_data->matmulTiling.batchM = 1;
    tiling_data->matmulTiling.batchN = 1;
    tiling_data->matmulTiling.singleBatchM = 1;
    tiling_data->matmulTiling.singleBatchN = 1;
    tiling_data->matmulRunInfo.transA = 0;
    tiling_data->matmulRunInfo.transB = 0;
    tiling_data->matmulRunInfo.nd2nzA = 0;
    tiling_data->matmulRunInfo.nd2nzB = 0;
    tiling_data->tileL2cacheTiling.mTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.nTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.mTileBlock = 5;
    tiling_data->tileL2cacheTiling.nTileBlock = 13;
    tiling_data->tileL2cacheTiling.calOrder = 0;
    tiling_data->l2cacheUseInfo.l2CacheFlag = 0;
    
    auto wrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        ::gemm_v3<0, 0, 0, 0, 0>(aGM, bGM, cGM, yGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(wrapper, 24, aGM, bGM, cGM, yGM, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)cGM);
    AscendC::GmFree((void*)yGM);
    free(path_);
}
