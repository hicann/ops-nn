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
#include "gtest/gtest.h"

#include <unistd.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"

#include "gemm_v2_tiling_def.h"
#include "../../../op_kernel/gemm_v2.cpp"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
using namespace std;

extern "C" __global__ __aicore__ void gemm_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR alpha, GM_ADDR beta, GM_ADDR ref_c,
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);

class gemm_v2_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "gemm_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "gemm_v2_test TearDown\n" << endl;
    }
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

TEST_F(gemm_v2_test, gemm_v2_test_1) {
    // {{16, 16}, {16, 16}}
    AscendC::SetKernelMode(KernelMode::AIC_MODE);

    size_t shape_a = 16 * 16 * sizeof(uint16_t);
    size_t shape_b = 16 * 16 * sizeof(uint16_t);
    size_t shape_alpha = 1 * sizeof(uint16_t);
    size_t shape_beta = 1 * sizeof(uint16_t);
    size_t shape_c = 16 * 16 * sizeof(uint32_t);
    size_t shape_output = 16 * 16 * sizeof(uint32_t);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(shape_a);
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(shape_b);
    uint8_t *ref_c = (uint8_t *)AscendC::GmAlloc(shape_c);
    uint8_t *alphaGM = (uint8_t *)AscendC::GmAlloc(shape_alpha);
    uint8_t *betaGM = (uint8_t *)AscendC::GmAlloc(shape_alpha);

    uint8_t *biasGM = nullptr;
    uint8_t *offsetWGM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(shape_output);
    uint8_t *contextGM = (uint8_t *)AscendC::GmAlloc(sizeof(HcclCombinOpParam));

    memset(aGM, 0, shape_a);
    memset(bGM, 0, shape_b);
    memset(ref_c, 0, shape_c);
    memset(output, 0, shape_output);

    system("cp -r ../../../../matmul/gemm_v2/tests/ut/op_kernel/gemm_v2_data ./");
    system("chmod -R 755 ./gemm_v2_data/");
    system("cd ./gemm_v2_data/ && rm -rf ./*bin");
    system("cd ./gemm_v2_data/ && python3 gen_data.py 16 16 16");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/gemm_v2_data/shape_a.bin", shape_a, aGM, shape_a);
    ReadFile(path + "/gemm_v2_data/shape_b.bin", shape_b, bGM, shape_b);
    ReadFile(path + "/gemm_v2_data/shape_c.bin", shape_c, ref_c, shape_c);
    ReadFile(path + "/gemm_v2_data/shape_output.bin", shape_output, output, shape_output);
    *alphaGM = 1;
    *betaGM = 1;

    MatmulTilingData *tiling_data = reinterpret_cast<MatmulTilingData*>(tiling);
    tiling_data->matmulTiling.usedCoreNum = 1;
    tiling_data->matmulTiling.M = 16;
    tiling_data->matmulTiling.N = 16;
    tiling_data->matmulTiling.Ka = 16;
    tiling_data->matmulTiling.Kb = 16;
    tiling_data->matmulTiling.singleCoreM = 16;
    tiling_data->matmulTiling.singleCoreN = 16;
    tiling_data->matmulTiling.singleCoreK = 16;
    tiling_data->matmulTiling.baseM = 16;
    tiling_data->matmulTiling.baseN = 16;
    tiling_data->matmulTiling.baseK = 16;
    tiling_data->matmulTiling.depthA1 = 1;
    tiling_data->matmulTiling.depthB1 = 1;
    tiling_data->matmulTiling.stepM = 1;
    tiling_data->matmulTiling.stepN = 1;
    tiling_data->matmulTiling.stepKa = 1;
    tiling_data->matmulTiling.stepKb = 1;
    tiling_data->matmulTiling.isBias = 0;
    tiling_data->matmulTiling.transLength = 0;
    tiling_data->matmulTiling.iterateOrder = 0;
    tiling_data->matmulTiling.shareL1Size = 1024;
    tiling_data->matmulTiling.shareL0CSize = 1024;
    tiling_data->matmulTiling.shareUbSize = 0;
    tiling_data->matmulTiling.batchM = 1;
    tiling_data->matmulTiling.batchN = 1;
    tiling_data->matmulTiling.singleBatchM = 1;
    tiling_data->matmulTiling.singleBatchN = 1;
    tiling_data->matmulTiling.depthAL1CacheUB = 0;
    tiling_data->matmulTiling.depthBL1CacheUB = 0;
    tiling_data->matmulTiling.dbL0A = 2;
    tiling_data->matmulTiling.dbL0B = 2;
    tiling_data->matmulTiling.dbL0C = 2;
    tiling_data->matmulRunInfo.transA = 0;
    tiling_data->matmulRunInfo.transB = 1;
    tiling_data->matmulRunInfo.nd2nzA = 0;
    tiling_data->matmulRunInfo.nd2nzB = 0;
    tiling_data->matmulRunInfo.isHf32 = 0;
    tiling_data->matmulRunInfo.isNzA = 0;
    tiling_data->matmulRunInfo.isNzB = 0;
    tiling_data->l2cacheUseInfo.l2CacheFlag = 0;
    tiling_data->tileL2cacheTiling.mTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.nTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.mTileBlock = 1;
    tiling_data->tileL2cacheTiling.nTileBlock = 1;
    tiling_data->tileL2cacheTiling.calOrder = 0;

    auto wrapper = [](GM_ADDR aGM, GM_ADDR bGM, GM_ADDR alpha, GM_ADDR beta, GM_ADDR ref_c,
                        GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
        ::gemm_v2<0, 0, 0, 1>(aGM, bGM, alpha, beta, ref_c, cGM, workspaceGM, tilingGM);
    };
    ICPU_RUN_KF(wrapper, 1, aGM, bGM, alphaGM, betaGM, ref_c, output, workspace, tiling);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    free(path_);
}

