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

#include "mat_mul_v3_tiling_def.h"
#include "mat_mul_v3.cpp"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
using namespace std;

class mat_mul_v3_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "mat_mul_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "mat_mul_v3_test TearDown\n" << endl;
    }
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

TEST_F(mat_mul_v3_test, mat_mul_v3_test_1) {
    // {{16, 16}, {16, 16}}
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t shape_a = 16 * 16 * sizeof(uint32_t);
    size_t shape_b = 16 * 16 * sizeof(uint32_t);
    size_t shape_output = 16 * 16 * sizeof(uint32_t);

    size_t sysWorkspaceSize = 20 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatmulTilingData);
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

    system("cp -r ../../../../matmul/mat_mul_v3/tests/ut/op_kernel/mat_mul_v3_data ./");
    system("chmod -R 755 ./mat_mul_v3_data/");
    system("cd ./mat_mul_v3_data/ && rm -rf ./*bin");
    system("cd ./mat_mul_v3_data/ && python3 gen_data_fp32.py 16 16 16");
    // system("cd ./fa_data/ && python3 gen_tiling.py case2");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/mat_mul_v3_data/shape_a.bin", shape_a, aGM, shape_a);
    ReadFile(path + "/mat_mul_v3_data/shape_b.bin", shape_b, bGM, shape_b);
    ReadFile(path + "/mat_mul_v3_data/shape_output.bin", shape_output, output, shape_output);

    MatmulTilingData *tiling_data = reinterpret_cast<MatmulTilingData*>(tiling);
    tiling_data->matmulTiling.usedCoreNum = 20;
    tiling_data->matmulTiling.M = 16;
    tiling_data->matmulTiling.N = 16;
    tiling_data->matmulTiling.Ka = 16;
    tiling_data->matmulTiling.Kb = 16;
    tiling_data->matmulTiling.singleCoreM = 16;
    tiling_data->matmulTiling.singleCoreN = 16;
    tiling_data->matmulTiling.singleCoreK = 16;
    tiling_data->matmulTiling.baseM = 128;
    tiling_data->matmulTiling.baseN = 128;
    tiling_data->matmulTiling.baseK = 64;
    tiling_data->matmulTiling.depthA1 = 8;
    tiling_data->matmulTiling.depthB1 = 8;
    tiling_data->matmulTiling.stepM = 1;
    tiling_data->matmulTiling.stepN = 1;
    tiling_data->matmulTiling.stepKa = 4;
    tiling_data->matmulTiling.stepKb = 4;
    tiling_data->matmulTiling.isBias = 0;
    tiling_data->matmulTiling.transLength = 0;
    tiling_data->matmulTiling.iterateOrder = 0;
    tiling_data->matmulTiling.shareL1Size = 401408;
    tiling_data->matmulTiling.shareL0CSize = 13312;
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
    tiling_data->tileL2cacheTiling.mTileCntL2 = 8;
    tiling_data->tileL2cacheTiling.nTileCntL2 = 1;
    tiling_data->tileL2cacheTiling.mTileBlock = 4;
    tiling_data->tileL2cacheTiling.nTileBlock = 1;
    tiling_data->tileL2cacheTiling.calOrder = 0;

    auto mat_mul_v3_wrapper = [](GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR contextGM, GM_ADDR y, GM_ADDR workspace,
                                 GM_ADDR tiling) {
        ::mat_mul_v3<0, 5, 0, 1>(x, w, bias, contextGM, y, workspace, tiling);
    };

    ICPU_RUN_KF(mat_mul_v3_wrapper, 20, aGM, bGM, nullptr, contextGM, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    free(path_);
}
