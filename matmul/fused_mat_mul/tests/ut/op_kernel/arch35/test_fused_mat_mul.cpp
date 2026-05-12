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

#include "../fused_mat_mul_tiling_def.h"
#include "fused_mat_mul.cpp"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"
using namespace std;

class fused_mat_mul_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "fused_mat_mul_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "fused_mat_mul_test TearDown\n" << endl;
    }
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};

TEST_F(fused_mat_mul_test, fused_mat_mul_test_1) {
    // {{16, 16}, {16, 16}}
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    size_t shape_a = 16 * 16 * sizeof(uint32_t);
    size_t shape_b = 16 * 16 * sizeof(uint32_t);
    size_t shape_output = 16 * 16 * sizeof(uint32_t);

    size_t sysWorkspaceSize = 20 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    size_t allWorkspaceSize = usrWorkspaceSize + sysWorkspaceSize;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(allWorkspaceSize);
    size_t tilingSize = sizeof(MatMulV3BasicTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t *aGM = (uint8_t *)AscendC::GmAlloc(shape_a);
    uint8_t *bGM = (uint8_t *)AscendC::GmAlloc(shape_b);
    uint8_t *biasGM = nullptr;
    uint8_t *x3GM = nullptr;
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(shape_output);

    memset(aGM, 0, shape_a);
    memset(bGM, 0, shape_b);
    memset(output, 0, shape_output);

    system("cp -r ../../../../matmul/fused_mat_mul/tests/ut/op_kernel/fused_mat_mul_data ./");
    system("chmod -R 755 ./fused_mat_mul_data/");
    system("cd ./fused_mat_mul_data/ && rm -rf ./*bin");
    system("cd ./fused_mat_mul_data/ && python3 gen_data.py 16 16 16");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/fused_mat_mul_data/shape_a.bin", shape_a, aGM, shape_a);
    ReadFile(path + "/fused_mat_mul_data/shape_b.bin", shape_b, bGM, shape_b);
    ReadFile(path + "/fused_mat_mul_data/shape_output.bin", shape_output, output, shape_output);

    MatMulV3BasicTilingData *tiling_data = reinterpret_cast<MatMulV3BasicTilingData*>(tiling);
    tiling_data->m = 16;
    tiling_data->n = 16;
    tiling_data->k = 16;
    tiling_data->mL1 = 16;
    tiling_data->nL1 = 16;
    tiling_data->kL1 = 16;
    tiling_data->baseM = 16;
    tiling_data->baseN = 16;
    tiling_data->baseK = 16;
    tiling_data->skSingleCoreK = 0;
    tiling_data->mTailCnt = 0;
    tiling_data->nTailCnt = 0;
    tiling_data->mBaseTailSplitCnt = 1;
    tiling_data->nBaseTailSplitCnt = 1;
    tiling_data->mTailMain = 1;
    tiling_data->nTailMain = 1;
    tiling_data->isHf32 = 0;
    tiling_data->l1BufferNum = 2;
    tiling_data->l0cDB = 2; // 默认不开db为1
    tiling_data->ubDB = 1; // ub默认不开db为1

    auto fused_mat_mul_wrapper = [](GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace,
                                  GM_ADDR tiling) {
        ::fused_mat_mul<0, 0, 0, 0, 0, 0, 0>(x1, x2, bias, x3, y, workspace, tiling);
    };

    ICPU_RUN_KF(fused_mat_mul_wrapper, 20, aGM, bGM, nullptr, x3GM, output, workspace, tiling);

    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)aGM);
    AscendC::GmFree((void*)bGM);
    AscendC::GmFree((void*)output);
    free(path_);
}