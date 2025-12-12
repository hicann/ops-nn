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
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void embedding_dense_grad_v2(GM_ADDR grad, GM_ADDR sortIndices,
    GM_ADDR posIdx, GM_ADDR backProps, GM_ADDR workSpace, GM_ADDR tiling);

class embedding_dense_grad_v2_test : public testing::Test {
    protected:

    static void SetUpTestCase() {
        cout << "embedding_dense_grad_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "embedding_dense_grad_v2_test TearDown\n" << endl;
    }
};

TEST_F(embedding_dense_grad_v2_test, test_case_1024_4096_100_false_false)
{
    size_t gradSize = 1024 * 4096 * sizeof(float);
    size_t sortIndiceSize = 1024 * sizeof(int32_t);
    size_t posIdxSize = 1024 * sizeof(int32_t);
    size_t backPropsSize = 100 * 4096 * sizeof(float);
    size_t tilingSize = sizeof(EmbeddingDenseGradV2TilingData);

    uint8_t *grad = (uint8_t *)AscendC::GmAlloc(gradSize);
    uint8_t *sortIndice = (uint8_t *)AscendC::GmAlloc(sortIndiceSize);
    uint8_t *posIdx = (uint8_t *)AscendC::GmAlloc(posIdxSize);
    uint8_t *backProps = (uint8_t *)AscendC::GmAlloc(backPropsSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../index/embedding_dense_grad_v2/tests/ut/op_kernel/data ./");
    system("chmod -R 755 ./data/");
    system("cd ./data/ && rm -rf ./*bin");
    system("cd ./data/ && python3 gen_data.py 1024 4096 100 False");
    system("cd ./data/ && python3 gen_tiling.py test_case_1024_4096_100_false_false");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/data/grad.bin", gradSize, grad, gradSize);
    ReadFile(path + "/data/sort_indices.bin", sortIndiceSize, sortIndice, sortIndiceSize);
    ReadFile(path + "/data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);
    ReadFile(path + "/data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(embedding_dense_grad_v2, 48, grad, sortIndice, posIdx, backProps, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(sortIndice);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(backProps);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(embedding_dense_grad_v2_test, test_case_1024_4096_100_false_true)
{
    size_t gradSize = 1024 * 4096 * sizeof(float);
    size_t sortIndiceSize = 1024 * sizeof(int32_t);
    size_t posIdxSize = 1024 * sizeof(int32_t);
    size_t backPropsSize = 100 * 4096 * sizeof(float);
    size_t tilingSize = sizeof(EmbeddingDenseGradV2TilingData);

    uint8_t *grad = (uint8_t *)AscendC::GmAlloc(gradSize);
    uint8_t *sortIndice = (uint8_t *)AscendC::GmAlloc(sortIndiceSize);
    uint8_t *posIdx = (uint8_t *)AscendC::GmAlloc(posIdxSize);
    uint8_t *backProps = (uint8_t *)AscendC::GmAlloc(backPropsSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../index/embedding_dense_grad_v2/tests/ut/op_kernel/data ./");
    system("chmod -R 755 ./data/");
    system("cd ./data/ && rm -rf ./*bin");
    system("cd ./data/ && python3 gen_data.py 1024 4096 100 False");
    system("cd ./data/ && python3 gen_tiling.py test_case_1024_4096_100_false_true");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/data/grad.bin", gradSize, grad, gradSize);
    ReadFile(path + "/data/sort_indices.bin", sortIndiceSize, sortIndice, sortIndiceSize);
    ReadFile(path + "/data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);
    ReadFile(path + "/data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(10);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(embedding_dense_grad_v2, 48, grad, sortIndice, posIdx, backProps, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(sortIndice);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(backProps);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(embedding_dense_grad_v2_test, test_case_1024_4096_100_true_false)
{
    size_t gradSize = 1024 * 4096 * sizeof(float);
    size_t sortIndiceSize = 1024 * sizeof(int32_t);
    size_t posIdxSize = 1024 * sizeof(int32_t);
    size_t backPropsSize = 100 * 4096 * sizeof(float);
    size_t tilingSize = sizeof(EmbeddingDenseGradV2TilingData);
    size_t workSpaceSize = 16 * 1024 * 1024 + 100 * sizeof(int32_t);

    uint8_t *grad = (uint8_t *)AscendC::GmAlloc(gradSize);
    uint8_t *sortIndice = (uint8_t *)AscendC::GmAlloc(sortIndiceSize);
    uint8_t *posIdx = (uint8_t *)AscendC::GmAlloc(posIdxSize);
    uint8_t *backProps = (uint8_t *)AscendC::GmAlloc(backPropsSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workSpaceSize);

    memset(workspace, 0, workSpaceSize);

    system("cp -r ../../../../index/embedding_dense_grad_v2/tests/ut/op_kernel/data ./");
    system("chmod -R 755 ./data/");
    system("cd ./data/ && rm -rf ./*bin");
    system("cd ./data/ && python3 gen_data.py 1024 4096 100 False");
    system("cd ./data/ && python3 gen_tiling.py test_case_1024_4096_100_true_false");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/data/grad.bin", gradSize, grad, gradSize);
    ReadFile(path + "/data/sort_indices.bin", sortIndiceSize, sortIndice, sortIndiceSize);
    ReadFile(path + "/data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);
    ReadFile(path + "/data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(embedding_dense_grad_v2, 1, grad, sortIndice, posIdx, backProps, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(sortIndice);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(backProps);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(embedding_dense_grad_v2_test, test_case_1024_4096_100_true_true)
{
    size_t gradSize = 1024 * 4096 * sizeof(float);
    size_t sortIndiceSize = 1024 * sizeof(int32_t);
    size_t posIdxSize = 1024 * sizeof(int32_t);
    size_t backPropsSize = 100 * 4096 * sizeof(float);
    size_t tilingSize = sizeof(EmbeddingDenseGradV2TilingData);
    size_t workSpaceSize = 16 * 1024 * 1024 + 100 * sizeof(int32_t);

    uint8_t *grad = (uint8_t *)AscendC::GmAlloc(gradSize);
    uint8_t *sortIndice = (uint8_t *)AscendC::GmAlloc(sortIndiceSize);
    uint8_t *posIdx = (uint8_t *)AscendC::GmAlloc(posIdxSize);
    uint8_t *backProps = (uint8_t *)AscendC::GmAlloc(backPropsSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workSpaceSize);

    memset(workspace, 0, workSpaceSize);

    system("cp -r ../../../../index/embedding_dense_grad_v2/tests/ut/op_kernel/data ./");
    system("chmod -R 755 ./data/");
    system("cd ./data/ && rm -rf ./*bin");
    system("cd ./data/ && python3 gen_data.py 1024 4096 100 False");
    system("cd ./data/ && python3 gen_tiling.py test_case_1024_4096_100_true_true");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/data/grad.bin", gradSize, grad, gradSize);
    ReadFile(path + "/data/sort_indices.bin", sortIndiceSize, sortIndice, sortIndiceSize);
    ReadFile(path + "/data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);
    ReadFile(path + "/data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(11);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(embedding_dense_grad_v2, 1, grad, sortIndice, posIdx, backProps, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(sortIndice);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(backProps);
    AscendC::GmFree(tiling);
    free(path_);
}