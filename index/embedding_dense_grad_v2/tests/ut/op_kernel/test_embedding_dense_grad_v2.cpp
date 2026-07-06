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
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;
extern "C" __global__ __aicore__ void embedding_dense_grad_v2(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx,
                                                              GM_ADDR backProps, GM_ADDR workSpace, GM_ADDR tiling);

class embedding_dense_grad_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "embedding_dense_grad_v2_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "embedding_dense_grad_v2_test TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./data");
    }
};

TEST_F(embedding_dense_grad_v2_test, test_case_1024_4096_100_false_false)
{
    size_t gradSize = 1024 * 4096 * sizeof(float);
    size_t sortIndiceSize = 1024 * sizeof(int32_t);
    size_t posIdxSize = 1024 * sizeof(int32_t);
    size_t backPropsSize = 100 * 4096 * sizeof(float);
    size_t tilingSize = sizeof(EmbeddingDenseGradV2TilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradSize);
    uint8_t* sortIndice = (uint8_t*)AscendC::GmAlloc(sortIndiceSize);
    uint8_t* posIdx = (uint8_t*)AscendC::GmAlloc(posIdxSize);
    uint8_t* backProps = (uint8_t*)AscendC::GmAlloc(backPropsSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);

    kernel_ut::SetupTestEnvironment("index/embedding_dense_grad_v2/tests/ut/op_kernel/data", "data");
    kernel_ut::RunGenData("./data", {"1024", "4096", "100", "False"});
    kernel_ut::RunGenTiling("./data", {"test_case_1024_4096_100_false_false"});

    string path = kernel_ut::GetTestWorkDir();
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
}

TEST_F(embedding_dense_grad_v2_test, test_case_1024_4096_100_false_true)
{
    size_t gradSize = 1024 * 4096 * sizeof(float);
    size_t sortIndiceSize = 1024 * sizeof(int32_t);
    size_t posIdxSize = 1024 * sizeof(int32_t);
    size_t backPropsSize = 100 * 4096 * sizeof(float);
    size_t tilingSize = sizeof(EmbeddingDenseGradV2TilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradSize);
    uint8_t* sortIndice = (uint8_t*)AscendC::GmAlloc(sortIndiceSize);
    uint8_t* posIdx = (uint8_t*)AscendC::GmAlloc(posIdxSize);
    uint8_t* backProps = (uint8_t*)AscendC::GmAlloc(backPropsSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);

    kernel_ut::SetupTestEnvironment("index/embedding_dense_grad_v2/tests/ut/op_kernel/data", "data");
    kernel_ut::RunGenData("./data", {"1024", "4096", "100", "False"});
    kernel_ut::RunGenTiling("./data", {"test_case_1024_4096_100_false_true"});

    string path = kernel_ut::GetTestWorkDir();
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
}

TEST_F(embedding_dense_grad_v2_test, test_case_1024_4096_100_true_false)
{
    size_t gradSize = 1024 * 4096 * sizeof(float);
    size_t sortIndiceSize = 1024 * sizeof(int32_t);
    size_t posIdxSize = 1024 * sizeof(int32_t);
    size_t backPropsSize = 100 * 4096 * sizeof(float);
    size_t tilingSize = sizeof(EmbeddingDenseGradV2TilingData);
    size_t workSpaceSize = 16 * 1024 * 1024 + 100 * sizeof(int32_t);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradSize);
    uint8_t* sortIndice = (uint8_t*)AscendC::GmAlloc(sortIndiceSize);
    uint8_t* posIdx = (uint8_t*)AscendC::GmAlloc(posIdxSize);
    uint8_t* backProps = (uint8_t*)AscendC::GmAlloc(backPropsSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workSpaceSize);

    memset(workspace, 0, workSpaceSize);

    kernel_ut::SetupTestEnvironment("index/embedding_dense_grad_v2/tests/ut/op_kernel/data", "data");
    kernel_ut::RunGenData("./data", {"1024", "4096", "100", "False"});
    kernel_ut::RunGenTiling("./data", {"test_case_1024_4096_100_true_false"});

    string path = kernel_ut::GetTestWorkDir();
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
}

TEST_F(embedding_dense_grad_v2_test, test_case_1024_4096_100_true_true)
{
    size_t gradSize = 1024 * 4096 * sizeof(float);
    size_t sortIndiceSize = 1024 * sizeof(int32_t);
    size_t posIdxSize = 1024 * sizeof(int32_t);
    size_t backPropsSize = 100 * 4096 * sizeof(float);
    size_t tilingSize = sizeof(EmbeddingDenseGradV2TilingData);
    size_t workSpaceSize = 16 * 1024 * 1024 + 100 * sizeof(int32_t);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradSize);
    uint8_t* sortIndice = (uint8_t*)AscendC::GmAlloc(sortIndiceSize);
    uint8_t* posIdx = (uint8_t*)AscendC::GmAlloc(posIdxSize);
    uint8_t* backProps = (uint8_t*)AscendC::GmAlloc(backPropsSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workSpaceSize);

    memset(workspace, 0, workSpaceSize);

    kernel_ut::SetupTestEnvironment("index/embedding_dense_grad_v2/tests/ut/op_kernel/data", "data");
    kernel_ut::RunGenData("./data", {"1024", "4096", "100", "False"});
    kernel_ut::RunGenTiling("./data", {"test_case_1024_4096_100_true_true"});

    string path = kernel_ut::GetTestWorkDir();
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
}

TEST_F(embedding_dense_grad_v2_test, test_case_1024_256_100_false_false)
{
    size_t gradSize = 48 * 256 * sizeof(float);
    size_t sortIndiceSize = 48 * sizeof(int32_t);
    size_t posIdxSize = 48 * sizeof(int32_t);
    size_t backPropsSize = 100 * 256 * sizeof(float);
    size_t tilingSize = sizeof(EmbeddingDenseGradV2TilingData);
    size_t workSpaceSize = 16 * 1024 * 1024 + 100 * sizeof(int32_t) + 100 * 256 * sizeof(float);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradSize);
    uint8_t* sortIndice = (uint8_t*)AscendC::GmAlloc(sortIndiceSize);
    uint8_t* posIdx = (uint8_t*)AscendC::GmAlloc(posIdxSize);
    uint8_t* backProps = (uint8_t*)AscendC::GmAlloc(backPropsSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workSpaceSize);

    memset(workspace, 0, workSpaceSize);

    kernel_ut::SetupTestEnvironment("index/embedding_dense_grad_v2/tests/ut/op_kernel/data", "data");
    kernel_ut::RunGenData("./data", {"48", "256", "100", "False"});

    EmbeddingDenseGradV2TilingData* tilingData = reinterpret_cast<EmbeddingDenseGradV2TilingData*>(tiling);
    tilingData->params.coreNum = 48;
    tilingData->params.numWeights = 100;
    tilingData->params.embeddingDim = 256;
    tilingData->params.paddingIdx = -2;
    tilingData->params.scaleGradByFreq = 0;
    tilingData->params.scaleWorkspaceLength = 0;
    tilingData->params.outCastedWorkspaceLength = 25600;
    tilingData->smallDimTiling.maxRowInUb = 81;
    tilingData->smallDimTiling.partNum = 1;
    tilingData->smallDimTiling.formerCopyRow = 1;
    tilingData->smallDimTiling.formerCopyTime = 1;
    tilingData->smallDimTiling.formerLastRow = 1;
    tilingData->smallDimTiling.tailCopyRow = 1;
    tilingData->smallDimTiling.tailCopyTime = 1;
    tilingData->smallDimTiling.tailLastRow = 1;

    string path = kernel_ut::GetTestWorkDir();
    ReadFile(path + "/data/grad.bin", gradSize, grad, gradSize);
    ReadFile(path + "/data/sort_indices.bin", sortIndiceSize, sortIndice, sortIndiceSize);
    ReadFile(path + "/data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);

    ICPU_SET_TILING_KEY(100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(embedding_dense_grad_v2, 48, grad, sortIndice, posIdx, backProps, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(sortIndice);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(backProps);
    AscendC::GmFree(tiling);
}
