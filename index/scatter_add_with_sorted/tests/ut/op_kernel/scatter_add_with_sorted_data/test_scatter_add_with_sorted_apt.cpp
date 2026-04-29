/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

#include "arch35/scatter_add_with_sorted_struct.h"

using namespace std;

extern "C" __global__ __aicore__ void scatter_add_with_sorted(
    GM_ADDR var, GM_ADDR value, GM_ADDR sorted_index, GM_ADDR pos, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class scatter_add_with_sorted_apt_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "scatter_add_with_sorted_apt_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "scatter_add_with_sorted_apt_test TearDown\n" << endl;
    }
};

static void FillSimdTilingData(ScatterAddWithSortedSimdTilingData *td, int64_t indicesNum, int64_t updatesInner,
                               bool withPos, uint64_t tilingKey)
{
    td->tilingKey = tilingKey;
    td->needCoreNum = 1;
    td->indicesNum = indicesNum;
    td->updatesInner = updatesInner;
    td->withPos = withPos;

    td->updatesBufferSize = 4096;
    td->outBufferSize = 4096;
    td->indicesBufferSize = 512;
    td->posBufferSize = withPos ? 512 : 0;
    td->FrontAndBackIndexSize = 0;

    td->coreNumInRow = 1;
    td->coreNumInCol = 1;
    td->normalCoreColNum = updatesInner;
    td->tailCoreColNum = updatesInner;
    td->normalCoreRowNum = indicesNum;
    td->tailCoreRowNum = 0;

    td->normalCoreRowUbLoop = 1;
    td->normalCoreNormalLoopRows = indicesNum;
    td->normalCoreTailLoopRows = 0;
    td->tailCoreRowUbLoop = 1;
    td->tailCoreNormalLoopRows = 1;
    td->tailCoreTailLoopRows = 0;

    td->normalCoreColUbLoop = 1;
    td->normalCoreNormalLoopCols = updatesInner;
    td->normalCoreTailLoopCols = updatesInner;
    td->tailCoreColUbLoop = 1;
    td->tailCoreNormalLoopCols = updatesInner;
    td->tailCoreTailLoopCols = updatesInner;

    td->vecAlignSize = 0;
    td->indicesWorkspaceBufferSize = 0;
    td->coreNumInColDeterm = 0;
    td->tailCoreColUbDetermLoop = 0;
    td->normalCoreColUbDetermLoop = 0;
    td->tailCoreNormalLoopDetermCols = 0;
    td->normalCoreNormalLoopDetermCols = 0;
    td->tailCoreTailLoopDetermCols = 0;
    td->normalCoreTailLoopDetermCols = 0;
    td->updatesDeterminBufferSize = 0;
    td->outBufferDeterminSize = 0;
    td->normalCoreColDetermNum = 0;
    td->tailCoreColNumDeterm = 0;
    td->ubBlock = 32;
}

TEST_F(scatter_add_with_sorted_apt_test, test_simd_float32_with_pos)
{
    int64_t rows = 16;
    int64_t cols = 128;
    size_t var_size = (rows + 2) * cols * sizeof(float);
    size_t src_size = rows * cols * sizeof(float);
    size_t ind_size = rows * sizeof(int32_t);
    size_t pos_size = rows * sizeof(int32_t);
    size_t output_size = (rows + 2) * cols * sizeof(float);
    size_t tiling_data_size = sizeof(ScatterAddWithSortedSimdTilingData);

    uint8_t *var = (uint8_t *)AscendC::GmAlloc(var_size);
    uint8_t *src = (uint8_t *)AscendC::GmAlloc(src_size);
    uint8_t *ind = (uint8_t *)AscendC::GmAlloc(ind_size);
    uint8_t *pos = (uint8_t *)AscendC::GmAlloc(pos_size);
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(output_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    memset(var, 0, var_size);
    memset(src, 1, src_size);
    memset(ind, 0, ind_size);
    memset(pos, 0, pos_size);
    memset(output, 0, output_size);

    ScatterAddWithSortedSimdTilingData *td = reinterpret_cast<ScatterAddWithSortedSimdTilingData *>(tiling);
    FillSimdTilingData(td, rows, cols, true, 0);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_add_with_sorted, blockDim, var, src, ind, pos, output, workspace, (uint8_t *)td);

    AscendC::GmFree(var);
    AscendC::GmFree(src);
    AscendC::GmFree(ind);
    AscendC::GmFree(pos);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_add_with_sorted_apt_test, test_simd_float32_without_pos)
{
    int64_t rows = 16;
    int64_t cols = 128;
    size_t var_size = (rows + 2) * cols * sizeof(float);
    size_t src_size = rows * cols * sizeof(float);
    size_t ind_size = rows * sizeof(int32_t);
    size_t pos_size = rows * sizeof(int32_t);
    size_t output_size = (rows + 2) * cols * sizeof(float);
    size_t tiling_data_size = sizeof(ScatterAddWithSortedSimdTilingData);

    uint8_t *var = (uint8_t *)AscendC::GmAlloc(var_size);
    uint8_t *src = (uint8_t *)AscendC::GmAlloc(src_size);
    uint8_t *ind = (uint8_t *)AscendC::GmAlloc(ind_size);
    uint8_t *pos = (uint8_t *)AscendC::GmAlloc(pos_size);
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(output_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    memset(var, 0, var_size);
    memset(src, 1, src_size);
    memset(ind, 0, ind_size);
    memset(output, 0, output_size);

    ScatterAddWithSortedSimdTilingData *td = reinterpret_cast<ScatterAddWithSortedSimdTilingData *>(tiling);
    FillSimdTilingData(td, rows, cols, false, 0);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_add_with_sorted, blockDim, var, src, ind, pos, output, workspace, (uint8_t *)td);

    AscendC::GmFree(var);
    AscendC::GmFree(src);
    AscendC::GmFree(ind);
    AscendC::GmFree(pos);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(scatter_add_with_sorted_apt_test, test_empty_shape)
{
    int64_t rows = 0;
    int64_t cols = 128;
    size_t var_size = 2 * cols * sizeof(float);
    size_t src_size = 1 * sizeof(float);
    size_t ind_size = 1 * sizeof(int32_t);
    size_t pos_size = 1 * sizeof(int32_t);
    size_t output_size = 2 * cols * sizeof(float);
    size_t tiling_data_size = sizeof(ScatterAddWithSortedSimdTilingData);

    uint8_t *var = (uint8_t *)AscendC::GmAlloc(var_size);
    uint8_t *src = (uint8_t *)AscendC::GmAlloc(src_size);
    uint8_t *ind = (uint8_t *)AscendC::GmAlloc(ind_size);
    uint8_t *pos = (uint8_t *)AscendC::GmAlloc(pos_size);
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(output_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    memset(var, 0, var_size);
    memset(output, 0, output_size);

    ScatterAddWithSortedSimdTilingData *td = reinterpret_cast<ScatterAddWithSortedSimdTilingData *>(tiling);
    FillSimdTilingData(td, rows, cols, false, 2);

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(scatter_add_with_sorted, blockDim, var, src, ind, pos, output, workspace, (uint8_t *)td);

    AscendC::GmFree(var);
    AscendC::GmFree(src);
    AscendC::GmFree(ind);
    AscendC::GmFree(pos);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
