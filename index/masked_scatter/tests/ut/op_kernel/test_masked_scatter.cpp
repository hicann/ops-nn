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
#include "../../../op_kernel/masked_scatter.cpp"
#include "../../../op_kernel/masked_scatter_tiling_data.h"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void masked_scatter(
    GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class masked_scatter_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "masked_scatter_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "masked_scatter_test TearDown\n" << endl;
    }
};

TEST_F(masked_scatter_test, test_case_mask)
{
    size_t xByteSize = 32 * 57 * sizeof(float);
    size_t maskByteSize = 32 * 1 * sizeof(bool);
    size_t updatesByteSize = 32 * 57 * sizeof(float);
    size_t yByteSize = 32 * 57 * sizeof(float);
    size_t tiling_data_size = sizeof(MaskedScatterV1TilingData);
    uint32_t numBlocks = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char *path_ = get_current_dir_name();
    string path(path_);

    MaskedScatterV1TilingData* tilingDatafrombin = reinterpret_cast<MaskedScatterV1TilingData*>(tiling);

    tilingDatafrombin->loopNum = 1;
    tilingDatafrombin->remainNum = 0;
    tilingDatafrombin->updatesLineNum = 57;
    tilingDatafrombin->totalUpdatesNum = 1824;
    tilingDatafrombin->maskTileLength = 64;

    auto MaskedScatterKernel = [](GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::masked_scatter<0>(x, mask, updates, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(MaskedScatterKernel,
        numBlocks,
        x,
        mask,
        updates,
        y,
        workspace,
        (uint8_t *)(tilingDatafrombin));
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(updates);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(masked_scatter_test, test_case_updates_1)
{
    size_t xByteSize = 32 * 1 * sizeof(float);
    size_t maskByteSize = 32 * 1 * sizeof(bool);
    size_t updatesByteSize = 32 * 1 * sizeof(float);
    size_t yByteSize = 32 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(MaskedScatterV1TilingData);
    uint32_t numBlocks = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char *path_ = get_current_dir_name();
    string path(path_);

    MaskedScatterV1TilingData* tilingDatafrombin = reinterpret_cast<MaskedScatterV1TilingData*>(tiling);

    tilingDatafrombin->loopNum = 1;
    tilingDatafrombin->remainNum = 0;
    tilingDatafrombin->updatesLineNum = 1;
    tilingDatafrombin->totalUpdatesNum = 32;
    tilingDatafrombin->maskTileLength = 64;

    auto MaskedScatterKernel = [](GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::masked_scatter<1>(x, mask, updates, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(MaskedScatterKernel,
        numBlocks,
        x,
        mask,
        updates,
        y,
        workspace,
        (uint8_t *)(tilingDatafrombin));
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(updates);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(masked_scatter_test, test_case_updates_short)
{
    size_t xByteSize = 32 * 257 * sizeof(float);
    size_t maskByteSize = 32 * 1 * sizeof(bool);
    size_t updatesByteSize = 32 * 257 * sizeof(float);
    size_t yByteSize = 32 * 257 * sizeof(float);
    size_t tiling_data_size = sizeof(MaskedScatterV1TilingData);
    uint32_t numBlocks = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char *path_ = get_current_dir_name();
    string path(path_);

    MaskedScatterV1TilingData* tilingDatafrombin = reinterpret_cast<MaskedScatterV1TilingData*>(tiling);

    tilingDatafrombin->loopNum = 1;
    tilingDatafrombin->remainNum = 0;
    tilingDatafrombin->updatesLineNum = 257;
    tilingDatafrombin->totalUpdatesNum = 8224;
    tilingDatafrombin->maskTileLength = 64;

    auto MaskedScatterKernel = [](GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::masked_scatter<1>(x, mask, updates, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(MaskedScatterKernel,
        numBlocks,
        x,
        mask,
        updates,
        y,
        workspace,
        (uint8_t *)(tilingDatafrombin));
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(updates);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(masked_scatter_test, test_case_updates_exceed)
{
    size_t xByteSize = 32 * 20000 * sizeof(float);
    size_t maskByteSize = 32 * 1 * sizeof(bool);
    size_t updatesByteSize = 32 * 20000 * sizeof(float);
    size_t yByteSize = 32 * 20000 * sizeof(float);
    size_t tiling_data_size = sizeof(MaskedScatterV1TilingData);
    uint32_t numBlocks = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(maskByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char *path_ = get_current_dir_name();
    string path(path_);

    MaskedScatterV1TilingData* tilingDatafrombin = reinterpret_cast<MaskedScatterV1TilingData*>(tiling);

    tilingDatafrombin->loopNum = 1;
    tilingDatafrombin->remainNum = 0;
    tilingDatafrombin->updatesLineNum = 20000;
    tilingDatafrombin->totalUpdatesNum = 640000;
    tilingDatafrombin->maskTileLength = 64;

    auto MaskedScatterKernel = [](GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::masked_scatter<1>(x, mask, updates, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(MaskedScatterKernel,
        numBlocks,
        x,
        mask,
        updates,
        y,
        workspace,
        (uint8_t *)(tilingDatafrombin));
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(updates);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}