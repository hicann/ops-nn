/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_dynamic_block_quant.cpp
 * \brief
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

using namespace std;

extern "C" __global__ __aicore__ void grouped_dynamic_block_quant(
    uint8_t* x, uint8_t* groupList, uint8_t* y, uint8_t* scale, uint8_t* workspace, uint8_t* tiling);

class grouped_dynamic_block_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "grouped_dynamic_block_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "grouped_dynamic_block_quant_test TearDown\n" << endl;
    }
};

TEST_F(grouped_dynamic_block_quant_test, test_case_half_001)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1111;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 36;
    tilingDatafromBin->rowBlockSize = 128;
    tilingDatafromBin->colBlockSize = 128;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    int64_t maxUbAvailableRows = tilingDatafromBin->rowBlockSize * (tilingDatafromBin->ubSize / 2) / totalElementSize;
    tilingDatafromBin->maxUbRow =
        maxUbAvailableRows / tilingDatafromBin->rowBlockSize * tilingDatafromBin->rowBlockSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);

    ICPU_SET_TILING_KEY(1111);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_half_002)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1121;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->rowBlockSize = 128;
    tilingDatafromBin->colBlockSize = 128;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    int64_t maxUbAvailableRows = tilingDatafromBin->rowBlockSize * (tilingDatafromBin->ubSize / 2) / totalElementSize;
    tilingDatafromBin->maxUbRow =
        maxUbAvailableRows / tilingDatafromBin->rowBlockSize * tilingDatafromBin->rowBlockSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);

    ICPU_SET_TILING_KEY(1121);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_half_003)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1134;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->rowBlockSize = 128;
    tilingDatafromBin->colBlockSize = 128;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    int64_t maxUbAvailableRows = tilingDatafromBin->rowBlockSize * (tilingDatafromBin->ubSize / 2) / totalElementSize;
    tilingDatafromBin->maxUbRow =
        maxUbAvailableRows / tilingDatafromBin->rowBlockSize * tilingDatafromBin->rowBlockSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);

    ICPU_SET_TILING_KEY(1134);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_half_004)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1137;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 7;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->rowBlockSize = 128;
    tilingDatafromBin->colBlockSize = 128;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    int64_t maxUbAvailableRows = tilingDatafromBin->rowBlockSize * (tilingDatafromBin->ubSize / 2) / totalElementSize;
    tilingDatafromBin->maxUbRow =
        maxUbAvailableRows / tilingDatafromBin->rowBlockSize * tilingDatafromBin->rowBlockSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);
    ICPU_SET_TILING_KEY(1137);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_half_005)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 2111;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 36;
    tilingDatafromBin->rowBlockSize = 512;
    tilingDatafromBin->colBlockSize = 256;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    tilingDatafromBin->maxUbRow = tilingDatafromBin->rowBlockSize *
                                  ((tilingDatafromBin->ubSize - tilingDatafromBin->vfLen) / 2) / totalElementSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);
    ICPU_SET_TILING_KEY(2111);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_half_006)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 2121;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->rowBlockSize = 512;
    tilingDatafromBin->colBlockSize = 256;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    tilingDatafromBin->maxUbRow = tilingDatafromBin->rowBlockSize *
                                  ((tilingDatafromBin->ubSize - tilingDatafromBin->vfLen) / 2) / totalElementSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);
    ICPU_SET_TILING_KEY(2121);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_half_007)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 2134;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->rowBlockSize = 512;
    tilingDatafromBin->colBlockSize = 256;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    tilingDatafromBin->maxUbRow = tilingDatafromBin->rowBlockSize *
                                  ((tilingDatafromBin->ubSize - tilingDatafromBin->vfLen) / 2) / totalElementSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);
    ICPU_SET_TILING_KEY(2134);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_half_008)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 2137;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 7;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->rowBlockSize = 512;
    tilingDatafromBin->colBlockSize = 256;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    tilingDatafromBin->maxUbRow = tilingDatafromBin->rowBlockSize *
                                  ((tilingDatafromBin->ubSize - tilingDatafromBin->vfLen) / 2) / totalElementSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);
    ICPU_SET_TILING_KEY(2137);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_bfloat16_001)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1211;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 36;
    tilingDatafromBin->rowBlockSize = 128;
    tilingDatafromBin->colBlockSize = 128;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    int64_t maxUbAvailableRows = tilingDatafromBin->rowBlockSize * (tilingDatafromBin->ubSize / 2) / totalElementSize;
    tilingDatafromBin->maxUbRow =
        maxUbAvailableRows / tilingDatafromBin->rowBlockSize * tilingDatafromBin->rowBlockSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);

    ICPU_SET_TILING_KEY(1211);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_bfloat16_002)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1221;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->rowBlockSize = 128;
    tilingDatafromBin->colBlockSize = 128;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    int64_t maxUbAvailableRows = tilingDatafromBin->rowBlockSize * (tilingDatafromBin->ubSize / 2) / totalElementSize;
    tilingDatafromBin->maxUbRow =
        maxUbAvailableRows / tilingDatafromBin->rowBlockSize * tilingDatafromBin->rowBlockSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);

    ICPU_SET_TILING_KEY(1221);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_bfloat16_003)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1234;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->rowBlockSize = 128;
    tilingDatafromBin->colBlockSize = 128;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    int64_t maxUbAvailableRows = tilingDatafromBin->rowBlockSize * (tilingDatafromBin->ubSize / 2) / totalElementSize;
    tilingDatafromBin->maxUbRow =
        maxUbAvailableRows / tilingDatafromBin->rowBlockSize * tilingDatafromBin->rowBlockSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);

    ICPU_SET_TILING_KEY(1234);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_bfloat16_004)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 1237;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 7;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->rowBlockSize = 128;
    tilingDatafromBin->colBlockSize = 128;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    int64_t maxUbAvailableRows = tilingDatafromBin->rowBlockSize * (tilingDatafromBin->ubSize / 2) / totalElementSize;
    tilingDatafromBin->maxUbRow =
        maxUbAvailableRows / tilingDatafromBin->rowBlockSize * tilingDatafromBin->rowBlockSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);

    ICPU_SET_TILING_KEY(1237);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_bfloat16_005)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 2211;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 36;
    tilingDatafromBin->rowBlockSize = 512;
    tilingDatafromBin->colBlockSize = 256;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    tilingDatafromBin->maxUbRow = tilingDatafromBin->rowBlockSize *
                                  ((tilingDatafromBin->ubSize - tilingDatafromBin->vfLen) / 2) / totalElementSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);
    ICPU_SET_TILING_KEY(2211);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_bfloat16_006)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 2221;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->rowBlockSize = 512;
    tilingDatafromBin->colBlockSize = 256;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    tilingDatafromBin->maxUbRow = tilingDatafromBin->rowBlockSize *
                                  ((tilingDatafromBin->ubSize - tilingDatafromBin->vfLen) / 2) / totalElementSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);
    ICPU_SET_TILING_KEY(2221);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_bfloat16_007)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 2234;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->rowBlockSize = 512;
    tilingDatafromBin->colBlockSize = 256;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    tilingDatafromBin->maxUbRow = tilingDatafromBin->rowBlockSize *
                                  ((tilingDatafromBin->ubSize - tilingDatafromBin->vfLen) / 2) / totalElementSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);
    ICPU_SET_TILING_KEY(2234);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

TEST_F(grouped_dynamic_block_quant_test, test_case_bfloat16_008)
{
    size_t xByteSize = 128 * 128 * sizeof(half);
    size_t groupListSize = 1 * sizeof(int32_t);
    size_t yByteSize = 128 * 128 * sizeof(uint8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(GroupedDynamicBlockQuantTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* groupList = (uint8_t*)AscendC::GmAlloc(groupListSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    std::string path_ = get_current_dir_name();
    string path(path_);

    GroupedDynamicBlockQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 2237;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->ubSize = 253952;
    tilingDatafromBin->vfLen = 256;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->headCoreNum = 1;
    tilingDatafromBin->tailCoreNum = 1;
    tilingDatafromBin->minScale = 0.0;
    tilingDatafromBin->roundMode = 7;
    tilingDatafromBin->dstType = 34;
    tilingDatafromBin->rowBlockSize = 512;
    tilingDatafromBin->colBlockSize = 256;
    tilingDatafromBin->batchNum = 1;
    tilingDatafromBin->rowNum = 128;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->scaleRowNum = 2;
    tilingDatafromBin->scaleColNum = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->tailBlockFactor = 128;
    tilingDatafromBin->groupBlockNumHeadCore = 1;
    tilingDatafromBin->groupBlockNumTailCore = 0;

    int64_t totalElementSize = 32 + tilingDatafromBin->rowBlockSize * tilingDatafromBin->colBlockSize * 3;
    tilingDatafromBin->maxUbRow = tilingDatafromBin->rowBlockSize *
                                  ((tilingDatafromBin->ubSize - tilingDatafromBin->vfLen) / 2) / totalElementSize;

    system(
        "cp -rf "
        "../../../../quant/grouped_dynamic_block_quant/tests/ut/op_kernel/grouped_dynamic_block_quant_data ./");
    system("chmod -R 755 ./grouped_dynamic_block_quant_data/");
    system("cd ./grouped_dynamic_block_quant_data/ && python3 gen_data.py '(128, 128)' '1' 'float16'");

    ReadFile("./grouped_dynamic_block_quant_data/input_size.bin", xByteSize, x, xByteSize);
    ICPU_SET_TILING_KEY(2237);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(grouped_dynamic_block_quant, blockDim, x, groupList, y, scale, workspace, tiling);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupList);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}