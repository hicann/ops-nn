/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_index_put_with_sort.cpp
 * \brief
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

extern "C" __global__ __aicore__ void index_put_with_sort(GM_ADDR self, GM_ADDR linearIndex, 
    GM_ADDR posIdx, GM_ADDR values, GM_ADDR self_ref, GM_ADDR workSpace, GM_ADDR tiling);

class index_put_with_sort_test : public testing::Test {
    protected:

    static void SetUpTestCase() {
        cout << "index_put_with_sort_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "index_put_with_sort_test TearDown\n" << endl;
    }
};

TEST_F(index_put_with_sort_test, test_case_scatter_between_core_float_false) 
{
    size_t selfSize = 48 * 1536 * sizeof(float); // 1536 = 48 * 32
    size_t linearIndexSize = 48 * sizeof(int32_t);
    size_t posIdxSize = 48 * sizeof(int32_t);
    size_t valuesSize = 48 * 1536 * sizeof(float);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    int64_t workspaceSize = sysWorkspaceSize + usrWorkspaceSize;

    size_t tilingSize = sizeof(IndexPutWithSortTilingData);

    uint8_t *self = (uint8_t *)AscendC::GmAlloc(selfSize);
    uint8_t *linearIndex = (uint8_t *)AscendC::GmAlloc(linearIndexSize);
    uint8_t *posIdx = (uint8_t *)AscendC::GmAlloc(posIdxSize);
    uint8_t *values = (uint8_t *)AscendC::GmAlloc(valuesSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    system("cp -r ../../../../index/index_put_with_sort/tests/ut/op_kernel/index_put_with_sort_data ./");
    system("chmod -R 755 ./index_put_with_sort_data/");
    system("cd ./index_put_with_sort_data/ && rm -rf ./*bin");
    system("cd ./index_put_with_sort_data/ && python3 gen_data.py 48 48 1536");
    
    IndexPutWithSortTilingData *tilingData = reinterpret_cast<IndexPutWithSortTilingData*>(tiling);
    tilingData->coreNums = 48;
    tilingData->indicesNums = 48;
    tilingData->sliceSize = 1536;
    tilingData->frontCoreNums = 48;
    tilingData->frontCoreDataLength = 32;
    tilingData->tailCoreDataLength = 0;
    tilingData->accumulate = 0;

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/index_put_with_sort_data/self.bin", selfSize, self, selfSize);
    ReadFile(path + "/index_put_with_sort_data/linear_index.bin", linearIndexSize, linearIndex, linearIndexSize);
    ReadFile(path + "/index_put_with_sort_data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);
    ReadFile(path + "/index_put_with_sort_data/values.bin", valuesSize, values, valuesSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(index_put_with_sort, 1, self, linearIndex, posIdx, values, self, workspace, tiling);

    AscendC::GmFree(self);
    AscendC::GmFree(linearIndex);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(values);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
    free(path_);
}

TEST_F(index_put_with_sort_test, test_case_scatter_between_core_float_true) 
{
    size_t selfSize = 48 * 1536 * sizeof(float); // 1536 = 48 * 32
    size_t linearIndexSize = 48 * sizeof(int32_t);
    size_t posIdxSize = 48 * sizeof(int32_t);
    size_t valuesSize = 48 * 1536 * sizeof(float);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = 0;
    int64_t workspaceSize = sysWorkspaceSize + usrWorkspaceSize;

    size_t tilingSize = sizeof(IndexPutWithSortTilingData);

    uint8_t *self = (uint8_t *)AscendC::GmAlloc(selfSize);
    uint8_t *linearIndex = (uint8_t *)AscendC::GmAlloc(linearIndexSize);
    uint8_t *posIdx = (uint8_t *)AscendC::GmAlloc(posIdxSize);
    uint8_t *values = (uint8_t *)AscendC::GmAlloc(valuesSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    system("cp -r ../../../../index/index_put_with_sort/tests/ut/op_kernel/index_put_with_sort_data ./");
    system("chmod -R 755 ./index_put_with_sort_data/");
    system("cd ./index_put_with_sort_data/ && rm -rf ./*bin");
    system("cd ./index_put_with_sort_data/ && python3 gen_data.py 48 48 1536");
    
    IndexPutWithSortTilingData *tilingData = reinterpret_cast<IndexPutWithSortTilingData*>(tiling);
    tilingData->coreNums = 48;
    tilingData->indicesNums = 48;
    tilingData->sliceSize = 1536;
    tilingData->frontCoreNums = 48;
    tilingData->frontCoreDataLength = 32;
    tilingData->tailCoreDataLength = 0;
    tilingData->accumulate = 1;

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/index_put_with_sort_data/self.bin", selfSize, self, selfSize);
    ReadFile(path + "/index_put_with_sort_data/linear_index.bin", linearIndexSize, linearIndex, linearIndexSize);
    ReadFile(path + "/index_put_with_sort_data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);
    ReadFile(path + "/index_put_with_sort_data/values.bin", valuesSize, values, valuesSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(index_put_with_sort, 1, self, linearIndex, posIdx, values, self, workspace, tiling);

    AscendC::GmFree(self);
    AscendC::GmFree(linearIndex);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(values);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
    free(path_);
}

TEST_F(index_put_with_sort_test, test_case_gather_float_false) 
{
    size_t selfSize = 96 * 1536 * sizeof(float); // 1536 = 48 * 32
    size_t linearIndexSize = 96 * sizeof(int32_t);
    size_t posIdxSize = 96 * sizeof(int32_t);
    size_t valuesSize = 96 * 1536 * sizeof(float);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = sysWorkspaceSize;
    int64_t workspaceSize = sysWorkspaceSize + usrWorkspaceSize;

    size_t tilingSize = sizeof(IndexPutWithSortTilingData);

    uint8_t *self = (uint8_t *)AscendC::GmAlloc(selfSize);
    uint8_t *linearIndex = (uint8_t *)AscendC::GmAlloc(linearIndexSize);
    uint8_t *posIdx = (uint8_t *)AscendC::GmAlloc(posIdxSize);
    uint8_t *values = (uint8_t *)AscendC::GmAlloc(valuesSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    system("cp -r ../../../../index/index_put_with_sort/tests/ut/op_kernel/index_put_with_sort_data ./");
    system("chmod -R 755 ./index_put_with_sort_data/");
    system("cd ./index_put_with_sort_data/ && rm -rf ./*bin");
    system("cd ./index_put_with_sort_data/ && python3 gen_data.py 48 48 1536");
    
    IndexPutWithSortTilingData *tilingData = reinterpret_cast<IndexPutWithSortTilingData*>(tiling);
    tilingData->coreNums = 48;
    tilingData->sliceSize = 1536;
    tilingData->sliceSizeAligned = 1536;
    tilingData->frontCoreNums = 48;
    tilingData->frontCoreIndicesNums = 2;
    tilingData->tailCoreIndicesNums = 0;
    tilingData->frontBlocks = 1;
    tilingData->frontBlockSize = 1536;
    tilingData->lastBlockSize = 0;
    tilingData->accumulate = 0;

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/index_put_with_sort_data/self.bin", selfSize, self, selfSize);
    ReadFile(path + "/index_put_with_sort_data/linear_index.bin", linearIndexSize, linearIndex, linearIndexSize);
    ReadFile(path + "/index_put_with_sort_data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);
    ReadFile(path + "/index_put_with_sort_data/values.bin", valuesSize, values, valuesSize);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(index_put_with_sort, 1, self, linearIndex, posIdx, values, self, workspace, tiling);

    AscendC::GmFree(self);
    AscendC::GmFree(linearIndex);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(values);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
    free(path_);
}

TEST_F(index_put_with_sort_test, test_case_gather_float_true) 
{
    size_t selfSize = 96 * 1536 * sizeof(float); // 1536 = 48 * 32
    size_t linearIndexSize = 96 * sizeof(int32_t);
    size_t posIdxSize = 96 * sizeof(int32_t);
    size_t valuesSize = 96 * 1536 * sizeof(float);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = sysWorkspaceSize;
    int64_t workspaceSize = sysWorkspaceSize + usrWorkspaceSize;

    size_t tilingSize = sizeof(IndexPutWithSortTilingData);

    uint8_t *self = (uint8_t *)AscendC::GmAlloc(selfSize);
    uint8_t *linearIndex = (uint8_t *)AscendC::GmAlloc(linearIndexSize);
    uint8_t *posIdx = (uint8_t *)AscendC::GmAlloc(posIdxSize);
    uint8_t *values = (uint8_t *)AscendC::GmAlloc(valuesSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    system("cp -r ../../../../index/index_put_with_sort/tests/ut/op_kernel/index_put_with_sort_data ./");
    system("chmod -R 755 ./index_put_with_sort_data/");
    system("cd ./index_put_with_sort_data/ && rm -rf ./*bin");
    system("cd ./index_put_with_sort_data/ && python3 gen_data.py 48 48 1536");
    
    IndexPutWithSortTilingData *tilingData = reinterpret_cast<IndexPutWithSortTilingData*>(tiling);
    tilingData->coreNums = 48;
    tilingData->sliceSize = 1536;
    tilingData->sliceSizeAligned = 1536;
    tilingData->frontCoreNums = 48;
    tilingData->frontCoreIndicesNums = 2;
    tilingData->tailCoreIndicesNums = 0;
    tilingData->frontBlocks = 1;
    tilingData->frontBlockSize = 1536;
    tilingData->lastBlockSize = 0;
    tilingData->accumulate = 1;

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/index_put_with_sort_data/self.bin", selfSize, self, selfSize);
    ReadFile(path + "/index_put_with_sort_data/linear_index.bin", linearIndexSize, linearIndex, linearIndexSize);
    ReadFile(path + "/index_put_with_sort_data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);
    ReadFile(path + "/index_put_with_sort_data/values.bin", valuesSize, values, valuesSize);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(index_put_with_sort, 1, self, linearIndex, posIdx, values, self, workspace, tiling);

    AscendC::GmFree(self);
    AscendC::GmFree(linearIndex);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(values);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
    free(path_);
}

TEST_F(index_put_with_sort_test, test_case_scatter_in_core_float_false) 
{
    size_t selfSize = 96 * 3072 * sizeof(float);
    size_t linearIndexSize = 96 * sizeof(int32_t);
    size_t posIdxSize = 96 * sizeof(int32_t);
    size_t valuesSize = 96 * 3072 * sizeof(float);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = sysWorkspaceSize;
    int64_t workspaceSize = sysWorkspaceSize + usrWorkspaceSize;

    size_t tilingSize = sizeof(IndexPutWithSortTilingData);

    uint8_t *self = (uint8_t *)AscendC::GmAlloc(selfSize);
    uint8_t *linearIndex = (uint8_t *)AscendC::GmAlloc(linearIndexSize);
    uint8_t *posIdx = (uint8_t *)AscendC::GmAlloc(posIdxSize);
    uint8_t *values = (uint8_t *)AscendC::GmAlloc(valuesSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    system("cp -r ../../../../index/index_put_with_sort/tests/ut/op_kernel/index_put_with_sort_data ./");
    system("chmod -R 755 ./index_put_with_sort_data/");
    system("cd ./index_put_with_sort_data/ && rm -rf ./*bin");
    system("cd ./index_put_with_sort_data/ && python3 gen_data.py 48 48 1536");
    
    IndexPutWithSortTilingData *tilingData = reinterpret_cast<IndexPutWithSortTilingData*>(tiling);
    tilingData->coreNums = 48;
    tilingData->sliceSize = 3072;
    tilingData->sliceSizeAligned = 3072;
    tilingData->frontCoreNums = 48;
    tilingData->frontCoreIndicesNums = 2;
    tilingData->tailCoreIndicesNums = 0;
    tilingData->frontBlocks = 1;
    tilingData->frontBlockSize = 3072;
    tilingData->lastBlockSize = 0;
    tilingData->accumulate = 0;

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/index_put_with_sort_data/self.bin", selfSize, self, selfSize);
    ReadFile(path + "/index_put_with_sort_data/linear_index.bin", linearIndexSize, linearIndex, linearIndexSize);
    ReadFile(path + "/index_put_with_sort_data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);
    ReadFile(path + "/index_put_with_sort_data/values.bin", valuesSize, values, valuesSize);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(index_put_with_sort, 1, self, linearIndex, posIdx, values, self, workspace, tiling);

    AscendC::GmFree(self);
    AscendC::GmFree(linearIndex);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(values);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
    free(path_);
}

TEST_F(index_put_with_sort_test, test_case_scatter_in_core_float_true) 
{
    size_t selfSize = 96 * 3072 * sizeof(float);
    size_t linearIndexSize = 96 * sizeof(int32_t);
    size_t posIdxSize = 96 * sizeof(int32_t);
    size_t valuesSize = 96 * 3072 * sizeof(float);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkspaceSize = sysWorkspaceSize;
    int64_t workspaceSize = sysWorkspaceSize + usrWorkspaceSize;

    size_t tilingSize = sizeof(IndexPutWithSortTilingData);

    uint8_t *self = (uint8_t *)AscendC::GmAlloc(selfSize);
    uint8_t *linearIndex = (uint8_t *)AscendC::GmAlloc(linearIndexSize);
    uint8_t *posIdx = (uint8_t *)AscendC::GmAlloc(posIdxSize);
    uint8_t *values = (uint8_t *)AscendC::GmAlloc(valuesSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    system("cp -r ../../../../index/index_put_with_sort/tests/ut/op_kernel/index_put_with_sort_data ./");
    system("chmod -R 755 ./index_put_with_sort_data/");
    system("cd ./index_put_with_sort_data/ && rm -rf ./*bin");
    system("cd ./index_put_with_sort_data/ && python3 gen_data.py 48 48 1536");
    
    IndexPutWithSortTilingData *tilingData = reinterpret_cast<IndexPutWithSortTilingData*>(tiling);
    tilingData->coreNums = 48;
    tilingData->sliceSize = 3072;
    tilingData->sliceSizeAligned = 3072;
    tilingData->frontCoreNums = 48;
    tilingData->frontCoreIndicesNums = 2;
    tilingData->tailCoreIndicesNums = 0;
    tilingData->frontBlocks = 1;
    tilingData->frontBlockSize = 3072;
    tilingData->lastBlockSize = 0;
    tilingData->accumulate = 1;

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/index_put_with_sort_data/self.bin", selfSize, self, selfSize);
    ReadFile(path + "/index_put_with_sort_data/linear_index.bin", linearIndexSize, linearIndex, linearIndexSize);
    ReadFile(path + "/index_put_with_sort_data/pos_idx.bin", posIdxSize, posIdx, posIdxSize);
    ReadFile(path + "/index_put_with_sort_data/values.bin", valuesSize, values, valuesSize);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(index_put_with_sort, 1, self, linearIndex, posIdx, values, self, workspace, tiling);

    AscendC::GmFree(self);
    AscendC::GmFree(linearIndex);
    AscendC::GmFree(posIdx);
    AscendC::GmFree(values);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
    free(path_);
}