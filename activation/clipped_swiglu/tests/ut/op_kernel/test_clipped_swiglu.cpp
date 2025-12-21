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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "clipped_swiglu_tiling_def.h"
#include "data_utils.h"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void clipped_swiglu(
    GM_ADDR xGM, GM_ADDR groupIndexGM, GM_ADDR yGM, GM_ADDR workspace, GM_ADDR tiling);

class clipped_swiglu_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "clipped_swiglu_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "clipped_swiglu_test TearDown\n" << endl;
    }
};

TEST_F(clipped_swiglu_test, test_case_bf16_half_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 2880 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_bf16_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_half_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, nullptr, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp16_half_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 2880 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp16_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_half_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, nullptr, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp32_half_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(float);
    size_t outputByteSize = 3200 * 2880 * sizeof(float);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp32_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_half_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, nullptr, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_bf16_interleaved_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 2880 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_bf16_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_interleaved_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, nullptr, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp16_interleaved_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 2880 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp16_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_interleaved_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, nullptr, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp32_interleaved_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(float);
    size_t outputByteSize = 3200 * 2880 * sizeof(float);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp32_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_interleaved_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, nullptr, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_bf16_half_grouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 2880 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_bf16_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_half_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp16_half_grouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 2880 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp16_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_half_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp32_half_grouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(float);
    size_t outputByteSize = 3200 * 2880 * sizeof(float);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp32_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_half_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_bf16_interleaved_grouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 2880 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_bf16_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_interleaved_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp16_interleaved_grouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 2880 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp16_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_interleaved_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp32_interleaved_grouped_shortH)
{
    size_t inputByteSize = 3200 * 2880 * 2 * sizeof(float);
    size_t outputByteSize = 3200 * 2880 * sizeof(float);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp32_shortH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_interleaved_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_bf16_half_grouped_longH)
{
    size_t inputByteSize = 3200 * 11520 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 11520 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_bf16_longH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_half_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp16_half_grouped_longH)
{
    size_t inputByteSize = 3200 * 11520 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 11520 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp16_longH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_half_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp32_half_grouped_longH)
{
    size_t inputByteSize = 3200 * 11520 * 2 * sizeof(float);
    size_t outputByteSize = 3200 * 11520 * sizeof(float);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp32_longH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_half_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_bf16_interleaved_grouped_longH)
{
    size_t inputByteSize = 3200 * 11520 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 11520 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_bf16_longH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_interleaved_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp16_interleaved_grouped_longH)
{
    size_t inputByteSize = 3200 * 11520 * 2 * sizeof(half);
    size_t outputByteSize = 3200 * 11520 * sizeof(half);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp16_longH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_interleaved_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_test, test_case_fp32_interleaved_grouped_longH)
{
    size_t inputByteSize = 3200 * 11520 * 2 * sizeof(float);
    size_t outputByteSize = 3200 * 11520 * sizeof(float);
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tiling_data_size = sizeof(ClippedSwigluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(ClippedSwigluTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu/tests/ut/op_kernel/clipped_swiglu_data ./");
    system("chmod -R 755 ./clipped_swiglu_data/");
    system("cd ./clipped_swiglu_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_data/ && python3 gen_data.py test_case_fp32_longH");
    system("cd ./clipped_swiglu_data/ && python3 gen_tiling.py case_interleaved_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/clipped_swiglu_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_data/input_group_index.bin", groupIndexByteSize, groupIndex, groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    
    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu, blockDim,
                x, groupIndex, y, workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}