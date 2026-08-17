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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "clipped_swiglu_grad_tiling_def.h"
#include "data_utils.h"
#include "../../../op_kernel/clipped_swiglu_grad.h"

using namespace std;
using namespace AscendC;

extern "C" __global__ __aicore__ void clipped_swiglu_grad(GM_ADDR gradYGM, GM_ADDR xGM, GM_ADDR groupIndexGM,
                                                          GM_ADDR gradXOutGM, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userspace = GetUserWorkspace(workspace);
    if (userspace == nullptr) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(ClippedSwigluGradTilingData, tilingData, tiling);
    TPipe pipe;

    if (tilingData.isInterleaved == 0) {
        if (tilingData.isGroup == 0) {
            ClippedSwigluGradOps::ClippedSwigluGradBase<DTYPE_X, false, false> op(&tilingData, &pipe);
            op.Init(gradYGM, xGM, groupIndexGM, gradXOutGM);
            op.Process();
        } else {
            ClippedSwigluGradOps::ClippedSwigluGradBase<DTYPE_X, false, true> op(&tilingData, &pipe);
            op.Init(gradYGM, xGM, groupIndexGM, gradXOutGM);
            op.Process();
        }
    } else {
        if (tilingData.isGroup == 0) {
            ClippedSwigluGradOps::ClippedSwigluGradBase<DTYPE_X, true, false> op(&tilingData, &pipe);
            op.Init(gradYGM, xGM, groupIndexGM, gradXOutGM);
            op.Process();
        } else {
            ClippedSwigluGradOps::ClippedSwigluGradBase<DTYPE_X, true, true> op(&tilingData, &pipe);
            op.Init(gradYGM, xGM, groupIndexGM, gradXOutGM);
            op.Process();
        }
    }
}

class clipped_swiglu_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "clipped_swiglu_grad_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "clipped_swiglu_grad_test TearDown\n" << endl; }
};

TEST_F(clipped_swiglu_grad_test, test_case_bf16_half_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(half);
    size_t gradYByteSize = 3200 * 2880 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_bf16_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_half_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, nullptr, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp16_half_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(half);
    size_t gradYByteSize = 3200 * 2880 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp16_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_half_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, nullptr, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp32_half_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(float);
    size_t gradYByteSize = 3200 * 2880 * sizeof(float);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp32_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_half_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, nullptr, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_bf16_interleaved_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(half);
    size_t gradYByteSize = 3200 * 2880 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_bf16_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_interleaved_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, nullptr, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp16_interleaved_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(half);
    size_t gradYByteSize = 3200 * 2880 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp16_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_interleaved_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, nullptr, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp32_interleaved_ungrouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(float);
    size_t gradYByteSize = 3200 * 2880 * sizeof(float);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp32_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_interleaved_ungrouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, nullptr, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_bf16_half_grouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(half);
    size_t gradYByteSize = 3200 * 2880 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_bf16_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_half_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp16_half_grouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(half);
    size_t gradYByteSize = 3200 * 2880 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp16_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_half_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp32_half_grouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(float);
    size_t gradYByteSize = 3200 * 2880 * sizeof(float);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp32_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_half_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_bf16_interleaved_grouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(half);
    size_t gradYByteSize = 3200 * 2880 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_bf16_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_interleaved_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp16_interleaved_grouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(half);
    size_t gradYByteSize = 3200 * 2880 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp16_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_interleaved_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp32_interleaved_grouped_shortH)
{
    size_t inputByteSize = 3200 * 5760 * sizeof(float);
    size_t gradYByteSize = 3200 * 2880 * sizeof(float);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp32_shortH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_interleaved_grouped_shortH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_bf16_half_grouped_longH)
{
    size_t inputByteSize = 3200 * 23040 * sizeof(half);
    size_t gradYByteSize = 3200 * 11520 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_bf16_longH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_half_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp16_half_grouped_longH)
{
    size_t inputByteSize = 3200 * 23040 * sizeof(half);
    size_t gradYByteSize = 3200 * 11520 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp16_longH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_half_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp32_half_grouped_longH)
{
    size_t inputByteSize = 3200 * 23040 * sizeof(float);
    size_t gradYByteSize = 3200 * 11520 * sizeof(float);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp32_longH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_half_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_bf16_interleaved_grouped_longH)
{
    size_t inputByteSize = 3200 * 23040 * sizeof(half);
    size_t gradYByteSize = 3200 * 11520 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_bf16_longH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_interleaved_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp16_interleaved_grouped_longH)
{
    size_t inputByteSize = 3200 * 23040 * sizeof(half);
    size_t gradYByteSize = 3200 * 11520 * sizeof(half);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp16_longH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_interleaved_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}

TEST_F(clipped_swiglu_grad_test, test_case_fp32_interleaved_grouped_longH)
{
    size_t inputByteSize = 3200 * 23040 * sizeof(float);
    size_t gradYByteSize = 3200 * 11520 * sizeof(float);
    size_t outputByteSize = inputByteSize;
    size_t groupIndexByteSize = 10 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(ClippedSwigluGradTilingData);

    uint8_t* gradY = (uint8_t*)AscendC::GmAlloc(gradYByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* gradXOut = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint64_t tilingKey = 1;
    uint32_t blockDim = 40;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../activation/clipped_swiglu_grad/tests/ut/op_kernel/clipped_swiglu_grad_data ./");
    system("chmod -R 755 ./clipped_swiglu_grad_data/");
    system("cd ./clipped_swiglu_grad_data/ && rm -rf ./*bin");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_data.py test_case_fp32_longH");
    system("cd ./clipped_swiglu_grad_data/ && python3 gen_tiling.py case_interleaved_grouped_longH");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/clipped_swiglu_grad_data/input_grad_y.bin", gradYByteSize, gradY, gradYByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/input_group_index.bin", groupIndexByteSize, groupIndex,
             groupIndexByteSize);
    ReadFile(path + "/clipped_swiglu_grad_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(clipped_swiglu_grad, blockDim, gradY, x, groupIndex, gradXOut, workspace, tiling);

    AscendC::GmFree((void*)gradY);
    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)groupIndex);
    AscendC::GmFree((void*)gradXOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    free(path_);
}
