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
#include "apply_top_k_top_p_with_sorted_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void apply_top_k_top_p_with_sorted(
    GM_ADDR sorted_value, GM_ADDR sorted_indices, GM_ADDR p, GM_ADDR k, GM_ADDR out, GM_ADDR workSpace, GM_ADDR tiling);

class apply_top_k_top_p_with_sorted_test : public testing::Test
{
    protected:

    static void SetUpTestCase()
    {
        cout << "apply_top_k_top_p_with_sorted_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "apply_top_k_top_p_with_sorted_test TearDown\n" << endl;
    }
};

TEST_F(apply_top_k_top_p_with_sorted_test, test_case_4_152064_fp32)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    size_t sortedValueSize = size_t(4) * size_t(152064) * sizeof(float);
    size_t sortedIndicesSize = size_t(4) * size_t(152064) * sizeof(int32_t);
    size_t pSize = size_t(4) * sizeof(float);
    size_t kSize = size_t(4) * sizeof(int32_t);
    size_t outSize = size_t(4) * size_t(152064) * sizeof(float);
    size_t sysWorkspaceSize = size_t(16) * size_t(1024) * size_t(1024);
    size_t workspaceSize = sysWorkspaceSize;

    size_t tilingSize = sizeof(ApplyTopKTopPWithSortedTilingData);

    uint8_t *sortedValue = (uint8_t *)AscendC::GmAlloc(sortedValueSize);
    uint8_t *sortedIndices = (uint8_t *)AscendC::GmAlloc(sortedIndicesSize);
    uint8_t *p = (uint8_t *)AscendC::GmAlloc(pSize);
    uint8_t *k = (uint8_t *)AscendC::GmAlloc(kSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    system("cp -r ../../../../index/apply_top_k_top_p_with_sorted/tests/ut/op_kernel/apply_top_k_top_p_with_sorted_data ./");
    system("chmod -R 755 ./apply_top_k_top_p_with_sorted_data/");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && rm -rf ./*bin");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && python3 gen_data.py 4 152064 fp32");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && python3 gen_tiling.py test_case_4_152064_fp32");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/sortedValue.bin", sortedValueSize, sortedValue, sortedValueSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/sortedIndices.bin", sortedIndicesSize, sortedIndices, sortedIndicesSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/p.bin", pSize, p, pSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/k.bin", kSize, k, kSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(apply_top_k_top_p_with_sorted, 1, sortedValue, sortedIndices, p, k, out, workspace, tiling);

    AscendC::GmFree(sortedValue);
    AscendC::GmFree(sortedIndices);
    AscendC::GmFree(p);
    AscendC::GmFree(k);
    AscendC::GmFree(out);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
    free(path_);
}

TEST_F(apply_top_k_top_p_with_sorted_test, test_case_4_152064_fp32_topp)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    size_t sortedValueSize = 4 * 152064 * sizeof(float);
    size_t sortedIndicesSize = 4 * 152064 * sizeof(int32_t);
    size_t pSize = 4 * sizeof(float);
    size_t outSize = 4 * 152064 * sizeof(float);
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    int64_t workspaceSize = sysWorkspaceSize;

    size_t tilingSize = sizeof(ApplyTopKTopPWithSortedTilingData);

    uint8_t *sortedValue = (uint8_t *)AscendC::GmAlloc(sortedValueSize);
    uint8_t *sortedIndices = (uint8_t *)AscendC::GmAlloc(sortedIndicesSize);
    uint8_t *p = (uint8_t *)AscendC::GmAlloc(pSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    system("cp -r ../../../../index/apply_top_k_top_p_with_sorted/tests/ut/op_kernel/apply_top_k_top_p_with_sorted_data ./");
    system("chmod -R 755 ./apply_top_k_top_p_with_sorted_data/");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && rm -rf ./*bin");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && python3 gen_data.py 4 152064 fp32");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && python3 gen_tiling.py test_case_4_152064_fp32");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/sortedValue.bin", sortedValueSize, sortedValue, sortedValueSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/sortedIndices.bin", sortedIndicesSize, sortedIndices, sortedIndicesSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/p.bin", pSize, p, pSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(apply_top_k_top_p_with_sorted, 1, sortedValue, sortedIndices, p, nullptr, out, workspace, tiling);

    AscendC::GmFree(sortedValue);
    AscendC::GmFree(sortedIndices);
    AscendC::GmFree(p);
    AscendC::GmFree(out);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
    free(path_);
}

TEST_F(apply_top_k_top_p_with_sorted_test, test_case_4_152064_fp16)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    size_t sortedValueSize = size_t(4) * size_t(152064) * sizeof(half);
    size_t sortedIndicesSize = size_t(4) * size_t(152064) * sizeof(int32_t);
    size_t pSize = size_t(4) * sizeof(half);
    size_t kSize = size_t(4) * sizeof(int32_t);
    size_t outSize = size_t(4) * size_t(152064) * sizeof(half);
    size_t sysWorkspaceSize = size_t(16) * size_t(1024) * size_t(1024);
    size_t workspaceSize = sysWorkspaceSize;

    size_t tilingSize = sizeof(ApplyTopKTopPWithSortedTilingData);

    uint8_t *sortedValue = (uint8_t *)AscendC::GmAlloc(sortedValueSize);
    uint8_t *sortedIndices = (uint8_t *)AscendC::GmAlloc(sortedIndicesSize);
    uint8_t *p = (uint8_t *)AscendC::GmAlloc(pSize);
    uint8_t *k = (uint8_t *)AscendC::GmAlloc(kSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    system("cp -r ../../../../index/apply_top_k_top_p_with_sorted/tests/ut/op_kernel/apply_top_k_top_p_with_sorted_data ./");
    system("chmod -R 755 ./apply_top_k_top_p_with_sorted_data/");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && rm -rf ./*bin");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && python3 gen_data.py 4 152064 fp16");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && python3 gen_tiling.py test_case_4_152064_fp16");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/sortedValue.bin", sortedValueSize, sortedValue, sortedValueSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/sortedIndices.bin", sortedIndicesSize, sortedIndices, sortedIndicesSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/p.bin", pSize, p, pSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/k.bin", kSize, k, kSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(apply_top_k_top_p_with_sorted, 1, sortedValue, sortedIndices, p, k, out, workspace, tiling);

    AscendC::GmFree(sortedValue);
    AscendC::GmFree(sortedIndices);
    AscendC::GmFree(p);
    AscendC::GmFree(k);
    AscendC::GmFree(out);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
    free(path_);
}

TEST_F(apply_top_k_top_p_with_sorted_test, test_case_4_152064_bf16)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    size_t sortedValueSize = size_t(4) * size_t(152064) * sizeof(bfloat16_t);
    size_t sortedIndicesSize = size_t(4) * size_t(152064) * sizeof(int32_t);
    size_t pSize = size_t(4) * sizeof(bfloat16_t);
    size_t kSize = size_t(4) * sizeof(int32_t);
    size_t outSize = size_t(4) * size_t(152064) * sizeof(bfloat16_t);
    size_t sysWorkspaceSize = size_t(16) * size_t(1024) * size_t(1024);
    size_t workspaceSize = sysWorkspaceSize;

    size_t tilingSize = sizeof(ApplyTopKTopPWithSortedTilingData);

    uint8_t *sortedValue = (uint8_t *)AscendC::GmAlloc(sortedValueSize);
    uint8_t *sortedIndices = (uint8_t *)AscendC::GmAlloc(sortedIndicesSize);
    uint8_t *p = (uint8_t *)AscendC::GmAlloc(pSize);
    uint8_t *k = (uint8_t *)AscendC::GmAlloc(kSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(outSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset(workspace, 0, workspaceSize);

    system("cp -r ../../../../index/apply_top_k_top_p_with_sorted/tests/ut/op_kernel/apply_top_k_top_p_with_sorted_data ./");
    system("chmod -R 755 ./apply_top_k_top_p_with_sorted_data/");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && rm -rf ./*bin");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && python3 gen_data.py 4 152064 bf16");
    system("cd ./apply_top_k_top_p_with_sorted_data/ && python3 gen_tiling.py test_case_4_152064_bf16");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/sortedValue.bin", sortedValueSize, sortedValue, sortedValueSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/sortedIndices.bin", sortedIndicesSize, sortedIndices, sortedIndicesSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/p.bin", pSize, p, pSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/k.bin", kSize, k, kSize);
    ReadFile(path + "/apply_top_k_top_p_with_sorted_data/tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(apply_top_k_top_p_with_sorted, 1, sortedValue, sortedIndices, p, k, out, workspace, tiling);

    AscendC::GmFree(sortedValue);
    AscendC::GmFree(sortedIndices);
    AscendC::GmFree(p);
    AscendC::GmFree(k);
    AscendC::GmFree(out);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
    free(path_);
}
