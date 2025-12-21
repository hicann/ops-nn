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
#include "test_chamfer_distance_grad_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void chamfer_distance_grad(
    GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR idx1, GM_ADDR idx2, GM_ADDR grad_dist1, GM_ADDR grad_dist2, GM_ADDR grad_xyz1,
    GM_ADDR grad_xyz2, GM_ADDR workspace, GM_ADDR tiling_data);
class chamfer_distance_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "chamfer_distance_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "chamfer_distance_grad_test TearDown\n" << endl;
    }
};

TEST_F(chamfer_distance_grad_test, test_case_fp32)
{
    uint32_t batch_size = 32;
    uint32_t num = 10000;

    // inputs
    size_t xyz_size = batch_size * num * 2 * sizeof(float);
    size_t grad_size = batch_size * num * sizeof(float);
    size_t idx_size = batch_size * num * sizeof(int32_t);
    size_t grad_xyz_size = batch_size * num * sizeof(float) * 2;
    size_t tiling_data_size = sizeof(ChamferDistanceGradTilingDataTest);

    uint8_t* xyz1 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* xyz2 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* grad_dist1 = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grad_dist2 = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* idx1 = (uint8_t*)AscendC::GmAlloc(idx_size);
    uint8_t* idx2 = (uint8_t*)AscendC::GmAlloc(idx_size);
    uint8_t* grad_xyz1 = (uint8_t*)AscendC::GmAlloc(grad_xyz_size);
    uint8_t* grad_xyz2 = (uint8_t*)AscendC::GmAlloc(grad_xyz_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    system(
        "cp -r "
        "../../../../loss/chamfer_distance_grad/tests/ut/op_kernel/chamfer_distance_grad_data ./");
    system("chmod -R 755 ./chamfer_distance_grad_data/");
    system("cd ./chamfer_distance_grad_data/ && rm -rf ./*bin");
    system("cd ./chamfer_distance_grad_data/ && python3 gen_data.py 32 10000 float");
    system("cd ./chamfer_distance_grad_data/ && python3 gen_tiling.py 32 10000");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/build/input/xyz1.bin", xyz_size, xyz1, xyz_size);
    ReadFile(path + "/build/input/xyz2.bin", xyz_size, xyz2, xyz_size);
    ReadFile(path + "/build/input/idx1.bin", idx_size, idx1, idx_size);
    ReadFile(path + "/build/input/idx2.bin", idx_size, idx2, idx_size);
    ReadFile(path + "/build/input/grad_dist1.bin", grad_size, grad_dist1, grad_size);
    ReadFile(path + "/build/input/grad_dist2.bin", grad_size, grad_dist2, grad_size);
    ReadFile(path + "/build/output/grad_xyz1.bin", grad_xyz_size, grad_xyz1, grad_xyz_size);
    ReadFile(path + "/build/output/grad_xyz2.bin", grad_xyz_size, grad_xyz2, grad_xyz_size);
    ReadFile(path + "/build/input/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        chamfer_distance_grad, blockDim, xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2, grad_xyz1, grad_xyz2,
        workspace, tiling);
    AscendC::GmFree(xyz1);
    AscendC::GmFree(xyz2);
    AscendC::GmFree(grad_dist1);
    AscendC::GmFree(grad_dist2);
    AscendC::GmFree(idx1);
    AscendC::GmFree(idx2);
    AscendC::GmFree(grad_xyz1);
    AscendC::GmFree(grad_xyz2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(chamfer_distance_grad_test, test_case_fp16)
{
    uint32_t batch_size = 32;
    uint32_t num = 10000;

    // inputs
    size_t xyz_size = batch_size * num * 2 * sizeof(half);
    size_t grad_size = batch_size * num * sizeof(half);
    size_t idx_size = batch_size * num * sizeof(int32_t);
    size_t grad_xyz_size = batch_size * num * sizeof(half) * 2;
    size_t tiling_data_size = sizeof(ChamferDistanceGradTilingDataTest);

    uint8_t* xyz1 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* xyz2 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* grad_dist1 = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grad_dist2 = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* idx1 = (uint8_t*)AscendC::GmAlloc(idx_size);
    uint8_t* idx2 = (uint8_t*)AscendC::GmAlloc(idx_size);
    uint8_t* grad_xyz1 = (uint8_t*)AscendC::GmAlloc(grad_xyz_size);
    uint8_t* grad_xyz2 = (uint8_t*)AscendC::GmAlloc(grad_xyz_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    system(
        "cp -r "
        "../../../../loss/chamfer_distance_grad/tests/ut/op_kernel/chamfer_distance_grad_data ./");
    system("chmod -R 755 ./chamfer_distance_grad_data/");
    system("cd ./chamfer_distance_grad_data/ && rm -rf ./*bin");
    system("cd ./chamfer_distance_grad_data/ && python3 gen_data.py 32 10000 float16");
    system("cd ./chamfer_distance_grad_data/ && python3 gen_tiling.py 32 10000");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/build/input/xyz1.bin", xyz_size, xyz1, xyz_size);
    ReadFile(path + "/build/input/xyz2.bin", xyz_size, xyz2, xyz_size);
    ReadFile(path + "/build/input/idx1.bin", idx_size, idx1, idx_size);
    ReadFile(path + "/build/input/idx2.bin", idx_size, idx2, idx_size);
    ReadFile(path + "/build/input/grad_dist1.bin", grad_size, grad_dist1, grad_size);
    ReadFile(path + "/build/input/grad_dist2.bin", grad_size, grad_dist2, grad_size);
    ReadFile(path + "/build/output/grad_xyz1.bin", grad_xyz_size, grad_xyz1, grad_xyz_size);
    ReadFile(path + "/build/output/grad_xyz2.bin", grad_xyz_size, grad_xyz2, grad_xyz_size);
    ReadFile(path + "/build/input/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        chamfer_distance_grad, blockDim, xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2, grad_xyz1, grad_xyz2,
        workspace, tiling);
    AscendC::GmFree(xyz1);
    AscendC::GmFree(xyz2);
    AscendC::GmFree(grad_dist1);
    AscendC::GmFree(grad_dist2);
    AscendC::GmFree(idx1);
    AscendC::GmFree(idx2);
    AscendC::GmFree(grad_xyz1);
    AscendC::GmFree(grad_xyz2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}