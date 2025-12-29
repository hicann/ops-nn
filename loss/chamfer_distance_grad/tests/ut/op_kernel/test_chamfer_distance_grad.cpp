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
    uint32_t batch_size = 2;
    uint32_t num = 2;
    // inputs
    size_t xyz_size = batch_size * num * 2 * sizeof(float) * 2;
    size_t grad_size = batch_size * num * sizeof(float) * 2;
    size_t idx_size = batch_size * num * sizeof(int32_t) * 2;
    size_t grad_xyz_size = batch_size * num * sizeof(float) * 2 * 2;
    size_t tiling_data_size = sizeof(ChamferDistanceGradTilingDataTest) + 1024;

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
    uint32_t blockDim = 1;
    ChamferDistanceGradTilingDataTest* tilingDatafromBin = reinterpret_cast<ChamferDistanceGradTilingDataTest*>(tiling);

    tilingDatafromBin->batch_size = 2;
    tilingDatafromBin->num = 2;
    tilingDatafromBin->ub_size = 195538;
    tilingDatafromBin->task_per_core = 1;
    tilingDatafromBin->core_used = 4;
    tilingDatafromBin->task_tail_core = 1;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
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
    ChamferDistanceGradTilingDataTest* tilingDatafromBin = reinterpret_cast<ChamferDistanceGradTilingDataTest*>(tiling);

    tilingDatafromBin->batch_size = 2;
    tilingDatafromBin->num = 2;
    tilingDatafromBin->ub_size = 195538;
    tilingDatafromBin->task_per_core = 1;
    tilingDatafromBin->core_used = 4;
    tilingDatafromBin->task_tail_core = 1;

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
}