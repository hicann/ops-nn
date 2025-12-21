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
#include "test_add_rms_norm_quant_v2_def.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void add_rms_norm_quant_v2(
  GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1,GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2,
  GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR res_out, GM_ADDR workspace, GM_ADDR tiling);

namespace {
class add_rms_norm_quant_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_rms_norm_quant_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_rms_norm_quant_v2_test TearDown\n" << endl;
    }
};
} // namespace

#define InitParams()                                                                                         \
    size_t row = 64;                                                                                          \
    size_t col = 2560;                                                                                        \
    uint32_t numRow = 64U;                                                                                    \
    uint32_t numCol = 2560U;                                                                                  \
    uint32_t blockFactor = 2U;                                                                               \
    uint32_t rowFactor = 64U;                                                                                \
    uint32_t ubFactor = 4864U;                                                                              \
    float epsilon = 0.01;                                                                                    \
    float floatavgFactor = 0.01;                                                                             \
    size_t inputByteSize = row * col * sizeof(int16_t);                                                      \
    size_t gammaByteSize = col * sizeof(int16_t);                                                            \
    size_t scalesByteSize = col * sizeof(float);                                                             \
    size_t zeroPointsByteSize = col * sizeof(int32_t);                                                       \
    size_t outputYByteSize = row * col * sizeof(int8_t);                                                     \
    size_t outputXByteSize = row * col * sizeof(int16_t);                                                    \
    size_t tiling_data_size = sizeof(AddRMSNormQuantV2TilingData);                                           \
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);                                                 \
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);                                                 \
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);                                              \
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(gammaByteSize);                                               \
    uint8_t* scales1 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);                                           \
    uint8_t* scales2 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);                                           \
    uint8_t* zero_points1 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);                                      \
    uint8_t* zero_points2 = (uint8_t*)AscendC::GmAlloc(scalesByteSize);                                      \
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputYByteSize);                                               \
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputYByteSize);                                               \
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(outputXByteSize);                                                \
    uint8_t* res_out = (uint8_t*)AscendC::GmAlloc(outputXByteSize);                                          \
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);                                                 \
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);                                          \
    uint32_t blockDim = 32;                                                                              \
    char* path_ = get_current_dir_name();                                                                    \
    string path(path_);                                                                                      \
    AddRMSNormQuantV2TilingData* tilingDatafromBin = reinterpret_cast<AddRMSNormQuantV2TilingData*>(tiling); \
    tilingDatafromBin->numRow = numRow;                                                                      \
    tilingDatafromBin->numCol = numCol;                                                                      \
    tilingDatafromBin->blockFactor = blockFactor;                                                            \
    tilingDatafromBin->rowFactor = rowFactor;                                                                \
    tilingDatafromBin->ubFactor = ubFactor;                                                                  \
    tilingDatafromBin->epsilon = 0.01;                                                                       \
    tilingDatafromBin->avgFactor = 0.01;                                                                     \
    tilingDatafromBin->hasZeroPoints1 = 1U;                                                                  \
    tilingDatafromBin->hasBeta = 1U;

#define FreeGM()                   \
    AscendC::GmFree(x1);           \
    AscendC::GmFree(x2);           \
    AscendC::GmFree(gamma);        \
    AscendC::GmFree(beta);         \
    AscendC::GmFree(scales1);      \
    AscendC::GmFree(scales2);      \
    AscendC::GmFree(zero_points1); \
    AscendC::GmFree(zero_points2); \
    AscendC::GmFree(y1);           \
    AscendC::GmFree(y2);           \
    AscendC::GmFree(x);            \
    AscendC::GmFree(workspace);    \
    AscendC::GmFree(tiling);       \
    free(path_);

TEST_F(add_rms_norm_quant_v2_test, test_case_1)
{
    InitParams();
    uint32_t tiling_key = 10U;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(
        add_rms_norm_quant_v2, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
        res_out, workspace, (uint8_t*)(tilingDatafromBin));
    FreeGM();
}

TEST_F(add_rms_norm_quant_v2_test, test_case_2)
{
    InitParams();
    uint32_t tiling_key = 10010U;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(
        add_rms_norm_quant_v2, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
        res_out, workspace, (uint8_t*)(tilingDatafromBin));
    FreeGM();
}

// TEST_F(add_rms_norm_quant_v2_test, test_case_3)
// {
//     InitParams();
//     uint32_t tiling_key = 30010U;
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_SET_TILING_KEY(tiling_key);
//     ICPU_RUN_KF(
//         add_rms_norm_quant_v2, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
//         res_out, workspace, (uint8_t*)(tilingDatafromBin));
//     FreeGM();
// }

TEST_F(add_rms_norm_quant_v2_test, test_case_4)
{
    InitParams();
    uint32_t tiling_key = 101U;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(
        add_rms_norm_quant_v2, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
        res_out, workspace, (uint8_t*)(tilingDatafromBin));
    FreeGM();
}

TEST_F(add_rms_norm_quant_v2_test, test_case_5)
{
    InitParams();
    uint32_t tiling_key = 10101U;
    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(
        add_rms_norm_quant_v2, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
        res_out, workspace, (uint8_t*)(tilingDatafromBin));
    FreeGM();
}

// TEST_F(add_rms_norm_quant_v2_test, test_case_6)
// {
//     InitParams();
//     uint32_t tiling_key = 30101U;
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_SET_TILING_KEY(tiling_key);
//     ICPU_RUN_KF(
//         add_rms_norm_quant_v2, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
//         res_out, workspace, (uint8_t*)(tilingDatafromBin));
//     FreeGM();
// }

TEST_F(add_rms_norm_quant_v2_test, test_case_7)
{
    InitParams();
    uint32_t tiling_key = 1011U;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(
        add_rms_norm_quant_v2, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
        res_out, workspace, (uint8_t*)(tilingDatafromBin));
    FreeGM();
}

TEST_F(add_rms_norm_quant_v2_test, test_case_8)
{
    InitParams();
    uint32_t tiling_key = 11011U;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tiling_key);
    ICPU_RUN_KF(
        add_rms_norm_quant_v2, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
        res_out, workspace, (uint8_t*)(tilingDatafromBin));
    FreeGM();
}

// TEST_F(add_rms_norm_quant_v2_test, test_case_9)
// {
//     InitParams();
//     uint32_t tiling_key = 31011U;
//     AscendC::SetKernelMode(KernelMode::AIV_MODE);
//     ICPU_SET_TILING_KEY(tiling_key);
//     ICPU_RUN_KF(
//         add_rms_norm_quant_v2, blockDim, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, x,
//         res_out, workspace, (uint8_t*)(tilingDatafromBin));
//     FreeGM();
// }