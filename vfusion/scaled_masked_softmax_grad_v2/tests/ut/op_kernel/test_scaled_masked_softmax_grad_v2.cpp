/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <cstdint>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "scaled_masked_softmax_grad_v2_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void scaled_masked_softmax_grad_v2(
    GM_ADDR yGrad,
    GM_ADDR y,
    GM_ADDR mask,
    GM_ADDR xGrad,
    GM_ADDR workspace,
    GM_ADDR tiling);

class scaled_masked_softmax_grad_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "scaled_masked_softmax_grad_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "scaled_masked_softmax_grad_v2_test TearDown\n" << endl;
    }
};

TEST_F(scaled_masked_softmax_grad_v2_test, test_tiling_key_2000_and_maskmode0)
{
    uint64_t B = 8;
    uint64_t N = 16;
    uint64_t S = 64;
    uint64_t D = 1536;
    uint64_t coreNum = 40;
    uint64_t totalLine = B * N * S;
    size_t inputAndOutputSize = totalLine * D * sizeof(DT_BF16);
    size_t inputMaskSize = totalLine * D * sizeof(bool);
    size_t tilingDataSize = sizeof(ScaledMaskedSoftmaxGradV2TilingData);

    uint8_t* yGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskSize);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    ScaledMaskedSoftmaxGradV2TilingData* tilingDatafromBin =
        reinterpret_cast<ScaledMaskedSoftmaxGradV2TilingData*>(tiling);

    ICPU_SET_TILING_KEY(2000);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_grad_v2/tests/ut/op_kernel/scaled_masked_softmax_grad_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_grad_v2_data/");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_data.py 8 16 64 1536 8 16 bfloat16");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_tiling.py test_tiling_key_2000_and_maskmode0");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(
        path + "/scaled_masked_softmax_grad_v2_data/input_y_grad.bin", inputAndOutputSize, yGrad, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_y.bin", inputAndOutputSize, y, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_mask.bin", inputMaskSize, mask, inputMaskSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    ICPU_RUN_KF(scaled_masked_softmax_grad_v2, coreNum, yGrad, y, mask, xGrad, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree(yGrad);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scaled_masked_softmax_grad_v2_test, test_tiling_key_2001_and_maskmode1)
{
    uint64_t B = 4;
    uint64_t N = 8;
    uint64_t S = 32;
    uint64_t D = 1536;
    uint64_t coreNum = 40;
    uint64_t totalLine = B * N * S;
    size_t inputAndOutputSize = totalLine * D * sizeof(half);
    size_t inputMaskSize = N * S * D * sizeof(bool);
    size_t tilingDataSize = sizeof(ScaledMaskedSoftmaxGradV2TilingData);

    uint8_t* yGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskSize);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    ScaledMaskedSoftmaxGradV2TilingData* tilingDatafromBin =
        reinterpret_cast<ScaledMaskedSoftmaxGradV2TilingData*>(tiling);

    ICPU_SET_TILING_KEY(2001);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_grad_v2/tests/ut/op_kernel/scaled_masked_softmax_grad_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_grad_v2_data/");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_data.py 4 8 32 1536 1 8 half");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_tiling.py test_tiling_key_2001_and_maskmode1");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(
        path + "/scaled_masked_softmax_grad_v2_data/input_y_grad.bin", inputAndOutputSize, yGrad, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_y.bin", inputAndOutputSize, y, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_mask.bin", inputMaskSize, mask, inputMaskSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);
    ICPU_RUN_KF(scaled_masked_softmax_grad_v2, coreNum, yGrad, y, mask, xGrad, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree(yGrad);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scaled_masked_softmax_grad_v2_test, test_tiling_key_2002_and_maskmode2)
{
    uint64_t B = 8;
    uint64_t N = 16;
    uint64_t S = 64;
    uint64_t D = 1536;
    uint64_t coreNum = 40;
    uint64_t totalLine = B * N * S;
    size_t inputAndOutputSize = totalLine * D * sizeof(float);
    size_t inputMaskSize = B * S * D * sizeof(bool);
    size_t tilingDataSize = sizeof(ScaledMaskedSoftmaxGradV2TilingData);

    uint8_t* yGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskSize);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    ScaledMaskedSoftmaxGradV2TilingData* tilingDatafromBin =
        reinterpret_cast<ScaledMaskedSoftmaxGradV2TilingData*>(tiling);

    ICPU_SET_TILING_KEY(2002);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_grad_v2/tests/ut/op_kernel/scaled_masked_softmax_grad_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_grad_v2_data/");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_data.py 8 16 64 1536 8 1 float");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_tiling.py test_tiling_key_2002_and_maskmode2");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(
        path + "/scaled_masked_softmax_grad_v2_data/input_y_grad.bin", inputAndOutputSize, yGrad, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_y.bin", inputAndOutputSize, y, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_mask.bin", inputMaskSize, mask, inputMaskSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);
    ICPU_RUN_KF(scaled_masked_softmax_grad_v2, coreNum, yGrad, y, mask, xGrad, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree(yGrad);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scaled_masked_softmax_grad_v2_test, test_tiling_key_2002_and_not_align)
{
    uint64_t B = 4;
    uint64_t N = 8;
    uint64_t S = 32;
    uint64_t D = 1025;
    uint64_t coreNum = 40;
    uint64_t totalLine = B * N * S;
    size_t inputAndOutputSize = totalLine * D * sizeof(float);
    size_t inputMaskSize = totalLine * D * sizeof(bool);
    size_t tilingDataSize = sizeof(ScaledMaskedSoftmaxGradV2TilingData);

    uint8_t* yGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskSize);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    ScaledMaskedSoftmaxGradV2TilingData* tilingDatafromBin =
        reinterpret_cast<ScaledMaskedSoftmaxGradV2TilingData*>(tiling);
    ICPU_SET_TILING_KEY(2002);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_grad_v2/tests/ut/op_kernel/scaled_masked_softmax_grad_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_grad_v2_data/");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_data.py 4 8 32 1025 4 8 float");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_tiling.py test_tiling_key_2002_and_not_align");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(
        path + "/scaled_masked_softmax_grad_v2_data/input_y_grad.bin", inputAndOutputSize, yGrad, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_y.bin", inputAndOutputSize, y, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_mask.bin", inputMaskSize, mask, inputMaskSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);
    ICPU_RUN_KF(scaled_masked_softmax_grad_v2, coreNum, yGrad, y, mask, xGrad, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree(yGrad);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scaled_masked_softmax_grad_v2_test, test_tiling_key_1000_and_maskmode3)
{
    uint64_t B = 8;
    uint64_t N = 16;
    uint64_t S = 64;
    uint64_t D = 256;
    uint64_t coreNum = 40;
    uint64_t totalLine = B * N * S;
    size_t inputAndOutputSize = totalLine * D * sizeof(DT_BF16);
    size_t inputMaskSize = S * D * sizeof(bool);
    size_t tilingDataSize = sizeof(ScaledMaskedSoftmaxGradV2TilingData);

    uint8_t* yGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskSize);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    ScaledMaskedSoftmaxGradV2TilingData* tilingDatafromBin =
        reinterpret_cast<ScaledMaskedSoftmaxGradV2TilingData*>(tiling);

    ICPU_SET_TILING_KEY(1000);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_grad_v2/tests/ut/op_kernel/scaled_masked_softmax_grad_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_grad_v2_data/");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_data.py 8 16 64 256 1 1 bfloat16");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_tiling.py test_tiling_key_1000_and_maskmode3");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(
        path + "/scaled_masked_softmax_grad_v2_data/input_y_grad.bin", inputAndOutputSize, yGrad, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_y.bin", inputAndOutputSize, y, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_mask.bin", inputMaskSize, mask, inputMaskSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);
    ICPU_RUN_KF(scaled_masked_softmax_grad_v2, coreNum, yGrad, y, mask, xGrad, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree(yGrad);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scaled_masked_softmax_grad_v2_test, test_tiling_key_1001_and_maskmode0)
{
    uint64_t B = 4;
    uint64_t N = 8;
    uint64_t S = 32;
    uint64_t D = 256;
    uint64_t coreNum = 40;
    uint64_t totalLine = B * N * S;
    size_t inputAndOutputSize = totalLine * D * sizeof(half);
    size_t inputMaskSize = totalLine * D * sizeof(bool);
    size_t tilingDataSize = sizeof(ScaledMaskedSoftmaxGradV2TilingData);

    uint8_t* yGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskSize);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    ScaledMaskedSoftmaxGradV2TilingData* tilingDatafromBin =
        reinterpret_cast<ScaledMaskedSoftmaxGradV2TilingData*>(tiling);

    ICPU_SET_TILING_KEY(1001);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_grad_v2/tests/ut/op_kernel/scaled_masked_softmax_grad_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_grad_v2_data/");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_data.py 4 8 32 256 4 8 half");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_tiling.py test_tiling_key_1001_and_maskmode0");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(
        path + "/scaled_masked_softmax_grad_v2_data/input_y_grad.bin", inputAndOutputSize, yGrad, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_y.bin", inputAndOutputSize, y, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_mask.bin", inputMaskSize, mask, inputMaskSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);
    ICPU_RUN_KF(scaled_masked_softmax_grad_v2, coreNum, yGrad, y, mask, xGrad, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree(yGrad);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scaled_masked_softmax_grad_v2_test, test_tiling_key_1002_and_maskmode1)
{
    uint64_t B = 8;
    uint64_t N = 16;
    uint64_t S = 64;
    uint64_t D = 256;
    uint64_t coreNum = 40;
    uint64_t totalLine = B * N * S;
    size_t inputAndOutputSize = totalLine * D * sizeof(float);
    size_t inputMaskSize = N * S * D * sizeof(bool);
    size_t tilingDataSize = sizeof(ScaledMaskedSoftmaxGradV2TilingData);

    uint8_t* yGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskSize);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    ScaledMaskedSoftmaxGradV2TilingData* tilingDatafromBin =
        reinterpret_cast<ScaledMaskedSoftmaxGradV2TilingData*>(tiling);

    ICPU_SET_TILING_KEY(1002);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_grad_v2/tests/ut/op_kernel/scaled_masked_softmax_grad_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_grad_v2_data/");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_data.py 8 16 64 256 1 16 float");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_tiling.py test_tiling_key_1002_and_maskmode1");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(
        path + "/scaled_masked_softmax_grad_v2_data/input_y_grad.bin", inputAndOutputSize, yGrad, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_y.bin", inputAndOutputSize, y, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_mask.bin", inputMaskSize, mask, inputMaskSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);
    ICPU_RUN_KF(scaled_masked_softmax_grad_v2, coreNum, yGrad, y, mask, xGrad, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree(yGrad);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(scaled_masked_softmax_grad_v2_test, test_tiling_key_1002_and_not_align)
{
    uint64_t B = 4;
    uint64_t N = 8;
    uint64_t S = 32;
    uint64_t D = 257;
    uint64_t coreNum = 40;
    uint64_t totalLine = B * N * S;
    size_t inputAndOutputSize = totalLine * D * sizeof(float);
    size_t inputMaskSize = totalLine * D * sizeof(bool);
    size_t tilingDataSize = sizeof(ScaledMaskedSoftmaxGradV2TilingData);

    uint8_t* yGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(inputMaskSize);
    uint8_t* xGrad = (uint8_t*)AscendC::GmAlloc(inputAndOutputSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    ScaledMaskedSoftmaxGradV2TilingData* tilingDatafromBin =
        reinterpret_cast<ScaledMaskedSoftmaxGradV2TilingData*>(tiling);

    ICPU_SET_TILING_KEY(1002);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_grad_v2/tests/ut/op_kernel/scaled_masked_softmax_grad_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_grad_v2_data/");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_data.py 4 8 32 257 4 8 float");
    system("cd ./scaled_masked_softmax_grad_v2_data/ && python3 gen_tiling.py test_tiling_key_1002_and_not_align");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(
        path + "/scaled_masked_softmax_grad_v2_data/input_y_grad.bin", inputAndOutputSize, yGrad, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_y.bin", inputAndOutputSize, y, inputAndOutputSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/input_mask.bin", inputMaskSize, mask, inputMaskSize);
    ReadFile(path + "/scaled_masked_softmax_grad_v2_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);
    ICPU_RUN_KF(scaled_masked_softmax_grad_v2, coreNum, yGrad, y, mask, xGrad, workspace, (uint8_t*)tilingDatafromBin);

    AscendC::GmFree(yGrad);
    AscendC::GmFree(y);
    AscendC::GmFree(mask);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
