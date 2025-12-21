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
 * \file test_multi_add_rms_norm_dynamic_quant.cpp
 * \brief
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "tensor_list_operate.h"
#include <cstdint>
#include "multi_add_rms_norm_dynamic_quant_tiling_def.h"

using namespace std;
constexpr uint32_t SLICE_COL_LEN = 8864;

extern "C" void multi_add_rms_norm_dynamic_quant(
    uint8_t* x1, uint8_t* x2, uint8_t* gamma, uint8_t* scales1, uint8_t* scales2, uint8_t* y1, uint8_t* y2, uint8_t* x,
    uint8_t* y, uint8_t* outScale1, uint8_t* outScale2, uint8_t* workspace, uint8_t* tiling);

class multi_add_rms_norm_dynamic_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "multi_add_rms_norm_dynamic_quant_test SetUp\n" << endl;
        system(
            "cp -rf "
            "../../../../norm/multi_add_rms_norm_dynamic_quant/tests/ut/op_kernel/multi_add_rms_norm_dynamic_quant_data ./");
    }
    static void TearDownTestCase()
    {
        cout << "multi_add_rms_norm_dynamic_quant_test TearDown\n" << endl;
    }
};

TEST_F(multi_add_rms_norm_dynamic_quant_test, test_case_001_5add_fp16_normal)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    system("chmod -R 755 ./multi_add_rms_norm_dynamic_quant_data/");
    system("cd ./multi_add_rms_norm_dynamic_quant_data/ && python3 gen_data.py '{128, 64}' 5 'float16'");
    int N = 128;
    int D = 64;
    size_t rowsByteSize = N * D * sizeof(half);
    size_t weightBetaByteSize = D * sizeof(half);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t reducedByteSize = N * sizeof(float);
    size_t tilingDataSize = sizeof(MultiAddRmsNormDynamicQuantTilingData);

    std::vector<std::vector<uint64_t>> shapeInfos = {{N * D}, {N * D}, {N * D}, {N * D}, {N * D}};
    uint8_t* x1 = CreateTensorList<int16_t>(shapeInfos, "float16", 0, "x1");
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth1 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth2 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 1);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;

    std::string path = ".";

    MultiAddRmsNormDynamicQuantTilingData* tilingDatafromBin =
        reinterpret_cast<MultiAddRmsNormDynamicQuantTilingData*>(tiling);

    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/x2.bin", rowsByteSize, x2, rowsByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/gamma.bin", weightBetaByteSize, gamma, weightBetaByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/smooth1.bin", weightBetaByteSize, smooth1, weightBetaByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/smooth2.bin", weightBetaByteSize, smooth2, weightBetaByteSize);

    tilingDatafromBin->useCore = blockDim;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numLastDimAligned = (D + 32 - 1) / 32 * 32;
    tilingDatafromBin->firstDimPerCore = 1;
    tilingDatafromBin->firstDimPerCoreTail = 1;
    tilingDatafromBin->firstDimPerLoop = 1;
    tilingDatafromBin->lastDimLoopNum = 1;
    tilingDatafromBin->lastDimSliceLen = 8864;
    tilingDatafromBin->lastDimSliceLenTail = D;
    tilingDatafromBin->smoothNum = 2;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = (1.0 / D);
    tilingDatafromBin->x1Num = 3;

    // dual normal fp16
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        multi_add_rms_norm_dynamic_quant, blockDim, x1, x2, gamma, smooth1, smooth2, y1, y2, x, y, outScale1, outScale2,
        workspace, (uint8_t*)(tilingDatafromBin));


    FreeTensorList<int16_t>(x1, shapeInfos, "float16", 0, "x1");
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(smooth1);
    AscendC::GmFree(smooth2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

}

TEST_F(multi_add_rms_norm_dynamic_quant_test, test_case_002_3add_bf16_normal)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    system("chmod -R 755 ./multi_add_rms_norm_dynamic_quant_data/");
    system("cd ./multi_add_rms_norm_dynamic_quant_data/ && python3 gen_data.py '{128, 64}' 3 'bfloat16_t'");
    int N = 128;
    int D = 64;
    size_t rowsByteSize = N * D * sizeof(bfloat16_t);
    size_t weightBetaByteSize = D * sizeof(bfloat16_t);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t reducedByteSize = N * sizeof(float);
    size_t tilingDataSize = sizeof(MultiAddRmsNormDynamicQuantTilingData);

    std::vector<std::vector<uint64_t>> shapeInfos = {{N * D}, {N * D}, {N * D}};
    uint8_t* x1 = CreateTensorList<int16_t>(shapeInfos, "bfloat16_t", 0, "x1");
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth1 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth2 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 1);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;

    std::string path = ".";

    MultiAddRmsNormDynamicQuantTilingData* tilingDatafromBin =
        reinterpret_cast<MultiAddRmsNormDynamicQuantTilingData*>(tiling);

    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/x2.bin", rowsByteSize, x2, rowsByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/gamma.bin", weightBetaByteSize, gamma, weightBetaByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/smooth1.bin", weightBetaByteSize, smooth1, weightBetaByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/smooth2.bin", weightBetaByteSize, smooth2, weightBetaByteSize);

  tilingDatafromBin->useCore = blockDim;
  tilingDatafromBin->numFirstDim = N;
  tilingDatafromBin->numLastDim = D;
  tilingDatafromBin->numLastDimAligned = (D + 32 - 1) / 32 * 32;
  tilingDatafromBin->firstDimPerCore = 1;
  tilingDatafromBin->firstDimPerCoreTail = 1;
  tilingDatafromBin->firstDimPerLoop = 1;
  tilingDatafromBin->lastDimLoopNum = 1;
  tilingDatafromBin->lastDimSliceLen = SLICE_COL_LEN;
  tilingDatafromBin->lastDimSliceLenTail = D;
  tilingDatafromBin->smoothNum = 2;
  tilingDatafromBin->epsilon = 1e-5;
  tilingDatafromBin->avgFactor = (1.0 / D);
  tilingDatafromBin->x1Num = 3;

    // dual normal bf16
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        multi_add_rms_norm_dynamic_quant, blockDim, x1, x2, gamma, smooth1, smooth2, y1, y2, x, y, outScale1, outScale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    FreeTensorList<int16_t>(x1, shapeInfos, "bfloat16_t", 0, "x1");
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(smooth1);
    AscendC::GmFree(smooth2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

}

TEST_F(multi_add_rms_norm_dynamic_quant_test, test_case_003_4add_fp16_single_row)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    system("chmod -R 755 ./multi_add_rms_norm_dynamic_quant_data/");
    system("cd ./multi_add_rms_norm_dynamic_quant_data/ && python3 gen_data.py '{128, 64}' 1 'float16'");
    int N = 128;
    int D = 64;
    size_t rowsByteSize = N * D * sizeof(half);
    size_t weightBetaByteSize = D * sizeof(half);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t reducedByteSize = N * sizeof(float);
    size_t tilingDataSize = sizeof(MultiAddRmsNormDynamicQuantTilingData);

    std::vector<std::vector<uint64_t>> shapeInfos = {{N * D}, {N * D}, {N * D}, {N * D}};
    uint8_t* x1 = CreateTensorList<int16_t>(shapeInfos, "float16", 0, "x1");
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth1 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth2 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 1);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;

    std::string path = ".";

    MultiAddRmsNormDynamicQuantTilingData* tilingDatafromBin =
        reinterpret_cast<MultiAddRmsNormDynamicQuantTilingData*>(tiling);

    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/x2.bin", rowsByteSize, x2, rowsByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/gamma.bin", weightBetaByteSize, gamma, weightBetaByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/smooth1.bin", weightBetaByteSize, smooth1, weightBetaByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/smooth2.bin", weightBetaByteSize, smooth2, weightBetaByteSize);

    tilingDatafromBin->useCore = blockDim;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numLastDimAligned = (D + 32 - 1) / 32 * 32;
    tilingDatafromBin->firstDimPerCore = 1;
    tilingDatafromBin->firstDimPerCoreTail = 1;
    tilingDatafromBin->firstDimPerLoop = 1;
    tilingDatafromBin->lastDimLoopNum = 1;
    tilingDatafromBin->lastDimSliceLen = 8864;
    tilingDatafromBin->lastDimSliceLenTail = D;
    tilingDatafromBin->smoothNum = 2;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = (1.0 / D);
    tilingDatafromBin->x1Num = 3;

    // dual normal fp16
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        multi_add_rms_norm_dynamic_quant, blockDim, x1, x2, gamma, smooth1, smooth2, y1, y2, x, y, outScale1, outScale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    FreeTensorList<int16_t>(x1, shapeInfos, "float16", 0, "x1");
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(smooth1);
    AscendC::GmFree(smooth2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

}

TEST_F(multi_add_rms_norm_dynamic_quant_test, test_case_004_4add_bf16_single_row)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    system("chmod -R 755 ./multi_add_rms_norm_dynamic_quant_data/");
    system("cd ./multi_add_rms_norm_dynamic_quant_data/ && python3 gen_data.py '{128, 64}' 4 'bfloat16_t'");
    int N = 128;
    int D = 64;
    size_t rowsByteSize = N * D * sizeof(bfloat16_t);
    size_t weightBetaByteSize = D * sizeof(bfloat16_t);
    size_t outQuantByteSize = N * D * sizeof(int8_t);
    size_t reducedByteSize = N * sizeof(float);
    size_t tilingDataSize = sizeof(MultiAddRmsNormDynamicQuantTilingData);

    std::vector<std::vector<uint64_t>> shapeInfos = {{N * D}, {N * D}, {N * D}, {N * D}};
    uint8_t* x1 = CreateTensorList<int16_t>(shapeInfos, "bfloat16_t", 0, "x1");
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth1 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);
    uint8_t* smooth2 = (uint8_t*)AscendC::GmAlloc(weightBetaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outQuantByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(rowsByteSize);
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(reducedByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 1);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;

    std::string path = ".";

    MultiAddRmsNormDynamicQuantTilingData* tilingDatafromBin =
        reinterpret_cast<MultiAddRmsNormDynamicQuantTilingData*>(tiling);

    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/x2.bin", rowsByteSize, x2, rowsByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/gamma.bin", weightBetaByteSize, gamma, weightBetaByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/smooth1.bin", weightBetaByteSize, smooth1, weightBetaByteSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/smooth2.bin", weightBetaByteSize, smooth2, weightBetaByteSize);

    tilingDatafromBin->useCore = blockDim;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numLastDimAligned = (D + 32 - 1) / 32 * 32;
    tilingDatafromBin->firstDimPerCore = 1;
    tilingDatafromBin->firstDimPerCoreTail = 1;
    tilingDatafromBin->firstDimPerLoop = 1;
    tilingDatafromBin->lastDimLoopNum = 1;
    tilingDatafromBin->lastDimSliceLen = 8864;
    tilingDatafromBin->lastDimSliceLenTail = D;
    tilingDatafromBin->smoothNum = 2;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = (1.0 / D);
    tilingDatafromBin->x1Num = 3;

    // dual normal bf16
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        multi_add_rms_norm_dynamic_quant, blockDim, x1, x2, gamma, smooth1, smooth2, y1, y2, x, y, outScale1, outScale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    FreeTensorList<int16_t>(x1, shapeInfos, "bfloat16_t", 0, "x1");
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(smooth1);
    AscendC::GmFree(smooth2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

}


TEST_F(multi_add_rms_norm_dynamic_quant_test, test_case_005_5add_fp16_cut_d)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    system("chmod -R 755 ./multi_add_rms_norm_dynamic_quant_data/");
    system("cd ./multi_add_rms_norm_dynamic_quant_data/ && python3 gen_data.py '{230, 31200}' 5 'float16'");
    int N = 230;  // Tensor rows
    int D = 31200;  // columns must be times of full-loaded single row, here we take a 4-cut case 
    size_t xShape = N * D;
    size_t xSize = N * D * sizeof(half);
    size_t gammaSize = N * D * sizeof(int8_t);
    size_t tilingDataSize = sizeof(MultiAddRmsNormDynamicQuantTilingData);

    std::vector<std::vector<uint64_t>> shapeInfos = {{xShape}, {xShape}, {xShape}, {xShape}, {xShape}};
    uint8_t* x1 = CreateTensorList<int16_t>(shapeInfos, "float16", 0, "x1");
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(int8_t));
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(int8_t));
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(half));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(half));
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(float));
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(float));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 1);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 22;

    std::string path = ".";

    MultiAddRmsNormDynamicQuantTilingData* tilingDatafromBin =
        reinterpret_cast<MultiAddRmsNormDynamicQuantTilingData*>(tiling);

    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/x2.bin", xSize, x2, xSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/gamma.bin", gammaSize, gamma, gammaSize);

    tilingDatafromBin->useCore = blockDim;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numLastDimAligned = (D + 32 - 1) / 32 * 32;
    // Tiling Policy Specialized
    tilingDatafromBin->firstDimPerCore = 5;
    tilingDatafromBin->firstDimPerCoreTail = 5;
    tilingDatafromBin->firstDimPerLoop = 1;
    tilingDatafromBin->lastDimLoopNum = 3;
    tilingDatafromBin->lastDimSliceLen = SLICE_COL_LEN;
    tilingDatafromBin->lastDimSliceLenTail = 4608;
    tilingDatafromBin->smoothNum = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = (1.0 / D);
    tilingDatafromBin->x1Num = 5;

    // dual normal fp16
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(
        multi_add_rms_norm_dynamic_quant, blockDim, x1, x2, gamma, nullptr, nullptr, y1, y2, x, y, outScale1, outScale2,
        workspace, (uint8_t*)(tilingDatafromBin));

    FreeTensorList<int16_t>(x1, shapeInfos, "float16", 0, "x1");
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(multi_add_rms_norm_dynamic_quant_test, test_case_006_2add_bf16_cut_d)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    system("chmod -R 755 ./multi_add_rms_norm_dynamic_quant_data/");
    system("cd ./multi_add_rms_norm_dynamic_quant_data/ && python3 gen_data.py '{201, 17459}' 2 'bfloat16_t'");
    int N = 201;  // Tensor rows
    int D = 17459;  // columns must be times of full-loaded single row, here we take a 4-cut case 
    size_t xShape = N * D;
    size_t xSize = N * D * sizeof(bfloat16_t);
    size_t gammaSize = D * sizeof(bfloat16_t);
    size_t smoothSize = D * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(MultiAddRmsNormDynamicQuantTilingData);

    std::vector<std::vector<uint64_t>> shapeInfos = {{xShape}, {xShape}};
    uint8_t* x1 = CreateTensorList<int16_t>(shapeInfos, "bfloat16_t", 0, "x1");
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* smooth1 = (uint8_t*)AscendC::GmAlloc(smoothSize);
    uint8_t* smooth2 = (uint8_t*)AscendC::GmAlloc(smoothSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(int8_t));
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(int8_t));
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(bfloat16_t));
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(bfloat16_t));
    uint8_t* outScale1 = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(float));
    uint8_t* outScale2 = (uint8_t*)AscendC::GmAlloc(xShape * sizeof(float));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 1);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 41;

    std::string path = ".";

    MultiAddRmsNormDynamicQuantTilingData* tilingDatafromBin =
        reinterpret_cast<MultiAddRmsNormDynamicQuantTilingData*>(tiling);

    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/x2.bin", xSize, x2, xSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/gamma.bin", gammaSize, gamma, gammaSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/smooth1.bin", smoothSize, smooth1, smoothSize);
    ReadFile(path + "/multi_add_rms_norm_dynamic_quant_data/smooth2.bin", smoothSize, smooth2, smoothSize);

    tilingDatafromBin->useCore = blockDim;
    tilingDatafromBin->numFirstDim = N;
    tilingDatafromBin->numLastDim = D;
    tilingDatafromBin->numLastDimAligned = (D + 32 - 1) / 32 * 32;
    // Tiling Policy Specialized
    tilingDatafromBin->firstDimPerCore = 5;
    tilingDatafromBin->firstDimPerCoreTail = 1;
    tilingDatafromBin->firstDimPerLoop = 1;
    tilingDatafromBin->lastDimLoopNum = 1;
    tilingDatafromBin->lastDimSliceLen = SLICE_COL_LEN;
    tilingDatafromBin->lastDimSliceLenTail = 8595;
    tilingDatafromBin->smoothNum = 2;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->avgFactor = (1.0 / D);
    tilingDatafromBin->x1Num = 2;

    // dual normal bf16
    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(
        multi_add_rms_norm_dynamic_quant, blockDim, x1, x2, gamma, smooth1, smooth2, y1, y2, x, y, outScale1, outScale2,
        workspace, (uint8_t*)(tilingDatafromBin));


    FreeTensorList<int16_t>(x1, shapeInfos, "bfloat16_t", 0, "x1");
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(smooth1);
    AscendC::GmFree(smooth2);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(outScale1);
    AscendC::GmFree(outScale2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}