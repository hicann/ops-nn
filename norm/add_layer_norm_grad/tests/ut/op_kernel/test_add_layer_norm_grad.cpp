/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_utils.h"
#include "add_layer_norm_grad_tiling_def.h"
#include "tikicpulib.h"
#include "gtest/gtest.h"
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

extern "C" void add_layer_norm_grad(
    uint8_t* dy, uint8_t* x1, uint8_t* x2, uint8_t* rstd, uint8_t* mean, uint8_t* gamma, uint8_t* dsum, uint8_t* dx,
    uint8_t* dgamma, uint8_t* dbeta, uint8_t* workspace, uint8_t* tiling);

class add_layer_norm_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_layer_norm_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_layer_norm_grad_test TearDown\n" << endl;
    }
};

TEST_F(add_layer_norm_grad_test, test_case_11)
{
    size_t dyByteSize = 3 * 133 * sizeof(float);
    size_t x1ByteSize = 3 * 133 * sizeof(float);
    size_t rstdByteSize = 3 * sizeof(float);
    size_t meanByteSize = 3 * sizeof(float);
    size_t gammaByteSize = 133 * sizeof(float);
    size_t dsumByteSize = 3 * 133 * sizeof(float);
    size_t dxByteSize = 3 * 133 * sizeof(float);
    size_t dgammaByteSize = 133 * sizeof(float);
    size_t dbetaByteSize = 133 * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* dsum = (uint8_t*)AscendC::GmAlloc(dsumByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormGradTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormGradTilingData*>(tiling);

    tilingDatafromBin->numCore = 3;
    tilingDatafromBin->numLastDim = 133;
    tilingDatafromBin->numFirstDim = 3;
    tilingDatafromBin->nInOneCoreLength = 1;
    tilingDatafromBin->nInOneCoreLengthTail = 1;
    tilingDatafromBin->ndInOneCoreLength = 133;
    tilingDatafromBin->nAvailInUb = 85;
    tilingDatafromBin->dInnerLength = 133;
    tilingDatafromBin->dInnerLengthTail = 133;
    tilingDatafromBin->dOuterLength = 1;
    tilingDatafromBin->nInOneCoreNorm = 1;
    tilingDatafromBin->gmOneCoreElemXYNorm = 133;
    tilingDatafromBin->nAvailInUbNorm = 1;
    tilingDatafromBin->nMiddleCountNorm = 1;
    tilingDatafromBin->ndRoundUpDtypeNorm = 544;
    tilingDatafromBin->n1RoundUpFloatNorm = 32;
    tilingDatafromBin->nInUbTotalNormTail = 1;
    tilingDatafromBin->nInOneCoreTail = 1;
    tilingDatafromBin->gmOneCoreElemXYTail = 133;
    tilingDatafromBin->nAvailInUbTail = 1;
    tilingDatafromBin->nMiddleCountTail = 1;
    tilingDatafromBin->ndRoundUpDtypeTail = 544;
    tilingDatafromBin->n1RoundUpFloatTail = 32;
    tilingDatafromBin->nInUbTotalTailTail = 1;
    tilingDatafromBin->dyPadRight = 3;
    tilingDatafromBin->rstdPadRight = 7;
    tilingDatafromBin->roundUpNumLastDim = 136;
    tilingDatafromBin->roundUpNumLastDimDtype = 544;
    tilingDatafromBin->roundUp1Dtype = 8;
    tilingDatafromBin->roundUpNumLastDimFloat = 544;
    uint32_t blockDim = tilingDatafromBin->numCore;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(
        add_layer_norm_grad, blockDim, dy, x1, x2, rstd, mean, gamma, dsum, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dsum);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_grad_test, test_case_10)
{
    size_t dyByteSize = 3 * 133 * sizeof(float);
    size_t x1ByteSize = 3 * 133 * sizeof(float);
    size_t rstdByteSize = 3 * sizeof(float);
    size_t meanByteSize = 3 * sizeof(float);
    size_t gammaByteSize = 133 * sizeof(float);
    size_t dsumByteSize = 3 * 133 * sizeof(float);
    size_t dxByteSize = 3 * 133 * sizeof(float);
    size_t dgammaByteSize = 133 * sizeof(float);
    size_t dbetaByteSize = 133 * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* dsum = (uint8_t*)AscendC::GmAlloc(dsumByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormGradTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormGradTilingData*>(tiling);

    tilingDatafromBin->numCore = 3;
    tilingDatafromBin->numLastDim = 133;
    tilingDatafromBin->numFirstDim = 3;
    tilingDatafromBin->nInOneCoreLength = 1;
    tilingDatafromBin->nInOneCoreLengthTail = 1;
    tilingDatafromBin->ndInOneCoreLength = 133;
    tilingDatafromBin->nAvailInUb = 85;
    tilingDatafromBin->dInnerLength = 133;
    tilingDatafromBin->dInnerLengthTail = 133;
    tilingDatafromBin->dOuterLength = 1;
    tilingDatafromBin->nInOneCoreNorm = 1;
    tilingDatafromBin->gmOneCoreElemXYNorm = 133;
    tilingDatafromBin->nAvailInUbNorm = 1;
    tilingDatafromBin->nMiddleCountNorm = 1;
    tilingDatafromBin->ndRoundUpDtypeNorm = 544;
    tilingDatafromBin->n1RoundUpFloatNorm = 32;
    tilingDatafromBin->nInUbTotalNormTail = 1;
    tilingDatafromBin->nInOneCoreTail = 1;
    tilingDatafromBin->gmOneCoreElemXYTail = 133;
    tilingDatafromBin->nAvailInUbTail = 1;
    tilingDatafromBin->nMiddleCountTail = 1;
    tilingDatafromBin->ndRoundUpDtypeTail = 544;
    tilingDatafromBin->n1RoundUpFloatTail = 32;
    tilingDatafromBin->nInUbTotalTailTail = 1;
    tilingDatafromBin->dyPadRight = 3;
    tilingDatafromBin->rstdPadRight = 7;
    tilingDatafromBin->roundUpNumLastDim = 136;
    tilingDatafromBin->roundUpNumLastDimDtype = 544;
    tilingDatafromBin->roundUp1Dtype = 8;
    tilingDatafromBin->roundUpNumLastDimFloat = 544;
    uint32_t blockDim = tilingDatafromBin->numCore;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(
        add_layer_norm_grad, blockDim, dy, x1, x2, rstd, mean, gamma, dsum, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dsum);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_grad_test, test_case_21)
{
    size_t dyByteSize = 2 * 291 * sizeof(int16_t);
    size_t x1ByteSize = 2 * 291 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t meanByteSize = 2 * sizeof(float);
    size_t gammaByteSize = 291 * sizeof(int16_t);
    size_t dsumByteSize = 2 * 291 * sizeof(int16_t);
    size_t dxByteSize = 2 * 291 * sizeof(int16_t);
    size_t dgammaByteSize = 291 * sizeof(float);
    size_t dbetaByteSize = 291 * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* dsum = (uint8_t*)AscendC::GmAlloc(dsumByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1600);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormGradTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormGradTilingData*>(tiling);

    tilingDatafromBin->numCore = 2;
    tilingDatafromBin->numLastDim = 291;
    tilingDatafromBin->numFirstDim = 2;
    tilingDatafromBin->nInOneCoreLength = 1;
    tilingDatafromBin->nInOneCoreLengthTail = 1;
    tilingDatafromBin->ndInOneCoreLength = 291;
    tilingDatafromBin->nAvailInUb = 21;
    tilingDatafromBin->dInnerLength = 291;
    tilingDatafromBin->dInnerLengthTail = 291;
    tilingDatafromBin->dOuterLength = 1;
    tilingDatafromBin->nInOneCoreNorm = 1;
    tilingDatafromBin->gmOneCoreElemXYNorm = 291;
    tilingDatafromBin->nAvailInUbNorm = 1;
    tilingDatafromBin->nMiddleCountNorm = 1;
    tilingDatafromBin->ndRoundUpDtypeNorm = 608;
    tilingDatafromBin->n1RoundUpFloatNorm = 32;
    tilingDatafromBin->nInUbTotalNormTail = 1;
    tilingDatafromBin->nInOneCoreTail = 1;
    tilingDatafromBin->gmOneCoreElemXYTail = 291;
    tilingDatafromBin->nAvailInUbTail = 1;
    tilingDatafromBin->nMiddleCountTail = 1;
    tilingDatafromBin->ndRoundUpDtypeTail = 608;
    tilingDatafromBin->n1RoundUpFloatTail = 32;
    tilingDatafromBin->nInUbTotalTailTail = 1;
    tilingDatafromBin->dyPadRight = 13;
    tilingDatafromBin->rstdPadRight = 7;
    tilingDatafromBin->roundUpNumLastDim = 304;
    tilingDatafromBin->roundUpNumLastDimDtype = 608;
    tilingDatafromBin->roundUp1Dtype = 8;
    tilingDatafromBin->roundUpNumLastDimFloat = 1184;
    uint32_t blockDim = tilingDatafromBin->numCore;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(21);
    ICPU_RUN_KF(
        add_layer_norm_grad, blockDim, dy, x1, x2, rstd, mean, gamma, dsum, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dsum);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_grad_test, test_case_20)
{
    size_t dyByteSize = 2 * 291 * sizeof(int16_t);
    size_t x1ByteSize = 2 * 291 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t meanByteSize = 2 * sizeof(float);
    size_t gammaByteSize = 291 * sizeof(int16_t);
    size_t dsumByteSize = 2 * 291 * sizeof(int16_t);
    size_t dxByteSize = 2 * 291 * sizeof(int16_t);
    size_t dgammaByteSize = 291 * sizeof(float);
    size_t dbetaByteSize = 291 * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* dsum = (uint8_t*)AscendC::GmAlloc(dsumByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1600);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormGradTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormGradTilingData*>(tiling);

    tilingDatafromBin->numCore = 2;
    tilingDatafromBin->numLastDim = 291;
    tilingDatafromBin->numFirstDim = 2;
    tilingDatafromBin->nInOneCoreLength = 1;
    tilingDatafromBin->nInOneCoreLengthTail = 1;
    tilingDatafromBin->ndInOneCoreLength = 291;
    tilingDatafromBin->nAvailInUb = 26;
    tilingDatafromBin->dInnerLength = 291;
    tilingDatafromBin->dInnerLengthTail = 291;
    tilingDatafromBin->dOuterLength = 1;
    tilingDatafromBin->nInOneCoreNorm = 1;
    tilingDatafromBin->gmOneCoreElemXYNorm = 291;
    tilingDatafromBin->nAvailInUbNorm = 1;
    tilingDatafromBin->nMiddleCountNorm = 1;
    tilingDatafromBin->ndRoundUpDtypeNorm = 608;
    tilingDatafromBin->n1RoundUpFloatNorm = 32;
    tilingDatafromBin->nInUbTotalNormTail = 1;
    tilingDatafromBin->nInOneCoreTail = 1;
    tilingDatafromBin->gmOneCoreElemXYTail = 291;
    tilingDatafromBin->nAvailInUbTail = 1;
    tilingDatafromBin->nMiddleCountTail = 1;
    tilingDatafromBin->ndRoundUpDtypeTail = 608;
    tilingDatafromBin->n1RoundUpFloatTail = 32;
    tilingDatafromBin->nInUbTotalTailTail = 1;
    tilingDatafromBin->dyPadRight = 13;
    tilingDatafromBin->rstdPadRight = 7;
    tilingDatafromBin->roundUpNumLastDim = 304;
    tilingDatafromBin->roundUpNumLastDimDtype = 608;
    tilingDatafromBin->roundUp1Dtype = 8;
    tilingDatafromBin->roundUpNumLastDimFloat = 1184;
    uint32_t blockDim = tilingDatafromBin->numCore;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(20);
    ICPU_RUN_KF(
        add_layer_norm_grad, blockDim, dy, x1, x2, rstd, mean, gamma, dsum, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dsum);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_grad_test, test_case_31)
{
    size_t dyByteSize = 2 * 291 * sizeof(int16_t);
    size_t x1ByteSize = 2 * 291 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t meanByteSize = 2 * sizeof(float);
    size_t gammaByteSize = 291 * sizeof(int16_t);
    size_t dsumByteSize = 2 * 291 * sizeof(int16_t);
    size_t dxByteSize = 2 * 291 * sizeof(int16_t);
    size_t dgammaByteSize = 291 * sizeof(float);
    size_t dbetaByteSize = 291 * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* dsum = (uint8_t*)AscendC::GmAlloc(dsumByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1600);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormGradTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormGradTilingData*>(tiling);

    tilingDatafromBin->numCore = 2;
    tilingDatafromBin->numLastDim = 291;
    tilingDatafromBin->numFirstDim = 2;
    tilingDatafromBin->nInOneCoreLength = 1;
    tilingDatafromBin->nInOneCoreLengthTail = 1;
    tilingDatafromBin->ndInOneCoreLength = 291;
    tilingDatafromBin->nAvailInUb = 21;
    tilingDatafromBin->dInnerLength = 291;
    tilingDatafromBin->dInnerLengthTail = 291;
    tilingDatafromBin->dOuterLength = 1;
    tilingDatafromBin->nInOneCoreNorm = 1;
    tilingDatafromBin->gmOneCoreElemXYNorm = 291;
    tilingDatafromBin->nAvailInUbNorm = 1;
    tilingDatafromBin->nMiddleCountNorm = 1;
    tilingDatafromBin->ndRoundUpDtypeNorm = 608;
    tilingDatafromBin->n1RoundUpFloatNorm = 32;
    tilingDatafromBin->nInUbTotalNormTail = 1;
    tilingDatafromBin->nInOneCoreTail = 1;
    tilingDatafromBin->gmOneCoreElemXYTail = 291;
    tilingDatafromBin->nAvailInUbTail = 1;
    tilingDatafromBin->nMiddleCountTail = 1;
    tilingDatafromBin->ndRoundUpDtypeTail = 608;
    tilingDatafromBin->n1RoundUpFloatTail = 32;
    tilingDatafromBin->nInUbTotalTailTail = 1;
    tilingDatafromBin->dyPadRight = 13;
    tilingDatafromBin->rstdPadRight = 7;
    tilingDatafromBin->roundUpNumLastDim = 304;
    tilingDatafromBin->roundUpNumLastDimDtype = 608;
    tilingDatafromBin->roundUp1Dtype = 8;
    tilingDatafromBin->roundUpNumLastDimFloat = 1184;
    uint32_t blockDim = tilingDatafromBin->numCore;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(31);
    ICPU_RUN_KF(
        add_layer_norm_grad, blockDim, dy, x1, x2, rstd, mean, gamma, dsum, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dsum);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_grad_test, test_case_30)
{
    size_t dyByteSize = 2 * 291 * sizeof(int16_t);
    size_t x1ByteSize = 2 * 291 * sizeof(int16_t);
    size_t rstdByteSize = 2 * sizeof(float);
    size_t meanByteSize = 2 * sizeof(float);
    size_t gammaByteSize = 291 * sizeof(int16_t);
    size_t dsumByteSize = 2 * 291 * sizeof(int16_t);
    size_t dxByteSize = 2 * 291 * sizeof(int16_t);
    size_t dgammaByteSize = 291 * sizeof(float);
    size_t dbetaByteSize = 291 * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* dsum = (uint8_t*)AscendC::GmAlloc(dsumByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1600);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormGradTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormGradTilingData*>(tiling);

    tilingDatafromBin->numCore = 2;
    tilingDatafromBin->numLastDim = 291;
    tilingDatafromBin->numFirstDim = 2;
    tilingDatafromBin->nInOneCoreLength = 1;
    tilingDatafromBin->nInOneCoreLengthTail = 1;
    tilingDatafromBin->ndInOneCoreLength = 291;
    tilingDatafromBin->nAvailInUb = 39;
    tilingDatafromBin->dInnerLength = 291;
    tilingDatafromBin->dInnerLengthTail = 291;
    tilingDatafromBin->dOuterLength = 1;
    tilingDatafromBin->nInOneCoreNorm = 1;
    tilingDatafromBin->gmOneCoreElemXYNorm = 291;
    tilingDatafromBin->nAvailInUbNorm = 1;
    tilingDatafromBin->nMiddleCountNorm = 1;
    tilingDatafromBin->ndRoundUpDtypeNorm = 608;
    tilingDatafromBin->n1RoundUpFloatNorm = 32;
    tilingDatafromBin->nInUbTotalNormTail = 1;
    tilingDatafromBin->nInOneCoreTail = 1;
    tilingDatafromBin->gmOneCoreElemXYTail = 291;
    tilingDatafromBin->nAvailInUbTail = 1;
    tilingDatafromBin->nMiddleCountTail = 1;
    tilingDatafromBin->ndRoundUpDtypeTail = 608;
    tilingDatafromBin->n1RoundUpFloatTail = 32;
    tilingDatafromBin->nInUbTotalTailTail = 1;
    tilingDatafromBin->dyPadRight = 13;
    tilingDatafromBin->rstdPadRight = 7;
    tilingDatafromBin->roundUpNumLastDim = 304;
    tilingDatafromBin->roundUpNumLastDimDtype = 608;
    tilingDatafromBin->roundUp1Dtype = 8;
    tilingDatafromBin->roundUpNumLastDimFloat = 1184;
    uint32_t blockDim = tilingDatafromBin->numCore;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(30);
    ICPU_RUN_KF(
        add_layer_norm_grad, blockDim, dy, x1, x2, rstd, mean, gamma, dsum, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dsum);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_grad_test, test_case_41)
{
    size_t dyByteSize = 4 * 7009 * sizeof(float);
    size_t x1ByteSize = 4 * 7009 * sizeof(float);
    size_t rstdByteSize = 4 * sizeof(float);
    size_t meanByteSize = 4 * sizeof(float);
    size_t gammaByteSize = 7009 * sizeof(float);
    size_t dsumByteSize = 4 * 7009 * sizeof(float);
    size_t dxByteSize = 4 * 7009 * sizeof(float);
    size_t dgammaByteSize = 7009 * sizeof(float);
    size_t dbetaByteSize = 7009 * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* dsum = (uint8_t*)AscendC::GmAlloc(dsumByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormGradTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormGradTilingData*>(tiling);

    tilingDatafromBin->numCore = 4;
    tilingDatafromBin->numLastDim = 7009;
    tilingDatafromBin->numFirstDim = 4;
    tilingDatafromBin->nInOneCoreLength = 1;
    tilingDatafromBin->nInOneCoreLengthTail = 1;
    tilingDatafromBin->ndInOneCoreLength = 7009;
    tilingDatafromBin->nAvailInUb = 1;
    tilingDatafromBin->dInnerLength = 4864;
    tilingDatafromBin->dInnerLengthTail = 2145;
    tilingDatafromBin->dOuterLength = 2;
    tilingDatafromBin->nInOneCoreNorm = 1;
    tilingDatafromBin->gmOneCoreElemXYNorm = 7009;
    tilingDatafromBin->nAvailInUbNorm = 1;
    tilingDatafromBin->nMiddleCountNorm = 1;
    tilingDatafromBin->ndRoundUpDtypeNorm = 28064;
    tilingDatafromBin->n1RoundUpFloatNorm = 32;
    tilingDatafromBin->nInUbTotalNormTail = 1;
    tilingDatafromBin->nInOneCoreTail = 1;
    tilingDatafromBin->gmOneCoreElemXYTail = 28064;
    tilingDatafromBin->nAvailInUbTail = 1;
    tilingDatafromBin->nMiddleCountTail = 1;
    tilingDatafromBin->ndRoundUpDtypeTail = 28064;
    tilingDatafromBin->n1RoundUpFloatTail = 32;
    tilingDatafromBin->nInUbTotalTailTail = 1;
    tilingDatafromBin->dyPadRight = 7;
    tilingDatafromBin->rstdPadRight = 7;
    tilingDatafromBin->roundUpNumLastDim = 7016;
    tilingDatafromBin->roundUpNumLastDimDtype = 28064;
    tilingDatafromBin->roundUp1Dtype = 8;
    tilingDatafromBin->roundUpNumLastDimFloat = 28064;

    uint32_t blockDim = tilingDatafromBin->numCore;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(41);
    ICPU_RUN_KF(
        add_layer_norm_grad, blockDim, dy, x1, x2, rstd, mean, gamma, dsum, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dsum);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_grad_test, test_case_51)
{
    size_t dyByteSize = 4 * 7009 * sizeof(int16_t);
    size_t x1ByteSize = 4 * 7009 * sizeof(int16_t);
    size_t rstdByteSize = 4 * sizeof(float);
    size_t meanByteSize = 4 * sizeof(float);
    size_t gammaByteSize = 7009 * sizeof(int16_t);
    size_t dsumByteSize = 4 * 7009 * sizeof(int16_t);
    size_t dxByteSize = 4 * 7009 * sizeof(int16_t);
    size_t dgammaByteSize = 7009 * sizeof(float);
    size_t dbetaByteSize = 7009 * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* dsum = (uint8_t*)AscendC::GmAlloc(dsumByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormGradTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormGradTilingData*>(tiling);

    tilingDatafromBin->numCore = 4;
    tilingDatafromBin->numLastDim = 7009;
    tilingDatafromBin->numFirstDim = 4;
    tilingDatafromBin->nInOneCoreLength = 1;
    tilingDatafromBin->nInOneCoreLengthTail = 1;
    tilingDatafromBin->ndInOneCoreLength = 7009;
    tilingDatafromBin->nAvailInUb = 1;
    tilingDatafromBin->dInnerLength = 4416;
    tilingDatafromBin->dInnerLengthTail = 2593;
    tilingDatafromBin->dOuterLength = 2;
    tilingDatafromBin->nInOneCoreNorm = 1;
    tilingDatafromBin->gmOneCoreElemXYNorm = 7009;
    tilingDatafromBin->nAvailInUbNorm = 1;
    tilingDatafromBin->nMiddleCountNorm = 1;
    tilingDatafromBin->ndRoundUpDtypeNorm = 14048;
    tilingDatafromBin->n1RoundUpFloatNorm = 32;
    tilingDatafromBin->nInUbTotalNormTail = 1;
    tilingDatafromBin->nInOneCoreTail = 1;
    tilingDatafromBin->gmOneCoreElemXYTail = 7009;
    tilingDatafromBin->nAvailInUbTail = 1;
    tilingDatafromBin->nMiddleCountTail = 1;
    tilingDatafromBin->ndRoundUpDtypeTail = 14048;
    tilingDatafromBin->n1RoundUpFloatTail = 32;
    tilingDatafromBin->nInUbTotalTailTail = 1;
    tilingDatafromBin->dyPadRight = 15;
    tilingDatafromBin->rstdPadRight = 7;
    tilingDatafromBin->roundUpNumLastDim = 7024;
    tilingDatafromBin->roundUpNumLastDimDtype = 14048;
    tilingDatafromBin->roundUp1Dtype = 8;
    tilingDatafromBin->roundUpNumLastDimFloat = 28064;

    uint32_t blockDim = tilingDatafromBin->numCore;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(51);
    ICPU_RUN_KF(
        add_layer_norm_grad, blockDim, dy, x1, x2, rstd, mean, gamma, dsum, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dsum);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(add_layer_norm_grad_test, test_case_61)
{
    size_t dyByteSize = 4 * 7009 * sizeof(int16_t);
    size_t x1ByteSize = 4 * 7009 * sizeof(int16_t);
    size_t rstdByteSize = 4 * sizeof(float);
    size_t meanByteSize = 4 * sizeof(float);
    size_t gammaByteSize = 7009 * sizeof(int16_t);
    size_t dsumByteSize = 4 * 7009 * sizeof(int16_t);
    size_t dxByteSize = 4 * 7009 * sizeof(int16_t);
    size_t dgammaByteSize = 7009 * sizeof(float);
    size_t dbetaByteSize = 7009 * sizeof(float);
    size_t tiling_data_size = sizeof(AddLayerNormGradTilingData);

    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* dsum = (uint8_t*)AscendC::GmAlloc(dsumByteSize);

    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* dgamma = (uint8_t*)AscendC::GmAlloc(dgammaByteSize);
    uint8_t* dbeta = (uint8_t*)AscendC::GmAlloc(dbetaByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AddLayerNormGradTilingData* tilingDatafromBin = reinterpret_cast<AddLayerNormGradTilingData*>(tiling);

    tilingDatafromBin->numCore = 4;
    tilingDatafromBin->numLastDim = 7009;
    tilingDatafromBin->numFirstDim = 4;
    tilingDatafromBin->nInOneCoreLength = 1;
    tilingDatafromBin->nInOneCoreLengthTail = 1;
    tilingDatafromBin->ndInOneCoreLength = 7009;
    tilingDatafromBin->nAvailInUb = 1;
    tilingDatafromBin->dInnerLength = 4416;
    tilingDatafromBin->dInnerLengthTail = 2593;
    tilingDatafromBin->dOuterLength = 2;
    tilingDatafromBin->nInOneCoreNorm = 1;
    tilingDatafromBin->gmOneCoreElemXYNorm = 7009;
    tilingDatafromBin->nAvailInUbNorm = 1;
    tilingDatafromBin->nMiddleCountNorm = 1;
    tilingDatafromBin->ndRoundUpDtypeNorm = 14048;
    tilingDatafromBin->n1RoundUpFloatNorm = 32;
    tilingDatafromBin->nInUbTotalNormTail = 1;
    tilingDatafromBin->nInOneCoreTail = 1;
    tilingDatafromBin->gmOneCoreElemXYTail = 7009;
    tilingDatafromBin->nAvailInUbTail = 1;
    tilingDatafromBin->nMiddleCountTail = 1;
    tilingDatafromBin->ndRoundUpDtypeTail = 14048;
    tilingDatafromBin->n1RoundUpFloatTail = 32;
    tilingDatafromBin->nInUbTotalTailTail = 1;
    tilingDatafromBin->dyPadRight = 15;
    tilingDatafromBin->rstdPadRight = 7;
    tilingDatafromBin->roundUpNumLastDim = 7024;
    tilingDatafromBin->roundUpNumLastDimDtype = 14048;
    tilingDatafromBin->roundUp1Dtype = 8;
    tilingDatafromBin->roundUpNumLastDimFloat = 28064;

    uint32_t blockDim = tilingDatafromBin->numCore;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_SET_TILING_KEY(61);
    ICPU_RUN_KF(
        add_layer_norm_grad, blockDim, dy, x1, x2, rstd, mean, gamma, dsum, dx, dgamma, dbeta, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(dy);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(mean);
    AscendC::GmFree(gamma);
    AscendC::GmFree(dsum);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}