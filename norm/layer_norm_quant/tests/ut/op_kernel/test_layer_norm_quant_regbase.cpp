/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_layer_norm_quant.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_layer_norm_quant_regbase.h"

using namespace std;

extern "C" void layer_norm_quant(
    uint8_t* x, uint8_t* gamma, uint8_t* beta, uint8_t* scales, uint8_t* offset,
    uint8_t* y, uint8_t* scale_out, uint8_t* workspace, uint8_t* tiling);

class layer_norm_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "layer_norm_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "layer_norm_quant_test TearDown\n" << endl;
    }
};

TEST_F(layer_norm_quant_test, test_case_fp32_normal)
{
    int N = 3;
    int D = 128;
    int scaleSize = 1;
    size_t inputByteSize = N * D * sizeof(int32_t);
    size_t weightByteSize = D * sizeof(int32_t);
    size_t outputByteSize = N * sizeof(int32_t);
    size_t quantByteSize = N * D * sizeof(int8_t);

    size_t tiling_data_size = sizeof(LayerNormQuantRegTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* s = (uint8_t*)AscendC::GmAlloc(scaleSize * sizeof(int32_t));
    uint8_t* o = (uint8_t*)AscendC::GmAlloc(scaleSize * sizeof(int8_t));

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(quantByteSize);
    uint8_t* out_scale = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormQuantRegTilingData* tilingDatafromBin =
        reinterpret_cast<LayerNormQuantRegTilingData*>(tiling);

    tilingDatafromBin->numCore = 3;
    tilingDatafromBin->numLastDim = 128;
    tilingDatafromBin->numFirstDim = 3;
    tilingDatafromBin->nlFirstdimPerCore = 1;
    tilingDatafromBin->lFirstdimPerCore = 1;
    tilingDatafromBin->firstDimPerTimes = 1;
    tilingDatafromBin->colsAligned = 128;
    tilingDatafromBin->epsStr = 1e-6 
    tilingDatafromBin->aveStr = 0.007812;
    tilingDatafromBin->sliceNum = 1;
    tilingDatafromBin->sliceSize = 128;
    tilingDatafromBin->tailSliceSize = 128;

    // normal fp32
    ICPU_SET_TILING_KEY(2310000000);
    ICPU_RUN_KF(
        layer_norm_quant, blockDim, x, gamma, beta, s, o, y, out_scale, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(s);
    AscendC::GmFree(o);

    AscendC::GmFree(y);
    AscendC::GmFree(out_scale);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(layer_norm_quant_test, test_case_fp32_split)
{
    int N = 1;
    int D = 11000;
    int scaleSize = 1;
    size_t inputByteSize = N * D * sizeof(int32_t);
    size_t weightByteSize = D * sizeof(int32_t);
    size_t outputByteSize = N * sizeof(int32_t);
    size_t quantByteSize = N * D * sizeof(int8_t);

    size_t tiling_data_size = sizeof(LayerNormQuantRegTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* s = (uint8_t*)AscendC::GmAlloc(scaleSize * sizeof(int32_t));
    uint8_t* o = (uint8_t*)AscendC::GmAlloc(scaleSize * sizeof(int8_t));

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(quantByteSize);
    uint8_t* out_scale = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    LayerNormQuantRegTilingData* tilingDatafromBin =
        reinterpret_cast<LayerNormQuantRegTilingData*>(tiling);

    tilingDatafromBin->numCore = 1;
    tilingDatafromBin->numLastDim = 11000;
    tilingDatafromBin->numFirstDim = 1;
    tilingDatafromBin->nlFirstdimPerCore = 1;
    tilingDatafromBin->lFirstdimPerCore = 1;
    tilingDatafromBin->firstDimPerTimes = 1;
    tilingDatafromBin->colsAligned = 11008;
    tilingDatafromBin->epsStr = 1e-6 
    tilingDatafromBin->aveStr = 0.000091;
    tilingDatafromBin->sliceNum = 2;
    tilingDatafromBin->sliceSize = 10112;
    tilingDatafromBin->tailSliceSize = 888;

    // normal fp32
    ICPU_SET_TILING_KEY(2300000000);
    ICPU_RUN_KF(
        layer_norm_quant, blockDim, x, gamma, beta, s, o, y, out_scale, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(s);
    AscendC::GmFree(o);

    AscendC::GmFree(y);
    AscendC::GmFree(out_scale);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
