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

#ifdef __CCE_KT_TEST__
#include <iostream>
#include <sstream>
#include <string>

#include "data_utils.h"
#include "string.h"
#include "tikicpulib.h"
#include "weight_quant_batch_matmul_v2_tiling_def.h"
#include "weight_quant_batch_matmul_v2.cpp"
#endif

#include <cstdint>

using namespace std;
// using namespace AscendC;

class TestWeightQuantBatchMatmulV2 : public testing::Test {};

TEST_F(TestWeightQuantBatchMatmulV2, matmul_x2_inc_1000110000000003001)
{
    // A16W8, bf16
    size_t m = 1;
    size_t n = 256;
    size_t k = 128;
    size_t group = 2;
    string caseName = "case_711300_0";
    size_t xSize = m * k * sizeof(uint16_t);
    size_t weightSize = n * k * sizeof(uint8_t);
    size_t antiquantSize = n * group * sizeof(uint16_t);
    size_t ySize = m * n * sizeof(uint16_t);
    size_t tilingSize = sizeof(WeightQuantBatchMatmulV2MsdGroupTilingData);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* antiquantScale = (uint8_t*)AscendC::GmAlloc(antiquantSize);
    uint8_t* antiquantOffset = (uint8_t*)AscendC::GmAlloc(antiquantSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(100 * 1024 * 1024 * 4);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    system(
        "cp -r "
        "../../../../matmul/weight_quant_batch_matmul_v2/tests/ut/op_kernel/"
        "weight_quant_batch_matmul_v2_data ./");
    system("chmod -R 755 ./weight_quant_batch_matmul_v2_data/");
    system("cd ./weight_quant_batch_matmul_v2_data/ && rm -rf ./*bin");

    stringstream genData;
    genData << "cd ./weight_quant_batch_matmul_v2_data/ && python3 gen_data.py " << m << " " << k << " " << n << " "
            << group;
    stringstream genTiling;
    genTiling << "cd ./weight_quant_batch_matmul_v2_data/ && python3 gen_tiling.py " << caseName;
    system(genData.str().c_str());
    system(genTiling.str().c_str());

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/weight_quant_batch_matmul_v2_data/x.bin", xSize, x, xSize);
    ReadFile(path + "/weight_quant_batch_matmul_v2_data/weight.bin", weightSize, weight, weightSize);
    ReadFile(
        path + "/weight_quant_batch_matmul_v2_data/antiquant_scale.bin", antiquantSize, antiquantScale, antiquantSize);
    ReadFile(
        path + "/weight_quant_batch_matmul_v2_data/antiquant_offset.bin", antiquantSize, antiquantOffset,
        antiquantSize);
    ReadFile(path + "/weight_quant_batch_matmul_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    auto wrapper = [](GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                      GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::weight_quant_batch_matmul_v2<1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 3, 5, 0>(
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace, tiling);
    };

    ICPU_RUN_KF(
        wrapper, 1, x, weight, antiquantScale, antiquantOffset, nullptr, nullptr, nullptr, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(antiquantScale);
    AscendC::GmFree(antiquantOffset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(TestWeightQuantBatchMatmulV2, matmul_x2_311210_0)
{
    // A16W8, bf16
    size_t m = 128;
    size_t n = 256;
    size_t k = 512;
    size_t group = 1;
    string caseName = "case_311210_0";
    size_t xSize = m * k * sizeof(uint16_t);
    size_t weightSize = n * k * sizeof(uint8_t);
    size_t antiquantSize = n * group * sizeof(uint16_t);
    size_t ySize = m * n * sizeof(uint16_t);
    size_t tilingSize = sizeof(WeightQuantBatchMatmulV2TilingData);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weight = (uint8_t *)AscendC::GmAlloc(weightSize);
    uint8_t *antiquantScale = (uint8_t *)AscendC::GmAlloc(antiquantSize);
    uint8_t *antiquantOffset = (uint8_t *)AscendC::GmAlloc(antiquantSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(100 * 1024 * 1024 * 4);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    system(
        "cp -r "
        "../../../../matmul/weight_quant_batch_matmul_v2/tests/ut/op_kernel/"
        "weight_quant_batch_matmul_v2_data ./");
    system("chmod -R 755 ./weight_quant_batch_matmul_v2_data/");
    system("cd ./weight_quant_batch_matmul_v2_data/ && rm -rf ./*bin");

    stringstream genData;
    genData << "cd ./weight_quant_batch_matmul_v2_data/ && python3 gen_data.py " << m << " " << k << " " << n << " "
            << group;
    stringstream genTiling;
    genTiling << "cd ./weight_quant_batch_matmul_v2_data/ && python3 gen_tiling.py " << caseName;
    system(genData.str().c_str());
    system(genTiling.str().c_str());

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/weight_quant_batch_matmul_v2_data/x.bin", xSize, x, xSize);
    ReadFile(path + "/weight_quant_batch_matmul_v2_data/weight.bin", weightSize, weight, weightSize);
    ReadFile(path + "/weight_quant_batch_matmul_v2_data/antiquant_scale.bin", antiquantSize, antiquantScale,
             antiquantSize);
    ReadFile(path + "/weight_quant_batch_matmul_v2_data/antiquant_offset.bin", antiquantSize, antiquantOffset,
             antiquantSize);
    ReadFile(path + "/weight_quant_batch_matmul_v2_data/tiling.bin", tilingSize, tiling, tilingSize);

    // ICPU_SET_TILING_KEY(311210);
    // ICPU_RUN_KF(weight_quant_batch_matmul_v2, 1, x, weight, antiquantScale, antiquantOffset, nullptr, nullptr, nullptr,
    //             y, workspace, tiling);
    
    auto wrapper = [](GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                      GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::weight_quant_batch_matmul_v2<1, 0, 0, 3, 0, 3, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 3, 5, 0>(
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace, tiling);
    };

    ICPU_RUN_KF(
        wrapper, 1, x, weight, antiquantScale, antiquantOffset, nullptr, nullptr, nullptr, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(antiquantScale);
    AscendC::GmFree(antiquantOffset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}