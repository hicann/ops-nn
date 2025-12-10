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
#include "dequant_swiglu_quant_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void dequant_swiglu_quant(GM_ADDR xGM, GM_ADDR weightSscaleGM,
                                                           GM_ADDR activationScaleGM, GM_ADDR biasGM,
                                                           GM_ADDR quantScaleGM, GM_ADDR quantOffsetGM,
                                                           GM_ADDR groupIndex, GM_ADDR yGM, GM_ADDR scaleGM,
                                                           GM_ADDR workspace, GM_ADDR tiling);

class dequant_swiglu_quant_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "dequant_swiglu_quant_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "dequant_swiglu_quant_test TearDown\n" << endl;
  }
};

// case1: 4608, 2048 with groupnum
// case2: 8, 2304
// case3: 128, 256
TEST_F(dequant_swiglu_quant_test, test_dequant_swiglu_quant_1)
{
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t xweightScaleize = 1 * inDimy * sizeof(int32_t);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = 0;
    size_t quantScaleSize = 1 * (inDimy / 2) * sizeof(half);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = 1 * sizeof(int64_t);


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(xweightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(inDimx * (inDimy / 2) * sizeof(int8_t));
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(inDimx * 1 * sizeof(int32_t));

    uint64_t tilingKey = 0;
    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_1");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_1");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", xweightScaleize, weightScale, xweightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, quantOffset, groupIndex,
                y, scale,
                workspace, tiling);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    // AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)quantScale);
    // AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_dequant_swiglu_quant_2)
{
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(bfloat16_t);
    size_t xweightScaleize = 1 * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(float);
    size_t biasSize = 0;
    size_t quantScaleSize = 1 * (inDimy / 2) * sizeof(float);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = 1 * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(xweightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint64_t tilingKey = 100000000;
    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_2");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_2");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", xweightScaleize, weightScale, xweightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, quantOffset, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    // AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)quantScale);
    // AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_dequant_swiglu_quant_3)
{
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(bfloat16_t);
    size_t xweightScaleize = 1 * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(float);
    size_t biasSize = 0;
    size_t quantScaleSize = 1 * (inDimy / 2) * sizeof(bfloat16_t);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = 1 * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(xweightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint64_t tilingKey = 100000200;
    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_3");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_3");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", xweightScaleize, weightScale, xweightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, quantOffset, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    // AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)quantScale);
    // AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_dequant_swiglu_quant_4)
{
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(bfloat16_t);
    size_t xweightScaleize = 1 * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(float);
    size_t biasSize = 0;
    size_t quantScaleSize = 1 * (inDimy / 2) * sizeof(half);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = 1 * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(xweightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint64_t tilingKey = 100000100;
    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_4");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_4");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", xweightScaleize, weightScale, xweightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, quantOffset, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    // AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)quantScale);
    // AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_activationScale_None_dynamic)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = 0;
    size_t biasSize = 0;
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(float);
    size_t quantOffsetSize = groupNum * (inDimy / 2) * sizeof(float);
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_5");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_5");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100000000);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, quantOffset, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_activationScale_None_static)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = 0;
    size_t biasSize = 0;
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(float);
    size_t quantOffsetSize = groupNum * (inDimy / 2) * sizeof(float);
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_6");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_6");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100000000);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, quantOffset, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_activationScale_None_static_quantIsOne)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = 0;
    size_t biasSize = 0;
    size_t quantScaleSize = groupNum * sizeof(float);
    size_t quantOffsetSize = groupNum * sizeof(float);
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);


    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_7");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_7");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100000000);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, quantOffset, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_fp16) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int16_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 1;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10000);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_bf16) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int16_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 1;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10001);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_bf16_2) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int16_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10002);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_fp16_2) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int16_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10003);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_StaticBiasInt32) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10004);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_StaticBiasInt32_2) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10005);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_StaticBiasFloat_0) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10006);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_StaticBiasFloat_1) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10007);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_StaticBiasFloat_2) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10008);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_StaticBiasFloat_3) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10009);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_StaticBiasFloat_4) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10010);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_StaticBiasFloat_5) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(10011);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_31) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30001);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_32) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30002);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_33) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30003);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_34) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30004);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_35) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30005);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_36) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30006);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_37) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30007);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_38) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30008);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_39) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30009);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_310) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30010);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_311) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30011);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_case_DynamicBiasInt32_312) {
    size_t inputByteSize = 256 * 640 * 2 * sizeof(int32_t);
    size_t input2ByteSize = 1 * sizeof(int32_t);
    size_t outputByteSize = 256 * 640 * sizeof(int8_t);
    size_t output2ByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(input2ByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(output2ByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 14;
    // system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/rms_norm/rms_norm_data ./");
    // system("chmod -R 755 ./rms_norm_data/");
    // system("cd ./rms_norm_data/ && rm -rf ./*bin");
    // system("cd ./rms_norm_data/ && python3 gen_data.py 1 80 2560 float16");
    // system("cd ./rms_norm_data/ && python3 gen_tiling.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluTilingData* tilingDatafromBin = reinterpret_cast<SwiGluTilingData*>(tiling);

    tilingDatafromBin->is32BAligned = 1;
    tilingDatafromBin->isDoubleBuffer = 0;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 320;
    tilingDatafromBin->baseRowLen = 38;
    tilingDatafromBin->baseColLen = 256;
    tilingDatafromBin->activateLeft = 0;

    // ReadFile(path + "/rms_norm_data/input_x.bin", inputByteSize, x, inputByteSize);
    ICPU_SET_TILING_KEY(30012);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim, x, nullptr, nullptr, nullptr, x1, x2, nullptr,
                y, y1,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(y);
    AscendC::GmFree(y1);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_int32_qs_f32)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(int32_t);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(float);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_8");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100003000);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    // //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    // //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_int32_qs_f16)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(int32_t);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(half);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_9");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100003100);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_int32_qs_bf16)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(int32_t);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(half);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_10");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100003200);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_f32_qs_f32)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(float);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(float);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_11");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100002000);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_f32_qs_f16)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(float);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(half);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_12");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100002100);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_f32_qs_bf16)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(float);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(half);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_13");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100002200);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_f16_qs_f32)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(half);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(float);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_14");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100001000);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_f16_qs_f16)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(half);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(half);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_15");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100001100);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_f16_qs_bf16)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(half);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(half);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_16");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100001200);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_bf16_qs_f32)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(half);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(float);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_17");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100000000);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_bf16_qs_f16)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(half);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(half);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_18");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100000100);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}

TEST_F(dequant_swiglu_quant_test, test_x_int32_bias_bf16_qs_bf16)
{
    size_t groupNum = 1;
    size_t inDimx = 128;
    size_t inDimy = 2048;
    size_t xSize = inDimx * inDimy * sizeof(int32_t);
    size_t weightScaleize = groupNum * inDimy * sizeof(float);
    size_t activationScaleSize = inDimx * 1 * sizeof(int32_t);
    size_t biasSize = groupNum * inDimy * sizeof(half);
    size_t quantScaleSize = groupNum * (inDimy / 2) * sizeof(half);
    size_t quantOffsetSize = 0;
    size_t groupIndexSize = groupNum * sizeof(int64_t);
    size_t ySize = inDimx * (inDimy / 2) * sizeof(int8_t);
    size_t scaleSize = inDimx * 1 * sizeof(int32_t);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *weightScale = (uint8_t *)AscendC::GmAlloc(weightScaleize);
    uint8_t *activationScale = (uint8_t *)AscendC::GmAlloc(activationScaleSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *quantScale = (uint8_t *)AscendC::GmAlloc(quantScaleSize);
    uint8_t *quantOffset = (uint8_t *)AscendC::GmAlloc(quantOffsetSize);
    uint8_t *groupIndex = (uint8_t *)AscendC::GmAlloc(groupIndexSize);

    uint8_t *y = (uint8_t *)AscendC::GmAlloc(ySize);
    uint8_t *scale = (uint8_t *)AscendC::GmAlloc(scaleSize);

    uint32_t blockDim = 36;
    size_t workspaceFileSize = 32;
    size_t tilingDataSize = sizeof(DequantSwigluQuantBaseTilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    // system("cp -r ../../../../quant/dequant_swiglu_quant/tests/ut/op_kernel/dequant_swiglu_quant_data ./");
    // system("chmod -R 755 ./dequant_swiglu_quant_data/");
    // system("cd ./dequant_swiglu_quant_data/ && rm -rf ./*bin");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_data.py test_dequant_swiglu_quant_19");
    // system("cd ./dequant_swiglu_quant_data/ && python3 gen_tiling.py test_dequant_swiglu_quant_bias_and_swiglugate");

    char * path_ = get_current_dir_name();
    string path(path_);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_x.bin", xSize, x, xSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_weight.bin", weightScaleize, weightScale, weightScaleize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_act_scale.bin", activationScaleSize, activationScale, activationScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_bias.bin", biasSize, bias, biasSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_quant_scale.bin", quantScaleSize, quantScale, quantScaleSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/input_group_index.bin", groupIndexSize, groupIndex, groupIndexSize);
    // ReadFile(path + "/dequant_swiglu_quant_data/tiling.bin", tilingDataSize, tiling, tilingDataSize);
    ICPU_SET_TILING_KEY(100000200);
    ICPU_RUN_KF(dequant_swiglu_quant, blockDim,
                x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex,
                y, scale,
                workspace, tiling);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_y.bin", y, ySize);
    //WriteFile(path + "/dequant_swiglu_quant_data/output_scale.bin", scale, scaleSize);
    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)weightScale);
    AscendC::GmFree((void *)activationScale);
    AscendC::GmFree((void *)quantScale);
    AscendC::GmFree((void *)quantOffset);
    AscendC::GmFree((void *)groupIndex);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)scale);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    free(path_);
}