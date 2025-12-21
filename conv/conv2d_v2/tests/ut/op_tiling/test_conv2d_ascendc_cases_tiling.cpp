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
 * \file test_conv2d_ascendc_cases.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "../../../op_host/op_tiling/conv2d_tiling.h"
#include "platform/platform_info.h"
#include "test_conv2d_ascendc_utils_tiling.h"
using namespace conv_api_tiling_test;

using namespace std;
using namespace conv_tiling;
using namespace conv_tiling_algo_m;
using namespace conv_tiling_algo_hw;

class TestConv2dTilingCases : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown()
    {
    }
};
class TestConv2dHWmode : public TestConv2dTilingCases {};
class TestConv2dMmode : public TestConv2dTilingCases {};

TEST_F(TestConv2dHWmode,test_c04_Case_1) {
    Conv2DCase({4,16,16},{16,3,3},{1,1,1,1},{1,1},{1,1},{16,2},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_2) {
    Conv2DCase({4,1,60000},{512,1,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_3) {
    Conv2DCase({4,1,65536},{512,1,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_4) {
    Conv2DCase({4,64,2055},{16,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_5) {
    Conv2DCase({3,214,214},{1408,14,14},{0,0,0,0},{14,14},{1,1},{2,2},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_6) {
    Conv2DCase({3,1344,1344},{64,7,7},{3,3,3,3},{2,2},{1,1},{16,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{false,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_7) {
    Conv2DCase({3,832,1216},{128,3,3},{1,1,1,1},{1,1},{1,1},{8,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_8) {
    Conv2DCase({3,87,97},{13,33,6},{1,2,3,4},{12,11},{1,1},{2,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_9) {
    Conv2DCase({4,1,16},{16,1,3},{1,1,1,1},{1,1},{1,1},{16,2},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_10) {
    Conv2DCase({4,1,106},{512,1,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_11) {
    Conv2DCase({4,1,188},{512,1,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_12) {
    Conv2DCase({4,1,2560},{16,1,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_13) {
    Conv2DCase({3,1,267},{1408,1,14},{0,0,0,0},{14,14},{1,1},{2,2},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_14) {
    Conv2DCase({3,1,56},{64,1,7},{3,3,3,3},{2,2},{1,1},{16,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{false,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_15) {
    Conv2DCase({3,1,1216},{128,1,3},{1,1,1,1},{1,1},{1,1},{8,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_16) {
    Conv2DCase({3,1,97},{13,1,6},{1,2,3,4},{12,11},{1,1},{2,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_17) {
    Conv2DCase({1,29707,2192},{160,21,10},{7,5,4,2},{56,5},{1,8},{8,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_c04_Case_18) {
    Conv2DCase({3,4096,4096},{64,7,7},{3,3,3,3},{2,2},{1,1},{32,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,true},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_hif8_fp8_nobias_Case_1) {
    uint8_t roundMode = 0;
    for(uint8_t dTypesIndex = 1; dTypesIndex < SUPPORTED_QUANT_TYPES_WITHOUT_BIAS.size(); dTypesIndex++) {
        if(SUPPORTED_QUANT_TYPES_WITHOUT_BIAS[dTypesIndex][2] == ConvDtype::HIFLOAT8) {
            roundMode = 2;
        } else if (SUPPORTED_QUANT_TYPES_WITHOUT_BIAS[dTypesIndex][2] == ConvDtype::FLOAT8_E4M3FN ||
                   SUPPORTED_QUANT_TYPES_WITHOUT_BIAS[dTypesIndex][2] == ConvDtype::INT8) {
            roundMode = 1;
        }
        Conv2DCase({3,16,64},{16,3,3},{1,1,1,1},{1,1},{1,1},{1,1},SUPPORTED_QUANT_TYPES_WITHOUT_BIAS[dTypesIndex],{false,true,false},{0,roundMode,0}, 1);
    }
}

TEST_F(TestConv2dHWmode,test_hif8_fp8_hasbias_Case_2) {
    uint8_t roundMode = 0;
    for(uint8_t dTypesIndex = 1; dTypesIndex < SUPPORTED_QUANT_TYPES_WITH_BIAS.size(); dTypesIndex++) {
        if(SUPPORTED_QUANT_TYPES_WITH_BIAS[dTypesIndex][3] == ConvDtype::HIFLOAT8) {
            roundMode = 2;
        } else if (SUPPORTED_QUANT_TYPES_WITH_BIAS[dTypesIndex][3] == ConvDtype::FLOAT8_E4M3FN ||
                   SUPPORTED_QUANT_TYPES_WITH_BIAS[dTypesIndex][3] == ConvDtype::INT8) {
            roundMode = 1;
        }
        Conv2DCase({3,1024,64},{4096,3,3},{1,1,1,1},{1,1},{1,1},{1,1},SUPPORTED_QUANT_TYPES_WITH_BIAS[dTypesIndex],{true,true,false},{0,roundMode,0}, 1);
    }
}

TEST_F(TestConv2dMmode,test_int8_0) {
   Conv2DCase({32,32,32},{16,1,1},{0,0,0,0},{1,1},{1,1},{32,1},{ConvDtype::INT8,ConvDtype::INT8,ConvDtype::INT32,ConvDtype::FLOAT16},{true,true,false},{1,1,0}, 1);
}

TEST_F(TestConv2dHWmode,test_fp16_sdxl_0) {
  Conv2DCase({1,1,16},{16,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_1) {
  Conv2DCase({1280,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_2) {
  Conv2DCase({640,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_3) {
  Conv2DCase({640,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_4) {
  Conv2DCase({320,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_5) {
  Conv2DCase({320,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_6) {
  Conv2DCase({1920,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_7) {
  Conv2DCase({640,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_8) {
  Conv2DCase({2560,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_9) {
  Conv2DCase({2560,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_10) {
  Conv2DCase({1920,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_11) {
  Conv2DCase({1280,72,56},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_12) {
  Conv2DCase({960,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_13) {
  Conv2DCase({640,144,112},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_14) {
  Conv2DCase({640,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_15) {
  Conv2DCase({1920,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_16) {
  Conv2DCase({640,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_17) {
  Conv2DCase({960,52,76},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_18) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_19) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_20) {
  Conv2DCase({640,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_21) {
  Conv2DCase({640,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_22) {
  Conv2DCase({640,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_23) {
  Conv2DCase({2560,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_24) {
  Conv2DCase({960,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_25) {
  Conv2DCase({1280,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_26) {
  Conv2DCase({1280,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_27) {
  Conv2DCase({640,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_28) {
  Conv2DCase({320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_29) {
  Conv2DCase({640,144,112},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_30) {
  Conv2DCase({960,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_31) {
  Conv2DCase({2560,36,28},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
// TEST_F(TestConv2dHWmode,test_fp16_sdxl_32) {
//   Conv2DCase({320,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
// }
TEST_F(TestConv2dHWmode,test_fp16_sdxl_33) {
  Conv2DCase({640,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_34) {
  Conv2DCase({1280,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_35) {
  Conv2DCase({1280,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_36) {
  Conv2DCase({640,80,192},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_37) {
  Conv2DCase({2560,26,38},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_38) {
  Conv2DCase({2560,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_39) {
  Conv2DCase({1280,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_40) {
  Conv2DCase({320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_41) {
  Conv2DCase({320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_42) {
  Conv2DCase({1280,52,76},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_43) {
  Conv2DCase({640,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_44) {
  Conv2DCase({1280,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_45) {
  Conv2DCase({1280,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_46) {
  Conv2DCase({960,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_47) {
  Conv2DCase({320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_48) {
  Conv2DCase({1920,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_49) {
  Conv2DCase({2560,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_50) {
  Conv2DCase({960,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_51) {
  Conv2DCase({1920,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_52) {
  Conv2DCase({320,144,112},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_53) {
  Conv2DCase({1920,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_54) {
  Conv2DCase({960,112,144},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_55) {
  Conv2DCase({960,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_56) {
  Conv2DCase({960,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_57) {
  Conv2DCase({1920,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_58) {
  Conv2DCase({960,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_59) {
  Conv2DCase({1920,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_60) {
  Conv2DCase({1280,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_61) {
  Conv2DCase({960,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_62) {
  Conv2DCase({960,152,104},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_63) {
  Conv2DCase({640,104,152},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_64) {
  Conv2DCase({2560,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_65) {
  Conv2DCase({1280,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_66) {
  Conv2DCase({1920,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_67) {
  Conv2DCase({320,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_68) {
  Conv2DCase({640,104,152},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_69) {
  Conv2DCase({1920,20,48},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_70) {
  Conv2DCase({320,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_71) {
  Conv2DCase({1920,36,28},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_72) {
  Conv2DCase({960,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_73) {
  Conv2DCase({960,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_74) {
  Conv2DCase({320,112,144},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_75) {
  Conv2DCase({960,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_76) {
  Conv2DCase({2560,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_77) {
  Conv2DCase({960,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_78) {
  Conv2DCase({3,1152,896},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_79) {
  Conv2DCase({960,144,112},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_80) {
  Conv2DCase({1920,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_81) {
  Conv2DCase({320,152,104},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_82) {
  Conv2DCase({3,1024,1024},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_83) {
  Conv2DCase({1920,26,38},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_84) {
  Conv2DCase({2560,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_86) {
  Conv2DCase({128,1152,896},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_87) {
  Conv2DCase({1280,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_88) {
  Conv2DCase({128,897,1153},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_89) {
  Conv2DCase({1280,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_90) {
  Conv2DCase({640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_91) {
  Conv2DCase({640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_92) {
  Conv2DCase({960,104,152},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_93) {
  Conv2DCase({960,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
// TEST_F(TestConv2dHWmode,test_fp16_sdxl_94) {
//   Conv2DCase({128,448,576},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
// }
TEST_F(TestConv2dHWmode,test_fp16_sdxl_95) {
  Conv2DCase({1920,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_97) {
  Conv2DCase({128,1153,897},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_98) {
  Conv2DCase({2560,20,48},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_99) {
  Conv2DCase({1920,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_100) {
  Conv2DCase({640,112,144},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_101) {
  Conv2DCase({128,1025,1025},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_102) {
  Conv2DCase({1280,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_103) {
  Conv2DCase({1280,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_104) {
  Conv2DCase({3,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_105) {
  Conv2DCase({640,152,104},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_106) {
  Conv2DCase({960,80,192},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_107) {
  Conv2DCase({128,576,448},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_108) {
  Conv2DCase({1280,56,72},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_109) {
  Conv2DCase({640,152,104},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_110) {
  Conv2DCase({256,512,512},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_112) {
  Conv2DCase({640,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_113) {
  Conv2DCase({640,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_114) {
  Conv2DCase({3,832,1216},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
// TEST_F(TestConv2dHWmode,test_fp16_sdxl_115) {
//   Conv2DCase({128,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
// }
TEST_F(TestConv2dHWmode,test_fp16_sdxl_116) {
  Conv2DCase({128,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_117) {
  Conv2DCase({320,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_118) {
  Conv2DCase({320,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_119) {
  Conv2DCase({3,1088,960},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_120) {
  Conv2DCase({128,544,480},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_121) {
  Conv2DCase({1280,40,96},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_122) {
  Conv2DCase({1920,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_123) {
  Conv2DCase({128,1216,832},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_124) {
  Conv2DCase({256,208,304},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_125) {
  Conv2DCase({128,833,1217},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_126) {
  Conv2DCase({1280,76,52},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_127) {
  Conv2DCase({256,609,417},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_128) {
  Conv2DCase({3,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_129) {
  Conv2DCase({256,416,608},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_130) {
  Conv2DCase({1280,76,52},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_131) {
  Conv2DCase({128,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_132) {
  Conv2DCase({256,288,224},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_133) {
  Conv2DCase({256,288,224},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_134) {
  Conv2DCase({128,608,416},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_135) {
  Conv2DCase({1920,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_136) {
  Conv2DCase({512,304,208},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_137) {
  Conv2DCase({128,1024,1024},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_138) {
  Conv2DCase({256,608,416},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_139) {
  Conv2DCase({640,112,144},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_140) {
  Conv2DCase({128,416,608},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_141) {
  Conv2DCase({256,256,256},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_142) {
  Conv2DCase({256,417,609},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_143) {
  Conv2DCase({1920,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}

TEST_F(TestConv2dHWmode,test_fp16_sdxl_144) {
  Conv2DCase({128,448,576},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_145) {
  Conv2DCase({512,112,144},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_146) {
  Conv2DCase({256,449,577},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_147) {
  Conv2DCase({128,1217,833},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_148) {
  Conv2DCase({1920,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_149) {
  Conv2DCase({512,144,112},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_150) {
  Conv2DCase({256,224,288},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_151) {
  Conv2DCase({640,80,192},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_152) {
  Conv2DCase({256,576,448},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_153) {
  Conv2DCase({512,152,104},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_154) {
  Conv2DCase({512,144,112},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_155) {
  Conv2DCase({256,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_156) {
  Conv2DCase({3,1216,832},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_157) {
  Conv2DCase({128,832,1216},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_158) {
  Conv2DCase({256,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_159) {
  Conv2DCase({512,128,128},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_160) {
  Conv2DCase({128,1088,960},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_161) {
  Conv2DCase({4,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_162) {
  Conv2DCase({256,448,576},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_163) {
  Conv2DCase({512,104,152},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_164) {
  Conv2DCase({256,304,208},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_165) {
  Conv2DCase({4,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_166) {
  Conv2DCase({128,576,448},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_167) {
  Conv2DCase({512,289,225},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_168) {
  Conv2DCase({512,209,305},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_169) {
  Conv2DCase({320,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_170) {
  Conv2DCase({512,112,144},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_171) {
  Conv2DCase({320,112,144},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_172) {
  Conv2DCase({128,512,512},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_173) {
  Conv2DCase({512,305,209},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_174) {
  Conv2DCase({256,545,481},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_175) {
  Conv2DCase({128,1089,961},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_176) {
  Conv2DCase({320,128,128},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_177) {
  Conv2DCase({512,128,128},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_178) {
  Conv2DCase({512,256,256},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_179) {
  Conv2DCase({256,208,304},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_180) {
  Conv2DCase({320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_181) {
  Conv2DCase({512,136,120},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_182) {
  Conv2DCase({512,272,240},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}

TEST_F(TestConv2dHWmode,test_fp16_sdxl_183) {
  Conv2DCase({512,208,304},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_184) {
  Conv2DCase({640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_185) {
  Conv2DCase({512,136,120},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_186) {
  Conv2DCase({512,224,288},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_187) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_188) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_189) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_190) {
  Conv2DCase({4,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_191) {
  Conv2DCase({512,152,104},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_192) {
  Conv2DCase({512,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_193) {
  Conv2DCase({320,64,64},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_194) {
  Conv2DCase({320,136,120},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_195) {
  Conv2DCase({512,104,152},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_196) {
  Conv2DCase({640,76,52},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_197) {
  Conv2DCase({320,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_198) {
  Conv2DCase({4,136,120},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_199) {
  Conv2DCase({640,52,76},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_200) {
  Conv2DCase({320,152,104},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_201) {
  Conv2DCase({4,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_202) {
  Conv2DCase({640,32,32},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_203) {
  Conv2DCase({320,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_204) {
  Conv2DCase({320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_205) {
  Conv2DCase({256,272,240},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_206) {
  Conv2DCase({1280,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_207) {
  Conv2DCase({320,64,64},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_208) {
  Conv2DCase({320,136,120},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_209) {
  Conv2DCase({8,112,144},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_210) {
  Conv2DCase({640,68,60},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_211) {
  Conv2DCase({320,104,152},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_212) {
  Conv2DCase({320,72,56},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_213) {
  Conv2DCase({640,56,72},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_214) {
  Conv2DCase({640,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_215) {
  Conv2DCase({8,136,120},{8,3,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_216) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_217) {
  Conv2DCase({640,64,64},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_218) {
  Conv2DCase({4,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_219) {
  Conv2DCase({640,34,30},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_220) {
  Conv2DCase({320,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_221) {
  Conv2DCase({1280,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_222) {
  Conv2DCase({640,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_223) {
  Conv2DCase({320,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_224) {
  Conv2DCase({640,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_225) {
  Conv2DCase({320,68,60},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_226) {
  Conv2DCase({320,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_227) {
  Conv2DCase({640,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_228) {
  Conv2DCase({640,68,60},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_229) {
  Conv2DCase({320,144,112},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_230) {
  Conv2DCase({640,64,64},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_231) {
  Conv2DCase({640,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_232) {
  Conv2DCase({640,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_233) {
  Conv2DCase({640,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_234) {
  Conv2DCase({256,224,288},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_235) {
  Conv2DCase({128,608,416},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_236) {
  Conv2DCase({128,512,512},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_237) {
  Conv2DCase({8,152,104},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_238) {
  Conv2DCase({512,225,289},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_239) {
  Conv2DCase({256,513,513},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_240) {
  Conv2DCase({256,577,449},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_241) {
  Conv2DCase({128,416,608},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_242) {
  Conv2DCase({256,256,256},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_243) {
  Conv2DCase({8,144,112},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp16_sdxl_244) {
  Conv2DCase({512,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1);
}

TEST_F(TestConv2dHWmode,test_bf16_gen_5) {
  Conv2DCase({3,114,376},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_6) {
  Conv2DCase({6,234,216},{9,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_7) {
  Conv2DCase({9,456,234},{10,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_8) {
  Conv2DCase({12,246,342},{11,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_9) {
  Conv2DCase({21,53,923},{14,1,3},{0,0,0,0},{6,7},{1,255},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_10) {
  Conv2DCase({24,57,821},{15,1,2},{0,0,0,0},{8,9},{1,255},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_11) {
  Conv2DCase({27,24,714},{16,1,1},{0,0,0,0},{10,11},{1,255},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_12) {
  Conv2DCase({30,42,523},{17,1,2},{0,0,0,0},{12,13},{1,255},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_15) {
  Conv2DCase({45,443,1},{22,255,1},{21,22,0,0},{22,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_59) {
  Conv2DCase({1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_60) {
  Conv2DCase({1,1,134217712},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_62) {
  Conv2DCase({1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_63) {
  Conv2DCase({1,1,134217712},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_65) {
  Conv2DCase({1,67108832,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_66) {
  Conv2DCase({1,1,67108832},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_68) {
  Conv2DCase({1,67108832,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_69) {
  Conv2DCase({1,1,67108832},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_70) {
  Conv2DCase({14,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_77) {
  Conv2DCase({22,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_83) {
  Conv2DCase({6,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_84) {
  Conv2DCase({8,128,128},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_85) {
  Conv2DCase({256,304,208},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_86) {
  Conv2DCase({8,104,152},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_87) {
  Conv2DCase({256,272,240},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_88) {
  Conv2DCase({320,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_89) {
  Conv2DCase({512,257,257},{512,1,1},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_90) {
  Conv2DCase({512,273,241},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_91) {
  Conv2DCase({320,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_92) {
  Conv2DCase({320,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_bf16_gen_93) {
  Conv2DCase({640,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1);
}

TEST_F(TestConv2dHWmode,test_fp32_gen_84) {
  Conv2DCase({3,54,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_94) {
  Conv2DCase({8,94,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_95) {
  Conv2DCase({9,95,97},{13,32,6},{1,2,3,4},{12,13},{3,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_106) {
  Conv2DCase({14,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_113) {
  Conv2DCase({22,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_119) {
  Conv2DCase({6,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_120) {
  Conv2DCase({3,54,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_130) {
  Conv2DCase({12,86,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_142) {
  Conv2DCase({14,55,88},{13,11,9},{4,5,3,2},{12,8},{2,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_149) {
  Conv2DCase({22,59,88},{13,11,9},{4,5,3,2},{7,8},{3,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_155) {
  Conv2DCase({6,55,88},{13,11,9},{4,5,3,2},{12,8},{3,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_156) {
  Conv2DCase({3,58,86},{12,7,5},{3,6,1,2},{3,8},{2,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_166) {
  Conv2DCase({12,86,96},{12,5,22},{1,2,3,4},{10,9},{1,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_178) {
  Conv2DCase({1,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_185) {
  Conv2DCase({24,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_191) {
  Conv2DCase({7,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_192) {
  Conv2DCase({6,58,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_202) {
  Conv2DCase({12,96,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_203) {
  Conv2DCase({11,97,97},{13,32,6},{1,2,3,4},{12,13},{3,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_216) {
  Conv2DCase({64,30,30},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_217) {
  Conv2DCase({15,15,16},{1,2,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_219) {
  Conv2DCase({3,4,4},{1000,2,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_221) {
  Conv2DCase({291,480,480},{291,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_227) {
  Conv2DCase({1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_228) {
  Conv2DCase({1,1,134217712},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_223) {
  Conv2DCase({320,68,60},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_224) {
  Conv2DCase({320,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_214) {
  Conv2DCase({512,18,18},{2048,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_220) {
  Conv2DCase({768,16,16},{768,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{0,0,0}, 1);
}
TEST_F(TestConv2dHWmode,test_fp32_gen_222) {
  Conv2DCase({512,105,105},{1024,1,1},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{0,0,0}, 1);
}


TEST_F(TestConv2dMmode,test_fp16_sdxl_0) {
  Conv2DCase({1,1,16},{16,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_1) {
  Conv2DCase({1280,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_2) {
  Conv2DCase({640,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_3) {
  Conv2DCase({640,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_4) {
  Conv2DCase({320,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_5) {
  Conv2DCase({320,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_6) {
  Conv2DCase({1920,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_7) {
  Conv2DCase({640,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_8) {
  Conv2DCase({2560,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_9) {
  Conv2DCase({2560,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_10) {
  Conv2DCase({1920,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_11) {
  Conv2DCase({1280,72,56},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_12) {
  Conv2DCase({960,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_13) {
  Conv2DCase({640,144,112},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_14) {
  Conv2DCase({640,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}

TEST_F(TestConv2dMmode,test_fp16_sdxl_15) {
  Conv2DCase({1920,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_16) {
  Conv2DCase({640,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_17) {
  Conv2DCase({960,52,76},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_18) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_19) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_20) {
  Conv2DCase({640,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_21) {
  Conv2DCase({640,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_22) {
  Conv2DCase({640,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_23) {
  Conv2DCase({2560,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_24) {
  Conv2DCase({960,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_25) {
  Conv2DCase({1280,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_26) {
  Conv2DCase({1280,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_27) {
  Conv2DCase({640,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_28) {
  Conv2DCase({320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_29) {
  Conv2DCase({640,144,112},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_30) {
  Conv2DCase({960,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_31) {
  Conv2DCase({2560,36,28},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_32) {
  Conv2DCase({320,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_33) {
  Conv2DCase({640,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_34) {
  Conv2DCase({1280,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_35) {
  Conv2DCase({1280,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_36) {
  Conv2DCase({640,80,192},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_37) {
  Conv2DCase({2560,26,38},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_38) {
  Conv2DCase({2560,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_39) {
  Conv2DCase({1280,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_40) {
  Conv2DCase({320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_41) {
  Conv2DCase({320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_42) {
  Conv2DCase({1280,52,76},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_43) {
  Conv2DCase({640,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_44) {
  Conv2DCase({1280,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_45) {
  Conv2DCase({1280,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_46) {
  Conv2DCase({960,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_47) {
  Conv2DCase({320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_48) {
  Conv2DCase({1920,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_49) {
  Conv2DCase({2560,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_50) {
  Conv2DCase({960,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_51) {
  Conv2DCase({1920,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_52) {
  Conv2DCase({320,144,112},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_53) {
  Conv2DCase({1920,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_54) {
  Conv2DCase({960,112,144},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_55) {
  Conv2DCase({960,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_56) {
  Conv2DCase({960,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_57) {
  Conv2DCase({1920,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_58) {
  Conv2DCase({960,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}

TEST_F(TestConv2dMmode,test_fp16_sdxl_59) {
  Conv2DCase({1920,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_60) {
  Conv2DCase({1280,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_61) {
  Conv2DCase({960,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_62) {
  Conv2DCase({960,152,104},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_63) {
  Conv2DCase({640,104,152},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_64) {
  Conv2DCase({2560,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_65) {
  Conv2DCase({1280,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_66) {
  Conv2DCase({1920,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_67) {
  Conv2DCase({320,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_68) {
  Conv2DCase({640,104,152},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_69) {
  Conv2DCase({1920,20,48},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_70) {
  Conv2DCase({320,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_71) {
  Conv2DCase({1920,36,28},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_72) {
  Conv2DCase({960,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_73) {
  Conv2DCase({960,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_74) {
  Conv2DCase({320,112,144},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_75) {
  Conv2DCase({960,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_76) {
  Conv2DCase({2560,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_77) {
  Conv2DCase({960,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_78) {
  Conv2DCase({3,1152,896},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_79) {
  Conv2DCase({960,144,112},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_80) {
  Conv2DCase({1920,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_81) {
  Conv2DCase({320,152,104},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_82) {
  Conv2DCase({3,1024,1024},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_83) {
  Conv2DCase({1920,26,38},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_84) {
  Conv2DCase({2560,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_86) {
  Conv2DCase({128,1152,896},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_87) {
  Conv2DCase({1280,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_88) {
  Conv2DCase({128,897,1153},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_89) {
  Conv2DCase({1280,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_90) {
  Conv2DCase({640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_91) {
  Conv2DCase({640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_92) {
  Conv2DCase({960,104,152},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_93) {
  Conv2DCase({960,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_94) {
  Conv2DCase({128,448,576},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_95) {
  Conv2DCase({1920,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_97) {
  Conv2DCase({128,1153,897},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_98) {
  Conv2DCase({2560,20,48},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_99) {
  Conv2DCase({1920,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_100) {
  Conv2DCase({640,112,144},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_101) {
  Conv2DCase({128,1025,1025},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_102) {
  Conv2DCase({1280,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_103) {
  Conv2DCase({1280,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_104) {
  Conv2DCase({3,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_105) {
  Conv2DCase({640,152,104},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_106) {
  Conv2DCase({960,80,192},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_107) {
  Conv2DCase({128,576,448},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_108) {
  Conv2DCase({1280,56,72},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_109) {
  Conv2DCase({640,152,104},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_110) {
  Conv2DCase({256,512,512},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_112) {
  Conv2DCase({640,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_113) {
  Conv2DCase({640,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_114) {
  Conv2DCase({3,832,1216},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_115) {
  Conv2DCase({128,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_116) {
  Conv2DCase({128,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_117) {
  Conv2DCase({320,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_118) {
  Conv2DCase({320,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_119) {
  Conv2DCase({3,1088,960},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_120) {
  Conv2DCase({128,544,480},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_121) {
  Conv2DCase({1280,40,96},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_122) {
  Conv2DCase({1920,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}


TEST_F(TestConv2dMmode,test_fp16_sdxl_123) {
  Conv2DCase({128,1216,832},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_124) {
  Conv2DCase({256,208,304},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_125) {
  Conv2DCase({128,833,1217},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_126) {
  Conv2DCase({1280,76,52},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_127) {
  Conv2DCase({256,609,417},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_128) {
  Conv2DCase({3,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_129) {
  Conv2DCase({256,416,608},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_130) {
  Conv2DCase({1280,76,52},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_131) {
  Conv2DCase({128,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_132) {
  Conv2DCase({256,288,224},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_133) {
  Conv2DCase({256,288,224},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_134) {
  Conv2DCase({128,608,416},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_135) {
  Conv2DCase({1920,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_136) {
  Conv2DCase({512,304,208},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_137) {
  Conv2DCase({128,1024,1024},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_138) {
  Conv2DCase({256,608,416},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_139) {
  Conv2DCase({640,112,144},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_140) {
  Conv2DCase({128,416,608},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_141) {
  Conv2DCase({256,256,256},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_142) {
  Conv2DCase({256,417,609},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_143) {
  Conv2DCase({1920,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_144) {
  Conv2DCase({128,448,576},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_145) {
  Conv2DCase({512,112,144},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_146) {
  Conv2DCase({256,449,577},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_147) {
  Conv2DCase({128,1217,833},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_148) {
  Conv2DCase({1920,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_149) {
  Conv2DCase({512,144,112},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_150) {
  Conv2DCase({256,224,288},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_151) {
  Conv2DCase({640,80,192},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_152) {
  Conv2DCase({256,576,448},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_153) {
  Conv2DCase({512,152,104},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_154) {
  Conv2DCase({512,144,112},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_155) {
  Conv2DCase({256,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_156) {
  Conv2DCase({3,1216,832},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_157) {
  Conv2DCase({128,832,1216},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_158) {
  Conv2DCase({256,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_159) {
  Conv2DCase({512,128,128},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_160) {
  Conv2DCase({128,1088,960},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_161) {
  Conv2DCase({4,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_162) {
  Conv2DCase({256,448,576},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_163) {
  Conv2DCase({512,104,152},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_164) {
  Conv2DCase({256,304,208},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_165) {
  Conv2DCase({4,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_166) {
  Conv2DCase({128,576,448},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_167) {
  Conv2DCase({512,289,225},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_168) {
  Conv2DCase({512,209,305},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_169) {
  Conv2DCase({320,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_170) {
  Conv2DCase({512,112,144},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_171) {
  Conv2DCase({320,112,144},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_172) {
  Conv2DCase({128,512,512},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_173) {
  Conv2DCase({512,305,209},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_174) {
  Conv2DCase({256,545,481},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_175) {
  Conv2DCase({128,1089,961},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_176) {
  Conv2DCase({320,128,128},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_177) {
  Conv2DCase({512,128,128},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_178) {
  Conv2DCase({512,256,256},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_179) {
  Conv2DCase({256,208,304},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_180) {
  Conv2DCase({320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_181) {
  Conv2DCase({512,136,120},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_182) {
  Conv2DCase({512,272,240},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_183) {
  Conv2DCase({512,208,304},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_184) {
  Conv2DCase({640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_185) {
  Conv2DCase({512,136,120},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_186) {
  Conv2DCase({512,224,288},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_187) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_188) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_189) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_190) {
  Conv2DCase({4,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_191) {
  Conv2DCase({512,152,104},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_192) {
  Conv2DCase({512,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_193) {
  Conv2DCase({320,64,64},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_194) {
  Conv2DCase({320,136,120},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_195) {
  Conv2DCase({512,104,152},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_196) {
  Conv2DCase({640,76,52},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_197) {
  Conv2DCase({320,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_198) {
  Conv2DCase({4,136,120},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_199) {
  Conv2DCase({640,52,76},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_200) {
  Conv2DCase({320,152,104},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_201) {
  Conv2DCase({4,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_202) {
  Conv2DCase({640,32,32},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_203) {
  Conv2DCase({320,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_204) {
  Conv2DCase({320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_205) {
  Conv2DCase({256,272,240},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_206) {
  Conv2DCase({1280,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_207) {
  Conv2DCase({320,64,64},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_208) {
  Conv2DCase({320,136,120},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_209) {
  Conv2DCase({8,112,144},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_210) {
  Conv2DCase({640,68,60},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_211) {
  Conv2DCase({320,104,152},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_212) {
  Conv2DCase({320,72,56},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_213) {
  Conv2DCase({640,56,72},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_214) {
  Conv2DCase({640,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}

TEST_F(TestConv2dMmode,test_fp16_sdxl_215) {
  Conv2DCase({8,136,120},{8,3,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_216) {
  Conv2DCase({640,72,56},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_217) {
  Conv2DCase({640,64,64},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_218) {
  Conv2DCase({4,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_219) {
  Conv2DCase({640,34,30},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_220) {
  Conv2DCase({320,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_221) {
  Conv2DCase({1280,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_222) {
  Conv2DCase({640,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_223) {
  Conv2DCase({320,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_224) {
  Conv2DCase({640,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_225) {
  Conv2DCase({320,68,60},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_226) {
  Conv2DCase({320,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_227) {
  Conv2DCase({640,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_228) {
  Conv2DCase({640,68,60},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_229) {
  Conv2DCase({320,144,112},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_230) {
  Conv2DCase({640,64,64},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_231) {
  Conv2DCase({640,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_232) {
  Conv2DCase({640,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_233) {
  Conv2DCase({640,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_234) {
  Conv2DCase({256,224,288},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_235) {
  Conv2DCase({128,608,416},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_236) {
  Conv2DCase({128,512,512},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_237) {
  Conv2DCase({8,152,104},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_238) {
  Conv2DCase({512,225,289},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_239) {
  Conv2DCase({256,513,513},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_240) {
  Conv2DCase({256,577,449},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_241) {
  Conv2DCase({128,416,608},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_242) {
  Conv2DCase({256,256,256},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_243) {
  Conv2DCase({8,144,112},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp16_sdxl_244) {
  Conv2DCase({512,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1);
}





TEST_F(TestConv2dMmode,test_bf16_gen_5) {
  Conv2DCase({3,114,376},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_6) {
  Conv2DCase({6,234,216},{9,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_7) {
  Conv2DCase({9,456,234},{10,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_8) {
  Conv2DCase({12,246,342},{11,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_9) {
  Conv2DCase({21,53,923},{14,1,3},{0,0,0,0},{6,7},{1,255},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_10) {
  Conv2DCase({24,57,821},{15,1,2},{0,0,0,0},{8,9},{1,255},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_11) {
  Conv2DCase({27,24,714},{16,1,1},{0,0,0,0},{10,11},{1,255},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_12) {
  Conv2DCase({30,42,523},{17,1,2},{0,0,0,0},{12,13},{1,255},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_15) {
  Conv2DCase({45,443,1},{22,255,1},{21,22,0,0},{22,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_59) {
  Conv2DCase({1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}


TEST_F(TestConv2dMmode,test_bf16_gen_62) {
  Conv2DCase({1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_65) {
  Conv2DCase({1,67108832,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_68) {
  Conv2DCase({1,67108832,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_70) {
  Conv2DCase({14,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_77) {
  Conv2DCase({22,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_83) {
  Conv2DCase({6,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_84) {
  Conv2DCase({8,128,128},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_85) {
  Conv2DCase({256,304,208},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_86) {
  Conv2DCase({8,104,152},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_87) {
  Conv2DCase({256,272,240},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_88) {
  Conv2DCase({320,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_89) {
  Conv2DCase({512,257,257},{512,1,1},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_90) {
  Conv2DCase({512,273,241},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_91) {
  Conv2DCase({320,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_92) {
  Conv2DCase({320,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_bf16_gen_93) {
  Conv2DCase({640,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1);
}





TEST_F(TestConv2dMmode,test_fp32_gen_84) {
  Conv2DCase({3,54,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_94) {
  Conv2DCase({8,94,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_95) {
  Conv2DCase({9,95,97},{13,32,6},{1,2,3,4},{12,13},{3,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_106) {
  Conv2DCase({14,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_113) {
  Conv2DCase({22,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_119) {
  Conv2DCase({6,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_120) {
  Conv2DCase({3,54,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_130) {
  Conv2DCase({12,86,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_142) {
  Conv2DCase({14,55,88},{13,11,9},{4,5,3,2},{12,8},{2,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_149) {
  Conv2DCase({22,59,88},{13,11,9},{4,5,3,2},{7,8},{3,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_155) {
  Conv2DCase({6,55,88},{13,11,9},{4,5,3,2},{12,8},{3,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_156) {
  Conv2DCase({3,58,86},{12,7,5},{3,6,1,2},{3,8},{2,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_166) {
  Conv2DCase({12,86,96},{12,5,22},{1,2,3,4},{10,9},{1,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_178) {
  Conv2DCase({1,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_185) {
  Conv2DCase({24,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_191) {
  Conv2DCase({7,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_192) {
  Conv2DCase({6,58,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_202) {
  Conv2DCase({12,96,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_203) {
  Conv2DCase({11,97,97},{13,32,6},{1,2,3,4},{12,13},{3,2},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_216) {
  Conv2DCase({64,30,30},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_217) {
  Conv2DCase({15,15,16},{1,2,3},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_219) {
  Conv2DCase({3,4,4},{1000,2,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_221) {
  Conv2DCase({291,480,480},{291,3,3},{1,1,1,1},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_227) {
  Conv2DCase({1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_223) {
  Conv2DCase({320,68,60},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_224) {
  Conv2DCase({320,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_214) {
  Conv2DCase({512,18,18},{2048,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_220) {
  Conv2DCase({768,16,16},{768,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{1,0,0}, 1);
}
TEST_F(TestConv2dMmode,test_fp32_gen_222) {
  Conv2DCase({512,105,105},{1024,1,1},{0,0,0,0},{2,2},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{1,0,0}, 1);
}

TEST_F(TestConv2dMmode,test_opt_group_fp16_base) {
  Conv2DCase({16,1,1},{16,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{false,false,false},{1,0,0}, 2);
}
TEST_F(TestConv2dMmode,test_opt_group_bf16_base) {
  Conv2DCase({16,1,1},{16,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{false,false,false},{1,0,0}, 2);
}
TEST_F(TestConv2dMmode,test_opt_group_fp32_base) {
  Conv2DCase({16,1,1},{16,1,1},{0,0,0,0},{1,1},{1,1},{1,1},{ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{1,0,0}, 2);
}