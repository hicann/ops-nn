/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_function_def.h"
#include "../data_utils.h"

extern "C" __global__ __aicore__ void ge_glu_grad_v2(
    GM_ADDR dy, GM_ADDR x, GM_ADDR g_gelu, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling);

std::string GetShapesString(const std::vector<std::vector<int64_t>>& shapeInfo)
{
    std::string ret = "{";
    for (auto shape : shapeInfo) {
        ret += "{";
        for (auto dim : shape) {
            ret += std::to_string(dim) + ",";
        }
        ret += "},";
    }
    return ret + "}";
}

std::string GetShapesString(const std::vector<int64_t>& shape)
{
    std::string ret = "{";
    for (auto dim : shape) {
        ret += std::to_string(dim) + ",";
    }
    return ret + "}";
}

int64_t GetShapeSize(const std::vector<std::vector<int64_t>>& shapeInfo, const size_t& index)
{
    int64_t shapeSize = 1;
    for (auto shape : shapeInfo[index]) {
        shapeSize *= shape;
    }
    return shapeSize;
}

class ge_glu_grad_v2_test : public testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        std::cout << "ge_glu_grad_v2_test SetUpTestSuite" << std::endl;
    }

    static void TearDownTestSuite()
    {
        std::cout << "ge_glu_grad_v2_test SetUpTestSuite" << std::endl;
    }

    template <typename T>
    void SingleCallOperator(
        const std::vector<std::vector<int64_t>>& shapeInfos, const std::vector<int64_t>& attrInfos,
        const ge::DataType dyDtype, const bool needCompare = false)
    {
        // get tiling data and tiling key
        optiling::GeGluGradV2Tiling tilingObject(shapeInfos, attrInfos, dyDtype);
        tilingObject.RunTiling4GeGluGradV2();
        int32_t tilingKey = tilingObject.GetTilingKey();

        system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/ge_glu_grad_v2/ge_glu_grad_v2_data ./");
        system("chmod -R 755 ./ge_glu_grad_v2_data/ && rm -rf ./ge_glu_grad_v2_data/*bin");
        std::string genCMD = "cd ./ge_glu_grad_v2_data/ && python3 gen_data.py '" + GetShapesString(shapeInfos) +
                             "' '" + GetShapesString(attrInfos) + "' " + std::to_string(tilingKey);
        EXPECT_EQ(system(genCMD.c_str()), 0);
        size_t usrWorkspaceSize = 4096;
        uint8_t* usrWorkSpace = (uint8_t*)AscendC::GmAlloc(usrWorkspaceSize);
        size_t tilingSize = sizeof(GeGluGradV2TilingData);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
        tilingObject.FillTilingData(tiling);

        size_t dyByteSize = GetShapeSize(shapeInfos, 0) * sizeof(T);
        size_t xByteSize = GetShapeSize(shapeInfos, 1) * sizeof(T);
        size_t geluByteSize = dyByteSize;
        size_t dxByteSize = xByteSize;

        uint8_t* dy = (uint8_t*)AscendC::GmAlloc(dyByteSize);
        uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
        uint8_t* gelu = (uint8_t*)AscendC::GmAlloc(geluByteSize);
        uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);

        ReadFile("./ge_glu_grad_v2_data/input_dy.bin", dyByteSize, dy, dyByteSize);
        ReadFile("./ge_glu_grad_v2_data/input_x.bin", xByteSize, x, xByteSize);
        ReadFile("./ge_glu_grad_v2_data/input_gelu.bin", geluByteSize, gelu, geluByteSize);

        ICPU_SET_TILING_KEY(tilingKey);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(ge_glu_grad_v2, tilingObject.GetNeedCoreNum(), dy, x, gelu, dx, usrWorkSpace, tiling);

        WriteFile("./ge_glu_grad_v2_data/output_dx.bin", dx, dxByteSize);

        AscendC::GmFree((void*)dx);
        AscendC::GmFree((void*)gelu);
        AscendC::GmFree((void*)x);
        AscendC::GmFree((void*)dy);
        AscendC::GmFree((void*)tiling);
        AscendC::GmFree((void*)usrWorkSpace);
        if (needCompare) {
            std::string cmpCMD = "cd ./ge_glu_grad_v2_data/ && python3 compare_data.py " + std::to_string(tilingKey);
            EXPECT_EQ(system(cmpCMD.c_str()), 0);
        }
    }
};

TEST_F(ge_glu_grad_v2_test, test_case_bfloat16_key_101)
{
    std::vector<std::vector<int64_t>> shapeInfos({{3, 23}, {3, 23 * 2}});
    std::vector<int64_t> attrInfos({-1, 1, 0});
    SingleCallOperator<bfloat16_t>(shapeInfos, attrInfos, ge::DT_BF16);
}

TEST_F(ge_glu_grad_v2_test, test_case_bfloat16_key_103)
{
    std::vector<std::vector<int64_t>> shapeInfos({{5, 7}, {5, 7 * 2}});
    std::vector<int64_t> attrInfos({-1, 1, 0});
    SingleCallOperator<bfloat16_t>(shapeInfos, attrInfos, ge::DT_BF16);
}

TEST_F(ge_glu_grad_v2_test, test_case_bfloat16_key_701)
{
    std::vector<std::vector<int64_t>> shapeInfos({{6, 25}, {6, 25 * 2}});
    std::vector<int64_t> attrInfos({-1, 0, 0});
    SingleCallOperator<bfloat16_t>(shapeInfos, attrInfos, ge::DT_BF16);
}

TEST_F(ge_glu_grad_v2_test, test_case_bfloat16_key_703)
{
    std::vector<std::vector<int64_t>> shapeInfos({{2, 11}, {2, 11 * 2}});
    std::vector<int64_t> attrInfos({-1, 0, 0});
    SingleCallOperator<bfloat16_t>(shapeInfos, attrInfos, ge::DT_BF16);
}

TEST_F(ge_glu_grad_v2_test, test_case_float16_key_201)
{
    std::vector<std::vector<int64_t>> shapeInfos({{2, 25}, {2, 25 * 2}});
    std::vector<int64_t> attrInfos({-1, 1, 0});
    SingleCallOperator<half>(shapeInfos, attrInfos, ge::DT_FLOAT16, false);
}

TEST_F(ge_glu_grad_v2_test, test_case_float16_key_203)
{
    std::vector<std::vector<int64_t>> shapeInfos({{2, 5}, {2, 5 * 2}});
    std::vector<int64_t> attrInfos({-1, 1, 0});
    SingleCallOperator<half>(shapeInfos, attrInfos, ge::DT_FLOAT16, false);
}

TEST_F(ge_glu_grad_v2_test, test_case_float16_key_801)
{
    std::vector<std::vector<int64_t>> shapeInfos({{3, 27}, {3, 27 * 2}});
    std::vector<int64_t> attrInfos({-1, 0, 0});
    SingleCallOperator<half>(shapeInfos, attrInfos, ge::DT_FLOAT16, false);
}

TEST_F(ge_glu_grad_v2_test, test_case_float16_key_803)
{
    std::vector<std::vector<int64_t>> shapeInfos({{2, 7}, {2, 7 * 2}});
    std::vector<int64_t> attrInfos({-1, 0, 0});
    SingleCallOperator<half>(shapeInfos, attrInfos, ge::DT_FLOAT16, false);
}

TEST_F(ge_glu_grad_v2_test, test_case_float32_key_301)
{
    std::vector<std::vector<int64_t>> shapeInfos({{2, 37}, {2, 37 * 2}});
    std::vector<int64_t> attrInfos({-1, 1, 0});
    SingleCallOperator<float>(shapeInfos, attrInfos, ge::DT_FLOAT, false);
}

TEST_F(ge_glu_grad_v2_test, test_case_float32_key_303)
{
    std::vector<std::vector<int64_t>> shapeInfos({{5, 5}, {5, 5 * 2}});
    std::vector<int64_t> attrInfos({-1, 1, 0});
    SingleCallOperator<float>(shapeInfos, attrInfos, ge::DT_FLOAT, false);
}

TEST_F(ge_glu_grad_v2_test, test_case_float32_key_901)
{
    std::vector<std::vector<int64_t>> shapeInfos({{3, 41}, {3, 41 * 2}});
    std::vector<int64_t> attrInfos({-1, 0, 0});
    SingleCallOperator<float>(shapeInfos, attrInfos, ge::DT_FLOAT, false);
}

TEST_F(ge_glu_grad_v2_test, test_case_float32_key_903)
{
    std::vector<std::vector<int64_t>> shapeInfos({{2, 7}, {2, 7 * 2}});
    std::vector<int64_t> attrInfos({-1, 0, 0});
    SingleCallOperator<float>(shapeInfos, attrInfos, ge::DT_FLOAT, false);
}
