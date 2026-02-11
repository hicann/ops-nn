/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_log_sigmoid.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <string>
#include <numeric>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../data_utils.h"
#include "../tiling_test_help.h"
#include "../atvoss_elewise_op_test_help.h"
#include "log_sigmoid.cpp"

using namespace LogSigmoidOp;

class log_sigmoid_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "log_sigmoid_test SetUp\n" << std::endl;
        system("cp -r ../../../../../../../activation/log_sigmoid/tests/ut/op_kernel/log_sigmoid_data ./");
        system("chmod -R 755 ./log_sigmoid_data/");
        system("cd ./log_sigmoid_data/ && rm -rf ./*bin");
    }
    static void TearDownTestCase()
    {
        std::cout << "log_sigmoid_test TearDown\n" << std::endl;
        system("rm -rf log_sigmoid_data");
    }
};

constexpr std::array<std::array<int, 1>, 3> TemplateNumValues = {{{TPL_FP16}, {TPL_BF16}, {TPL_FP32}}};

// CommonElewiseParams为elewise模板提供的参数集合
constexpr auto logSigmoidParams = CombineParams(CommonElewiseParams, TemplateNumValues);

template <std::size_t Index>
void InvokeLogSigmoidKernel(
    const TilingTestHelp::TilingInfo& tilingInfo, GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // 没有额外模板参数的用CommonElementWiseParams，有额外模板参数用对应算子的Params
    std::cout << "tilingInfo.tilingKey:" << tilingInfo.tilingKey << std::endl;
    if constexpr (Index < logSigmoidParams.size()) {
        // 没有额外模板参数作为tilingkey用COMMON_ELEWISE_PARAM
        // 有额外模板参数作为tilingkey用CUSTOM_ELEWISE_PARAM
        // 此算子在原有的elewise模板的基础上扩展了1个参数，因此extParamCnt填1
        if (tilingInfo.tilingKey == GET_TPL_TILING_KEY(CUSTOM_ELEWISE_PARAM(logSigmoidParams, Index, 1))) {
            ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
            std::cout << __func__ << " Params:" << ToString(logSigmoidParams[Index]) << std::endl;
            auto func = log_sigmoid<CUSTOM_ELEWISE_PARAM(logSigmoidParams, Index, 1)>;
            AscendC::SetKernelMode(KernelMode::AIV_MODE);
            ICPU_RUN_KF(func, tilingInfo.blockNum, x, y, workspace, tiling);
        } else {
            InvokeLogSigmoidKernel<Index + 1>(tilingInfo, x, y, workspace, tiling);
        }
    }
}

// 数据路径配置
const std::string INPUT_FILE_PATH = "log_sigmoid_data/x.bin";
const std::string OUTPUT_FILE_PATH = "log_sigmoid_data/y.bin";

// 生成golden数据
static void GenGolden(int64_t dataSize, const std::string& typeStr)
{
    auto cmd = "cd ./log_sigmoid_data/ && rm -rf ./*bin && python3 gen_data.py '[" + std::to_string(dataSize) + "]' '" +
               typeStr + "'";
    system(cmd.c_str());
}

// 比较数据
static int32_t CompareData(std::string typeStr)
{
    auto cmd = "cd ./log_sigmoid_data/ && python3 compare_data.py '" + typeStr + "'";
    return system(cmd.c_str());
}

void TestLogSigmoid(const std::vector<int64_t>& shapeX, int inDType, int64_t dtypesize, const std::string& typeStr){
    // 1. 准备Tiling参数
    TilingTestHelp::ParamManager manager;
    manager.Init(1, 1);
    const auto shapeY = shapeX; // 输出形状与输入相同
    manager.AddInputShape(shapeX, inDType, FORMAT_ND);
    manager.AddOutputShape(shapeY, inDType, FORMAT_ND);

    // 2. 执行Tiling计算
    TilingTestHelp::TilingInfo tilingInfo;
    EXPECT_TRUE(DoTiling("LogSigmoid", manager, tilingInfo, UT_SOC_VERSION, "{}"));
    ASSERT_GT(tilingInfo.tilingDataSize, 0);

    // 3. 分配设备内存
    size_t xSize = TilingTestHelp::ShapeSize(shapeX) * dtypesize;
    size_t ySize = TilingTestHelp::ShapeSize(shapeY) * dtypesize;
    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(ySize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingInfo.workspaceSizes.front()));

    // 4. 生成Golden数据
    const int64_t totalElements = std::accumulate(shapeX.begin(), shapeX.end(), 1, std::multiplies<int64_t>());
    GenGolden(totalElements, typeStr);

    // 5. 输入数据
    ReadFile(INPUT_FILE_PATH, xSize, x, xSize);

    // 6. 执行内核
    InvokeLogSigmoidKernel<0>(tilingInfo, x, y, workspace, tilingInfo.tilingData.get());

    // 7. 输出数据
    WriteFile(OUTPUT_FILE_PATH, y, ySize);

    // 8. 清理资源
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);

    // 9. 对比输出
    EXPECT_EQ(CompareData(typeStr), 0);
}

TEST_F(log_sigmoid_test, case_half_001)
{
    TestLogSigmoid({16, 4}, DT_FLOAT16, sizeof(uint16_t), "float16");
}

TEST_F(log_sigmoid_test, case_bfloat16_001)
{
    TestLogSigmoid({16, 4}, DT_BF16, sizeof(uint16_t), "bfloat16");
}

TEST_F(log_sigmoid_test, case_float_001)
{
    TestLogSigmoid({16, 4}, DT_FLOAT, sizeof(float), "float32");
}