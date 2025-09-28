/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_cross_entropy_loss_regbase.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "../tiling_test_help.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_kernel/cross_entropy_loss_apt.cpp"

using namespace std;

class cross_entropy_loss_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "cross_entropy_loss_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "cross_entropy_loss_test TearDown\n" << endl;
    }
};

static bool CompareData()
{
    std::string cmd = "cd ./cross_entropy_loss_apt_data/ && python3 verify.py loss.bin golden.bin ";
    return system(cmd.c_str()) == 0;
}

static void InitEnv(std::string caseName, std::string dtype)
{
    std::string gen = "cd ./cross_entropy_loss_apt_data/ && python3 gen_data.py " + caseName + " " + dtype;
    system("cp -r ../../../../../../../ops/loss/cross_entropy_loss/tests/ut/op_kernel/cross_entropy_loss_apt_data ./");
    system("chmod -R 755 ./cross_entropy_loss_apt_data/");
    system("cd ./cross_entropy_loss_apt_data/ && rm -rf ./*bin");
    system(gen.c_str());
}

TEST_F(cross_entropy_loss_test, test_case_0)
{
    string caseName = "test_case_0";
    string dtype = "float";
    std::vector<int64_t> input{4, 64};
    std::vector<int64_t> targetShape{4};
    std::vector<int64_t> weightShape{64};
    std::vector<int64_t> output1{1};
    auto inDType = DT_FLOAT;
    uint32_t reduction = 2; // sum场景
    std::string reductionStr = "sum";
    TilingTestHelp::ParamManager manager;
    manager.Init(3, 4);
    manager.AddInputShape(input, inDType, FORMAT_ND);
    manager.AddInputShape(targetShape, DT_INT64, FORMAT_ND);
    manager.AddInputShape(weightShape, DT_FLOAT, FORMAT_ND);
    manager.AddAttr("reduction", reductionStr);
    manager.AddAttr("ignore_index", static_cast<int64_t>(-100));
    manager.AddAttr("label_smoothing", 0.0f);
    manager.AddAttr("lse_square_scale_for_zloss", 0.0f);
    manager.AddAttr("return_zloss", false);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(input, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    TilingTestHelp::TilingInfo tilingInfo;
    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
    EXPECT_EQ(DoTiling("CrossEntropyLoss", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

    size_t predictSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t targetSize = TilingTestHelp::ShapeSize(targetShape) * sizeof(int64_t);
    size_t weightSize = TilingTestHelp::ShapeSize(weightShape) * sizeof(float);
    size_t ySize = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y1Size = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t y2Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y3Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t labelSize = predictSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1Size);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2Size);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(y3Size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/cross_entropy_loss_apt_data/predict.bin", predictSize, predict, predictSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/target.bin", targetSize, target, targetSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    auto cross_entropy_loss_func = [](GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob,
                                      GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) {
        cross_entropy_loss<0, 2, 1, 0>(input, target, weight, loss, log_prob, zloss, lse_for_zloss, workspace, tiling);
    };
    ICPU_RUN_KF(
        cross_entropy_loss_func, tilingInfo.blockNum, predict, target, weight, y, y1, y2, y3, workspace,
        tilingInfo.tilingData.get());
    WriteFile(path + "/cross_entropy_loss_apt_data/loss.bin", y, ySize);
    WriteFile(path + "/cross_entropy_loss_apt_data/logprobs.bin", y1, y1Size);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)target);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)y1);
    AscendC::GmFree((void*)y2);
    AscendC::GmFree((void*)y3);
    AscendC::GmFree((void*)workspace);

    EXPECT_EQ(CompareData(), true);
}

TEST_F(cross_entropy_loss_test, test_case_1)
{
    string caseName = "test_case_1";
    string dtype = "float";
    std::vector<int64_t> input{4, 64};
    std::vector<int64_t> targetShape{4};
    std::vector<int64_t> weightShape{64};
    std::vector<int64_t> output1{1};
    auto inDType = DT_FLOAT;
    uint32_t reduction = 1;
    std::string reductionStr = "mean";
    TilingTestHelp::ParamManager manager;
    manager.Init(3, 4);
    manager.AddInputShape(input, inDType, FORMAT_ND);
    manager.AddInputShape(targetShape, DT_INT64, FORMAT_ND);
    manager.AddInputShape(weightShape, DT_FLOAT, FORMAT_ND);
    manager.AddAttr("reduction", reductionStr);
    manager.AddAttr("ignore_index", static_cast<int64_t>(-100));
    manager.AddAttr("label_smoothing", 0.0f);
    manager.AddAttr("lse_square_scale_for_zloss", 0.0f);
    manager.AddAttr("return_zloss", false);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(input, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    TilingTestHelp::TilingInfo tilingInfo;
    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
    EXPECT_EQ(DoTiling("CrossEntropyLoss", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

    size_t predictSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t targetSize = TilingTestHelp::ShapeSize(targetShape) * sizeof(int64_t);
    size_t weightSize = TilingTestHelp::ShapeSize(weightShape) * sizeof(float);
    size_t ySize = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y1Size = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t y2Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y3Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t labelSize = predictSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1Size);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2Size);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(y3Size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/cross_entropy_loss_apt_data/predict.bin", predictSize, predict, predictSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/target.bin", targetSize, target, targetSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    auto cross_entropy_loss_func = [](GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob,
                                      GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) {
        cross_entropy_loss<0, 1, 1, 0>(input, target, weight, loss, log_prob, zloss, lse_for_zloss, workspace, tiling);
    };
    ICPU_RUN_KF(
        cross_entropy_loss_func, tilingInfo.blockNum, predict, target, weight, y, y1, y2, y3, workspace,
        tilingInfo.tilingData.get());
    WriteFile(path + "/cross_entropy_loss_apt_data/loss.bin", y, ySize);
    WriteFile(path + "/cross_entropy_loss_apt_data/logprobs.bin", y1, y1Size);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)target);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)y1);
    AscendC::GmFree((void*)y2);
    AscendC::GmFree((void*)y3);
    AscendC::GmFree((void*)workspace);

    EXPECT_EQ(CompareData(), true);
}

TEST_F(cross_entropy_loss_test, test_case_2)
{
    string caseName = "test_case_2";
    string dtype = "float";
    std::vector<int64_t> input{4, 64};
    std::vector<int64_t> targetShape{4};
    std::vector<int64_t> weightShape{64};
    std::vector<int64_t> output1{4};
    auto inDType = DT_FLOAT;
    uint32_t reduction = 1;
    std::string reductionStr = "none";
    TilingTestHelp::ParamManager manager;
    manager.Init(3, 4);
    manager.AddInputShape(input, inDType, FORMAT_ND);
    manager.AddInputShape(targetShape, DT_INT64, FORMAT_ND);
    manager.AddInputShape(weightShape, DT_FLOAT, FORMAT_ND);
    manager.AddAttr("reduction", reductionStr);
    manager.AddAttr("ignore_index", static_cast<int64_t>(-100));
    manager.AddAttr("label_smoothing", 0.0f);
    manager.AddAttr("lse_square_scale_for_zloss", 0.0f);
    manager.AddAttr("return_zloss", false);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(input, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    TilingTestHelp::TilingInfo tilingInfo;
    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
    EXPECT_EQ(DoTiling("CrossEntropyLoss", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

    size_t predictSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t targetSize = TilingTestHelp::ShapeSize(targetShape) * sizeof(int64_t);
    size_t weightSize = TilingTestHelp::ShapeSize(weightShape) * sizeof(float);
    size_t ySize = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y1Size = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t y2Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y3Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t labelSize = predictSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1Size);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2Size);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(y3Size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/cross_entropy_loss_apt_data/predict.bin", predictSize, predict, predictSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/target.bin", targetSize, target, targetSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    auto cross_entropy_loss_func = [](GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob,
                                      GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) {
        cross_entropy_loss<0, 0, 1, 0>(input, target, weight, loss, log_prob, zloss, lse_for_zloss, workspace, tiling);
    };
    ICPU_RUN_KF(
        cross_entropy_loss_func, tilingInfo.blockNum, predict, target, weight, y, y1, y2, y3, workspace,
        tilingInfo.tilingData.get());
    WriteFile(path + "/cross_entropy_loss_apt_data/loss.bin", y, ySize);
    WriteFile(path + "/cross_entropy_loss_apt_data/logprobs.bin", y1, y1Size);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)target);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)y1);
    AscendC::GmFree((void*)y2);
    AscendC::GmFree((void*)y3);
    AscendC::GmFree((void*)workspace);

    EXPECT_EQ(CompareData(), true);
}

TEST_F(cross_entropy_loss_test, test_case_3)
{
    string caseName = "test_case_3";
    string dtype = "float16";
    std::vector<int64_t> input{4, 64};
    std::vector<int64_t> targetShape{4};
    std::vector<int64_t> weightShape{64};
    std::vector<int64_t> output1{1};
    auto inDType = DT_FLOAT16;
    uint32_t reduction = 2; // sum场景
    std::string reductionStr = "sum";
    TilingTestHelp::ParamManager manager;
    manager.Init(3, 4);
    manager.AddInputShape(input, inDType, FORMAT_ND);
    manager.AddInputShape(targetShape, DT_INT64, FORMAT_ND);
    manager.AddInputShape(weightShape, DT_FLOAT, FORMAT_ND);
    manager.AddAttr("reduction", reductionStr);
    manager.AddAttr("ignore_index", static_cast<int64_t>(-100));
    manager.AddAttr("label_smoothing", 0.0f);
    manager.AddAttr("lse_square_scale_for_zloss", 0.0f);
    manager.AddAttr("return_zloss", false);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(input, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    TilingTestHelp::TilingInfo tilingInfo;
    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
    EXPECT_EQ(DoTiling("CrossEntropyLoss", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

    size_t predictSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t targetSize = TilingTestHelp::ShapeSize(targetShape) * sizeof(int64_t);
    size_t weightSize = TilingTestHelp::ShapeSize(weightShape) * sizeof(float);
    size_t ySize = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y1Size = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t y2Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y3Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t labelSize = predictSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1Size);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2Size);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(y3Size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/cross_entropy_loss_apt_data/predict.bin", predictSize, predict, predictSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/target.bin", targetSize, target, targetSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    auto cross_entropy_loss_func = [](GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob,
                                      GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) {
        cross_entropy_loss<0, 2, 1, 0>(input, target, weight, loss, log_prob, zloss, lse_for_zloss, workspace, tiling);
    };
    ICPU_RUN_KF(
        cross_entropy_loss_func, tilingInfo.blockNum, predict, target, weight, y, y1, y2, y3, workspace,
        tilingInfo.tilingData.get());
    WriteFile(path + "/cross_entropy_loss_apt_data/loss.bin", y, ySize);
    WriteFile(path + "/cross_entropy_loss_apt_data/logprobs.bin", y1, y1Size);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)target);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)y1);
    AscendC::GmFree((void*)y2);
    AscendC::GmFree((void*)y3);
    AscendC::GmFree((void*)workspace);
}

TEST_F(cross_entropy_loss_test, test_case_4)
{
    string caseName = "test_case_4";
    string dtype = "float16";
    std::vector<int64_t> input{4, 64};
    std::vector<int64_t> targetShape{4};
    std::vector<int64_t> weightShape{64};
    std::vector<int64_t> output1{1};
    auto inDType = DT_FLOAT16;
    uint32_t reduction = 1;
    std::string reductionStr = "mean";
    TilingTestHelp::ParamManager manager;
    manager.Init(3, 4);
    manager.AddInputShape(input, inDType, FORMAT_ND);
    manager.AddInputShape(targetShape, DT_INT64, FORMAT_ND);
    manager.AddInputShape(weightShape, DT_FLOAT, FORMAT_ND);
    manager.AddAttr("reduction", reductionStr);
    manager.AddAttr("ignore_index", static_cast<int64_t>(-100));
    manager.AddAttr("label_smoothing", 0.0f);
    manager.AddAttr("lse_square_scale_for_zloss", 0.0f);
    manager.AddAttr("return_zloss", false);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(input, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    TilingTestHelp::TilingInfo tilingInfo;
    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
    EXPECT_EQ(DoTiling("CrossEntropyLoss", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

    size_t predictSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t targetSize = TilingTestHelp::ShapeSize(targetShape) * sizeof(int64_t);
    size_t weightSize = TilingTestHelp::ShapeSize(weightShape) * sizeof(float);
    size_t ySize = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y1Size = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t y2Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y3Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t labelSize = predictSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1Size);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2Size);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(y3Size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/cross_entropy_loss_apt_data/predict.bin", predictSize, predict, predictSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/target.bin", targetSize, target, targetSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    auto cross_entropy_loss_func = [](GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob,
                                      GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) {
        cross_entropy_loss<0, 1, 1, 0>(input, target, weight, loss, log_prob, zloss, lse_for_zloss, workspace, tiling);
    };
    ICPU_RUN_KF(
        cross_entropy_loss_func, tilingInfo.blockNum, predict, target, weight, y, y1, y2, y3, workspace,
        tilingInfo.tilingData.get());
    WriteFile(path + "/cross_entropy_loss_apt_data/loss.bin", y, ySize);
    WriteFile(path + "/cross_entropy_loss_apt_data/logprobs.bin", y1, y1Size);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)target);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)y1);
    AscendC::GmFree((void*)y2);
    AscendC::GmFree((void*)y3);
    AscendC::GmFree((void*)workspace);
}

TEST_F(cross_entropy_loss_test, test_case_5)
{
    string caseName = "test_case_5";
    string dtype = "float16";
    std::vector<int64_t> input{4, 64};
    std::vector<int64_t> targetShape{4};
    std::vector<int64_t> weightShape{64};
    std::vector<int64_t> output1{4};
    auto inDType = DT_FLOAT16;
    uint32_t reduction = 1;
    std::string reductionStr = "none";
    TilingTestHelp::ParamManager manager;
    manager.Init(3, 4);
    manager.AddInputShape(input, inDType, FORMAT_ND);
    manager.AddInputShape(targetShape, DT_INT64, FORMAT_ND);
    manager.AddInputShape(weightShape, DT_FLOAT, FORMAT_ND);
    manager.AddAttr("reduction", reductionStr);
    manager.AddAttr("ignore_index", static_cast<int64_t>(-100));
    manager.AddAttr("label_smoothing", 0.0f);
    manager.AddAttr("lse_square_scale_for_zloss", 0.0f);
    manager.AddAttr("return_zloss", false);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(input, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    TilingTestHelp::TilingInfo tilingInfo;
    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
    EXPECT_EQ(DoTiling("CrossEntropyLoss", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

    size_t predictSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t targetSize = TilingTestHelp::ShapeSize(targetShape) * sizeof(int64_t);
    size_t weightSize = TilingTestHelp::ShapeSize(weightShape) * sizeof(float);
    size_t ySize = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y1Size = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t y2Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y3Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t labelSize = predictSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1Size);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2Size);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(y3Size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/cross_entropy_loss_apt_data/predict.bin", predictSize, predict, predictSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/target.bin", targetSize, target, targetSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    auto cross_entropy_loss_func = [](GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob,
                                      GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) {
        cross_entropy_loss<0, 0, 1, 0>(input, target, weight, loss, log_prob, zloss, lse_for_zloss, workspace, tiling);
    };
    ICPU_RUN_KF(
        cross_entropy_loss_func, tilingInfo.blockNum, predict, target, weight, y, y1, y2, y3, workspace,
        tilingInfo.tilingData.get());
    WriteFile(path + "/cross_entropy_loss_apt_data/loss.bin", y, ySize);
    WriteFile(path + "/cross_entropy_loss_apt_data/logprobs.bin", y1, y1Size);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)target);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)y1);
    AscendC::GmFree((void*)y2);
    AscendC::GmFree((void*)y3);
    AscendC::GmFree((void*)workspace);
}

TEST_F(cross_entropy_loss_test, test_case_6)
{
    string caseName = "test_case_6";
    string dtype = "bfloat16";
    std::vector<int64_t> input{4, 64};
    std::vector<int64_t> targetShape{4};
    std::vector<int64_t> weightShape{64};
    std::vector<int64_t> output1{1};
    auto inDType = DT_BF16;
    uint32_t reduction = 2; // sum场景
    std::string reductionStr = "sum";
    TilingTestHelp::ParamManager manager;
    manager.Init(3, 4);
    manager.AddInputShape(input, inDType, FORMAT_ND);
    manager.AddInputShape(targetShape, DT_INT64, FORMAT_ND);
    manager.AddInputShape(weightShape, DT_FLOAT, FORMAT_ND);
    manager.AddAttr("reduction", reductionStr);
    manager.AddAttr("ignore_index", static_cast<int64_t>(-100));
    manager.AddAttr("label_smoothing", 0.0f);
    manager.AddAttr("lse_square_scale_for_zloss", 0.0f);
    manager.AddAttr("return_zloss", false);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(input, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    TilingTestHelp::TilingInfo tilingInfo;
    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
    EXPECT_EQ(DoTiling("CrossEntropyLoss", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

    size_t predictSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t targetSize = TilingTestHelp::ShapeSize(targetShape) * sizeof(int64_t);
    size_t weightSize = TilingTestHelp::ShapeSize(weightShape) * sizeof(float);
    size_t ySize = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y1Size = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t y2Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y3Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t labelSize = predictSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1Size);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2Size);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(y3Size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/cross_entropy_loss_apt_data/predict.bin", predictSize, predict, predictSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/target.bin", targetSize, target, targetSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    auto cross_entropy_loss_func = [](GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob,
                                      GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) {
        cross_entropy_loss<0, 2, 1, 0>(input, target, weight, loss, log_prob, zloss, lse_for_zloss, workspace, tiling);
    };
    ICPU_RUN_KF(
        cross_entropy_loss_func, tilingInfo.blockNum, predict, target, weight, y, y1, y2, y3, workspace,
        tilingInfo.tilingData.get());
    WriteFile(path + "/cross_entropy_loss_apt_data/loss.bin", y, ySize);
    WriteFile(path + "/cross_entropy_loss_apt_data/logprobs.bin", y1, y1Size);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)target);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)y1);
    AscendC::GmFree((void*)y2);
    AscendC::GmFree((void*)y3);
    AscendC::GmFree((void*)workspace);
}

TEST_F(cross_entropy_loss_test, test_case_7)
{
    string caseName = "test_case_7";
    string dtype = "bfloat16";
    std::vector<int64_t> input{4, 64};
    std::vector<int64_t> targetShape{4};
    std::vector<int64_t> weightShape{64};
    std::vector<int64_t> output1{1};
    auto inDType = DT_BF16;
    uint32_t reduction = 1;
    std::string reductionStr = "mean";
    TilingTestHelp::ParamManager manager;
    manager.Init(3, 4);
    manager.AddInputShape(input, inDType, FORMAT_ND);
    manager.AddInputShape(targetShape, DT_INT64, FORMAT_ND);
    manager.AddInputShape(weightShape, DT_FLOAT, FORMAT_ND);
    manager.AddAttr("reduction", reductionStr);
    manager.AddAttr("ignore_index", static_cast<int64_t>(-100));
    manager.AddAttr("label_smoothing", 0.0f);
    manager.AddAttr("lse_square_scale_for_zloss", 0.0f);
    manager.AddAttr("return_zloss", false);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(input, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    TilingTestHelp::TilingInfo tilingInfo;
    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
    EXPECT_EQ(DoTiling("CrossEntropyLoss", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

    size_t predictSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t targetSize = TilingTestHelp::ShapeSize(targetShape) * sizeof(int64_t);
    size_t weightSize = TilingTestHelp::ShapeSize(weightShape) * sizeof(float);
    size_t ySize = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y1Size = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t y2Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y3Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t labelSize = predictSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1Size);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2Size);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(y3Size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/cross_entropy_loss_apt_data/predict.bin", predictSize, predict, predictSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/target.bin", targetSize, target, targetSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    auto cross_entropy_loss_func = [](GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob,
                                      GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) {
        cross_entropy_loss<0, 1, 1, 0>(input, target, weight, loss, log_prob, zloss, lse_for_zloss, workspace, tiling);
    };
    ICPU_RUN_KF(
        cross_entropy_loss_func, tilingInfo.blockNum, predict, target, weight, y, y1, y2, y3, workspace,
        tilingInfo.tilingData.get());
    WriteFile(path + "/cross_entropy_loss_apt_data/loss.bin", y, ySize);
    WriteFile(path + "/cross_entropy_loss_apt_data/logprobs.bin", y1, y1Size);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)target);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)y1);
    AscendC::GmFree((void*)y2);
    AscendC::GmFree((void*)y3);
    AscendC::GmFree((void*)workspace);
}

TEST_F(cross_entropy_loss_test, test_case_8)
{
    string caseName = "test_case_8";
    string dtype = "bfloat16";
    std::vector<int64_t> input{4, 64};
    std::vector<int64_t> targetShape{4};
    std::vector<int64_t> weightShape{64};
    std::vector<int64_t> output1{4};
    auto inDType = DT_BF16;
    uint32_t reduction = 1;
    std::string reductionStr = "none";
    TilingTestHelp::ParamManager manager;
    manager.Init(3, 4);
    manager.AddInputShape(input, inDType, FORMAT_ND);
    manager.AddInputShape(targetShape, DT_INT64, FORMAT_ND);
    manager.AddInputShape(weightShape, DT_FLOAT, FORMAT_ND);
    manager.AddAttr("reduction", reductionStr);
    manager.AddAttr("ignore_index", static_cast<int64_t>(-100));
    manager.AddAttr("label_smoothing", 0.0f);
    manager.AddAttr("lse_square_scale_for_zloss", 0.0f);
    manager.AddAttr("return_zloss", false);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(input, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    manager.AddOutputShape(output1, inDType, FORMAT_ND);
    TilingTestHelp::TilingInfo tilingInfo;
    // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
    EXPECT_EQ(DoTiling("CrossEntropyLoss", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
    ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

    size_t predictSize = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t targetSize = TilingTestHelp::ShapeSize(targetShape) * sizeof(int64_t);
    size_t weightSize = TilingTestHelp::ShapeSize(weightShape) * sizeof(float);
    size_t ySize = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y1Size = TilingTestHelp::ShapeSize(input) * sizeof(float);
    size_t y2Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    size_t y3Size = TilingTestHelp::ShapeSize(output1) * sizeof(float);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t labelSize = predictSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(predictSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1Size);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2Size);
    uint8_t* y3 = (uint8_t*)AscendC::GmAlloc(y3Size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes.front());
    InitEnv(caseName, dtype);
    char* cpath = get_current_dir_name();
    string path(cpath);
    free(cpath);
    ReadFile(path + "/cross_entropy_loss_apt_data/predict.bin", predictSize, predict, predictSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/target.bin", targetSize, target, targetSize);
    ReadFile(path + "/cross_entropy_loss_apt_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    auto cross_entropy_loss_func = [](GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob,
                                      GM_ADDR zloss, GM_ADDR lse_for_zloss, GM_ADDR workspace, GM_ADDR tiling) {
        cross_entropy_loss<0, 0, 1, 0>(input, target, weight, loss, log_prob, zloss, lse_for_zloss, workspace, tiling);
    };
    ICPU_RUN_KF(
        cross_entropy_loss_func, tilingInfo.blockNum, predict, target, weight, y, y1, y2, y3, workspace,
        tilingInfo.tilingData.get());
    WriteFile(path + "/cross_entropy_loss_apt_data/loss.bin", y, ySize);
    WriteFile(path + "/cross_entropy_loss_apt_data/logprobs.bin", y1, y1Size);

    AscendC::GmFree((void*)predict);
    AscendC::GmFree((void*)target);
    AscendC::GmFree((void*)weight);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)y1);
    AscendC::GmFree((void*)y2);
    AscendC::GmFree((void*)y3);
    AscendC::GmFree((void*)workspace);
}
