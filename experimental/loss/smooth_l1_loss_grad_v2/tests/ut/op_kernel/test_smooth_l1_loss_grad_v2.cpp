/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <filesystem>
#include <string>
#include <cstdint>
#include <fstream>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "any_value.h"

namespace AscendC {
static constexpr auto dec = ::dec;
}

#include "data_utils.h"
#include "tiling_case_executor.h"
#include "../op_host/smooth_l1_loss_grad_v2_tiling.h"
#include "../../../op_kernel/smooth_l1_loss_grad_v2.h"

namespace fs = std::filesystem;

extern "C" __global__ __aicore__ void smooth_l1_loss_grad_v2(GM_ADDR predict,
                                                             GM_ADDR label,
                                                             GM_ADDR dout,
                                                             GM_ADDR gradient,
                                                             GM_ADDR workspace,
                                                             GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SmoothL1LossGradV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SmoothL1LossGradV2TilingData, tilingData, tiling);
    MySmoothL1LossGradV2::KernelSmoothL1LossGradV2<DTYPE_PREDICT, DTYPE_LABEL, DTYPE_DOUT, DTYPE_GRADIENT> op;
    op.Init(predict, label, dout, gradient,
            tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
            tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
            tilingData.tileDataNum, tilingData.smallTailDataNum,
            tilingData.bigTailDataNum, tilingData.tailBlockNum,
            tilingData.totalDataNum, tilingData.doutDataNum,
            tilingData.sigma, tilingData.reduction);
    op.Process();
}

static bool EnsureSmoothL1LossGradV2DataDir()
{
    const fs::path dst = fs::current_path() / "smooth_l1_loss_grad_v2_data";
    if (fs::exists(dst) && !fs::is_directory(dst)) {
        fs::remove(dst);
    }

    const fs::path srcFromFile = fs::path(__FILE__).parent_path() / "smooth_l1_loss_grad_v2_data";
    if (fs::exists(srcFromFile) && fs::is_directory(srcFromFile)) {
        fs::create_directories(dst);
        fs::copy(srcFromFile, dst, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        return true;
    }

    const fs::path cwd = fs::current_path();
    const fs::path candidates[] = {
        cwd / "../../../experimental/math/smooth_l1_loss_grad_v2/tests/ut/op_kernel/smooth_l1_loss_grad_v2_data",
        cwd / "../../../../experimental/math/smooth_l1_loss_grad_v2/tests/ut/op_kernel/smooth_l1_loss_grad_v2_data",
        cwd / "../../../../../experimental/math/smooth_l1_loss_grad_v2/tests/ut/op_kernel/smooth_l1_loss_grad_v2_data",
    };
    for (const auto& src : candidates) {
        if (fs::exists(src) && fs::is_directory(src)) {
            fs::create_directories(dst);
            fs::copy(src, dst, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
            return true;
        }
    }

    std::fprintf(stderr,
                 "[SmoothL1LossGradV2Test] cannot locate smooth_l1_loss_grad_v2_data, cwd=%s, __FILE__=%s\n",
                 fs::current_path().c_str(), __FILE__);
    return false;
}

static std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(float sigma, const std::string& reduction)
{
    using Ops::Math::AnyValue;
    return {
        {"sigma", AnyValue::CreateFrom<float>(sigma)},
        {"reduction", AnyValue::CreateFrom<std::string>(reduction)}
    };
}

class SmoothL1LossGradV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "smooth_l1_loss_grad_v2_test SetUp" << std::endl;
        ASSERT_TRUE(EnsureSmoothL1LossGradV2DataDir());
        system("chmod -R 755 ./smooth_l1_loss_grad_v2_data/");
    }
    static void TearDownTestCase()
    {
        std::cout << "smooth_l1_loss_grad_v2_test TearDown" << std::endl;
    }
};

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

static std::vector<float> ReadGoldenFloat32(const std::string& fileName)
{
    std::ifstream ifs(fileName, std::ios::binary);
    if (!ifs) {
        return {};
    }
    ifs.seekg(0, std::ios::end);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    if (!ifs.read(reinterpret_cast<char*>(data.data()), size)) {
        return {};
    }
    return data;
}

static void ExpectNearRel(float actual, float expected, float rtol, float atol)
{
    float diff = std::fabs(actual - expected);
    float tol = atol + rtol * std::fabs(expected);
    EXPECT_LE(diff, tol);
}

static void ExpectNearRelVector(const std::vector<float>& actual,
                                const std::vector<float>& expected,
                                float rtol, float atol)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        ExpectNearRel(actual[i], expected[i], rtol, atol);
    }
}

TEST_F(SmoothL1LossGradV2Test, test_case_float16_none)
{
    optiling::SmoothL1LossGradV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 0}; // reduction=none
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossGradV2",
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // predict
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // label
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // dout (none 模式)
        },
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // gradient
        },
        MakeAttrs(1.0f, "none"),
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./smooth_l1_loss_grad_v2_data/ && python3 gen_data.py '(128,64)' 'float16' '1.0' 'none'");
    uint32_t dataCount = 128 * 64;
    size_t inputByteSize = dataCount * sizeof(half);

    std::string fileName = "./smooth_l1_loss_grad_v2_data/float16_input_t_predict.bin";
    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, predict, inputByteSize);

    fileName = "./smooth_l1_loss_grad_v2_data/float16_input_t_label.bin";
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, label, inputByteSize);

    fileName = "./smooth_l1_loss_grad_v2_data/float16_input_t_dout.bin";
    uint8_t* dout = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, dout, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* gradient = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(smooth_l1_loss_grad_v2, tilingInfo.blockNum,
                predict, label, dout, gradient, workspace, tiling);

    fileName = "./smooth_l1_loss_grad_v2_data/float16_output_t_gradient.bin";
    WriteFile(fileName, gradient, outputByteSize);

    AscendC::GmFree((void*)(predict));
    AscendC::GmFree((void*)(label));
    AscendC::GmFree((void*)(dout));
    AscendC::GmFree((void*)(gradient));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./smooth_l1_loss_grad_v2_data/ && python3 compare_data.py 'float16'");
}

TEST_F(SmoothL1LossGradV2Test, test_case_float32_none)
{
    optiling::SmoothL1LossGradV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 0};
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossGradV2",
        {
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1.0f, "none"),
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./smooth_l1_loss_grad_v2_data/ && python3 gen_data.py '(256,33)' 'float32' '1.0' 'none'");
    uint32_t dataCount = 256 * 33;
    size_t inputByteSize = dataCount * sizeof(float);

    std::string fileName = "./smooth_l1_loss_grad_v2_data/float32_input_t_predict.bin";
    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, predict, inputByteSize);

    fileName = "./smooth_l1_loss_grad_v2_data/float32_input_t_label.bin";
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, label, inputByteSize);

    fileName = "./smooth_l1_loss_grad_v2_data/float32_input_t_dout.bin";
    uint8_t* dout = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, dout, inputByteSize);

    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* gradient = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(smooth_l1_loss_grad_v2, tilingInfo.blockNum,
                predict, label, dout, gradient, workspace, tiling);

    fileName = "./smooth_l1_loss_grad_v2_data/float32_output_t_gradient.bin";
    WriteFile(fileName, gradient, outputByteSize);

    AscendC::GmFree((void*)(predict));
    AscendC::GmFree((void*)(label));
    AscendC::GmFree((void*)(dout));
    AscendC::GmFree((void*)(gradient));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./smooth_l1_loss_grad_v2_data/ && python3 compare_data.py 'float32'");
}

TEST_F(SmoothL1LossGradV2Test, test_case_float32_mean)
{
    optiling::SmoothL1LossGradV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 2}; // reduction=mean
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossGradV2",
        {
            {{{128, 32}, {128, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128, 32}, {128, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dout 标量
        },
        {
            {{{128, 32}, {128, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1.0f, "mean"),
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./smooth_l1_loss_grad_v2_data/ && python3 gen_data.py '(128,32)' 'float32' '1.0' 'mean'");
    uint32_t dataCount = 128 * 32;
    size_t inputByteSize = dataCount * sizeof(float);
    size_t doutByteSize = sizeof(float); // 标量

    std::string fileName = "./smooth_l1_loss_grad_v2_data/float32_input_t_predict.bin";
    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, predict, inputByteSize);

    fileName = "./smooth_l1_loss_grad_v2_data/float32_input_t_label.bin";
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, label, inputByteSize);

    fileName = "./smooth_l1_loss_grad_v2_data/float32_input_t_dout.bin";
    uint8_t* dout = (uint8_t*)AscendC::GmAlloc(CeilAlign(doutByteSize, 32));
    ReadFile(fileName, doutByteSize, dout, doutByteSize);

    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* gradient = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(smooth_l1_loss_grad_v2, tilingInfo.blockNum,
                predict, label, dout, gradient, workspace, tiling);

    fileName = "./smooth_l1_loss_grad_v2_data/float32_output_t_gradient.bin";
    WriteFile(fileName, gradient, outputByteSize);

    AscendC::GmFree((void*)(predict));
    AscendC::GmFree((void*)(label));
    AscendC::GmFree((void*)(dout));
    AscendC::GmFree((void*)(gradient));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    std::vector<float> golden = ReadGoldenFloat32("./smooth_l1_loss_grad_v2_data/float32_golden_t_gradient.bin");
    ASSERT_FALSE(golden.empty());
    std::vector<float> actual(dataCount);
    std::ifstream ifs("./smooth_l1_loss_grad_v2_data/float32_output_t_gradient.bin", std::ios::binary);
    ifs.read(reinterpret_cast<char*>(actual.data()), outputByteSize);
    ExpectNearRelVector(actual, golden, 1e-5f, 1e-6f);
}

TEST_F(SmoothL1LossGradV2Test, test_case_float32_sum)
{
    optiling::SmoothL1LossGradV2CompileInfo compileInfo = {64, 262144, true, 1.0f, 1}; // reduction=sum
    gert::TilingContextPara tilingContextPara(
        "SmoothL1LossGradV2",
        {
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dout 标量
        },
        {
            {{{64, 64}, {64, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        MakeAttrs(1.0f, "sum"),
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./smooth_l1_loss_grad_v2_data/ && python3 gen_data.py '(64,64)' 'float32' '1.0' 'sum'");
    uint32_t dataCount = 64 * 64;
    size_t inputByteSize = dataCount * sizeof(float);
    size_t doutByteSize = sizeof(float);

    std::string fileName = "./smooth_l1_loss_grad_v2_data/float32_input_t_predict.bin";
    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, predict, inputByteSize);

    fileName = "./smooth_l1_loss_grad_v2_data/float32_input_t_label.bin";
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, label, inputByteSize);

    fileName = "./smooth_l1_loss_grad_v2_data/float32_input_t_dout.bin";
    uint8_t* dout = (uint8_t*)AscendC::GmAlloc(CeilAlign(doutByteSize, 32));
    ReadFile(fileName, doutByteSize, dout, doutByteSize);

    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* gradient = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(smooth_l1_loss_grad_v2, tilingInfo.blockNum,
                predict, label, dout, gradient, workspace, tiling);

    fileName = "./smooth_l1_loss_grad_v2_data/float32_output_t_gradient.bin";
    WriteFile(fileName, gradient, outputByteSize);

    AscendC::GmFree((void*)(predict));
    AscendC::GmFree((void*)(label));
    AscendC::GmFree((void*)(dout));
    AscendC::GmFree((void*)(gradient));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    std::vector<float> golden = ReadGoldenFloat32("./smooth_l1_loss_grad_v2_data/float32_golden_t_gradient.bin");
    ASSERT_FALSE(golden.empty());
    std::vector<float> actual(dataCount);
    std::ifstream ifs("./smooth_l1_loss_grad_v2_data/float32_output_t_gradient.bin", std::ios::binary);
    ifs.read(reinterpret_cast<char*>(actual.data()), outputByteSize);
    ExpectNearRelVector(actual, golden, 1e-5f, 1e-6f);
}